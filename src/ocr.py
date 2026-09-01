"""OCR for the scanned PDFs in data/.

Nine of the eleven source PDFs are pure image scans with no text layer at all, so
PyMuPDF extracts nothing from them. This module rasterises those pages and runs
RapidOCR (ONNX Runtime) over them.

Two things drive the design:

*Every page is cached to disk.* The full corpus is ~4,900 scanned pages and takes
over an hour, so an interrupted run must not start over. Results land in
`.ocr_cache/<pdf stem>/<page>.txt` and re-runs skip whatever is already there.

*One ONNX thread per worker, many workers.* Benchmarked on this machine, a single
OCR call scales badly with threads (12.7s/page at 1 thread vs 7.7s at 8 - only
1.65x for 8x the cores) and 16 threads is actually slower than 1 through
oversubscription. Running one thread per process and one process per core is
roughly 17x faster end to end than the library's default of "all cores, one page
at a time".

Usage:

    uv run python src/ocr.py                 # every PDF in data/ that needs it
    uv run python src/ocr.py data/bihar05.pdf
    uv run python src/ocr.py --workers 8
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".ocr_cache"

# RapidOCR's detector downscales large inputs internally, so rendering above the
# scan's own resolution costs time without adding detail. Render at native
# resolution, bounded: below MIN the text is too small to segment reliably (word
# boundaries start disappearing), above MAX we are just pushing pixels.
MIN_DPI = 200
MAX_DPI = 400

_engine = None
_docs: dict = {}


# --------------------------------------------------------------------------- #
# Worker setup
# --------------------------------------------------------------------------- #

def _thread_limited_config() -> str:
    """Write a RapidOCR config pinned to a single ONNX thread.

    The stock config uses intra_op_num_threads: -1 (every core), which makes a
    pool of workers fight over the same cores. Derived from the installed config
    rather than vendored, so it tracks the library version.
    """
    import rapidocr_onnxruntime
    import yaml

    src = Path(rapidocr_onnxruntime.__file__).parent / "config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    for section in ("Global", "Det", "Cls", "Rec"):
        if section in cfg:
            cfg[section]["intra_op_num_threads"] = 1
            cfg[section]["inter_op_num_threads"] = 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / "rapidocr_1thread.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(out)


def _init_worker(config_path: str) -> None:
    # Belt and braces: these are read by some ONNX Runtime builds at import time,
    # so set them before the library is loaded in this process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ORT_NUM_THREADS", "1")
    global _engine
    from rapidocr_onnxruntime import RapidOCR

    _engine = RapidOCR(config_path=config_path)


def _get_doc(path: str):
    """Keep each worker's PDF handle open. PyMuPDF maps lazily, but reopening a
    493MB file once per page still costs real time."""
    if path not in _docs:
        _docs[path] = pymupdf.open(path)
    return _docs[path]


# --------------------------------------------------------------------------- #
# Rendering + OCR
# --------------------------------------------------------------------------- #

def _native_dpi(page) -> int:
    """DPI that reproduces the embedded scan at its own resolution."""
    images = page.get_images(full=True)
    if not images or not page.rect.width:
        return 300
    native_px = images[0][2]
    dpi = int(72 * native_px / page.rect.width)
    return max(MIN_DPI, min(dpi, MAX_DPI))


def cache_path(pdf_path: str | Path, page_no: int) -> Path:
    return CACHE_DIR / Path(pdf_path).stem / f"{page_no:05d}.txt"


def _ocr_page(args) -> tuple[int, int]:
    pdf_path, page_no = args
    out = cache_path(pdf_path, page_no)
    if out.exists():
        return page_no, -1  # -1 marks "already cached"

    page = _get_doc(pdf_path)[page_no]
    pixmap = page.get_pixmap(dpi=_native_dpi(page))
    result, _ = _engine(pixmap.tobytes("png"))
    text = "\n".join(line[1] for line in result) if result else ""

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file so an interrupt cannot leave a half-written page that
    # a later run would treat as complete.
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    return page_no, len(text)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def pages_needing_ocr(pdf_path: str | Path) -> list[int]:
    """Page numbers with no extractable text layer."""
    doc = pymupdf.open(pdf_path)
    try:
        return [i for i in range(doc.page_count) if not doc[i].get_text().strip()]
    finally:
        doc.close()


def cached_text(pdf_path: str | Path, page_no: int) -> str | None:
    path = cache_path(pdf_path, page_no)
    return path.read_text(encoding="utf-8") if path.exists() else None


def ocr_document(pdf_path: str | Path, workers: int | None = None,
                 progress_every: int = 25, max_pages: int | None = None,
                 skip_front: int = 0) -> int:
    """OCR every page of `pdf_path` that lacks a text layer, caching each page.

    Returns the number of pages OCR'd in this call (already-cached pages are
    skipped and not counted). Safe to interrupt and re-run.

    `max_pages` caps how many uncached pages this call will do, which is what
    makes breadth-first passes possible: a slice of all eleven documents makes
    every source reachable by retrieval far sooner than finishing them one at a
    time. Re-running with a larger cap continues where this left off.

    `skip_front` ignores the first N pages of the document. Front matter (title
    pages, blank leaves, tables of contents) OCRs poorly and carries almost no
    retrievable content, so it is a bad use of the first pass.
    """
    pdf_path = str(pdf_path)
    todo = [
        p for p in pages_needing_ocr(pdf_path)
        if p >= skip_front and not cache_path(pdf_path, p).exists()
    ]
    if max_pages is not None:
        todo = todo[:max_pages]
    if not todo:
        return 0

    workers = workers or os.cpu_count() or 4
    config_path = _thread_limited_config()
    name = Path(pdf_path).name
    print(f"  {name}: OCR {len(todo)} pages on {workers} workers", flush=True)

    done = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(config_path,)) as pool:
        for _page_no, n_chars in pool.imap_unordered(
            _ocr_page, [(pdf_path, p) for p in todo], chunksize=4
        ):
            if n_chars >= 0:
                done += 1
            if done and done % progress_every == 0:
                print(f"    {name}: {done}/{len(todo)} pages", flush=True)
    print(f"  {name}: done ({done} pages)", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR the scanned PDFs in data/.")
    ap.add_argument("pdfs", nargs="*", help="PDFs to process (default: all of data/*.pdf)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker processes (default: one per core).")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Cap uncached pages per document this run. Use this for a "
                         "breadth-first pass: a slice of every document makes all "
                         "sources searchable far sooner than finishing one at a time. "
                         "Re-run with a larger value to go deeper.")
    ap.add_argument("--skip-front", type=int, default=0,
                    help="Ignore the first N pages of each document (front matter OCRs "
                         "poorly and carries little retrievable content).")
    args = ap.parse_args()

    targets = [Path(p) for p in args.pdfs] if args.pdfs else sorted((ROOT / "data").glob("*.pdf"))
    if not targets:
        print("No PDFs found.", file=sys.stderr)
        return 1

    total = 0
    for pdf in targets:
        needed = pages_needing_ocr(pdf)
        if not needed:
            print(f"  {pdf.name}: has a text layer, skipping")
            continue
        total += ocr_document(pdf, workers=args.workers,
                              max_pages=args.max_pages, skip_front=args.skip_front)

    print(f"\nOCR'd {total} pages this run. Cache: {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
