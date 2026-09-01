"""Build the Chroma index from the PDFs in data/.

Reads every PDF in data/, takes each page's text from its text layer where it has
one and from the OCR cache (see src/ocr.py) where it does not, chunks the result,
embeds it, and *writes it to Chroma*.

That last part is the point: the previous version of this script loaded and split
all eleven PDFs and then only opened the vector store without ever calling
add_documents(), so every chunk was discarded and the index kept whatever a much
earlier run had left behind - one document out of eleven.

Usage:

    uv run python src/ingest.py --reset     # rebuild the index from scratch
    uv run python src/ingest.py             # add only sources not yet indexed
    uv run python src/ingest.py --skip-ocr  # text-layer PDFs only

Nine of the eleven PDFs are image scans, so a first full run has to OCR ~4,900
pages. That work is cached per page, so re-running this script afterwards is fast.
"""

import argparse
import os
import sys
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import ocr  # noqa: E402

DATA_DIR = "data"
PERSIST_DIR = "./.chroma_db"
EMBED_BATCH = 500


def load_pages(pdf_path: str, use_ocr: bool = True) -> list[Document]:
    """One Document per page, text layer where available, OCR cache otherwise.

    The decision is per page rather than per document so a partly-scanned PDF
    needs no special handling.
    """
    doc = pymupdf.open(pdf_path)
    total = doc.page_count
    pages: list[Document] = []
    ocr_used = 0

    try:
        for page_no in range(total):
            text = doc[page_no].get_text().strip()
            from_ocr = False
            if not text and use_ocr:
                text = (ocr.cached_text(pdf_path, page_no) or "").strip()
                from_ocr = bool(text)
                ocr_used += from_ocr
            if not text:
                continue
            pages.append(Document(
                page_content=text,
                # Only `source` and `page` are read downstream - by
                # check_citations() in src/agent.py, the API's source panel, and
                # the eval's gold labels. `source` keeps os.path.join's native
                # separator so it matches the values already in the index.
                metadata={
                    "source": pdf_path,
                    "file_path": pdf_path,
                    "page": page_no,
                    "total_pages": total,
                    "ocr": from_ocr,
                },
            ))
    finally:
        doc.close()

    suffix = f" ({ocr_used} via OCR)" if ocr_used else ""
    print(f"  {Path(pdf_path).name}: {len(pages)}/{total} pages with text{suffix}")
    return pages


def indexed_sources(vectorstore) -> set[str]:
    existing = vectorstore.get(include=["metadatas"])
    return {m.get("source") for m in existing["metadatas"] if m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true",
                    help="Delete the existing collection and rebuild from scratch. "
                         "Use this once, so every document shares one metadata format.")
    ap.add_argument("--skip-ocr", action="store_true",
                    help="Index only the text that is already available (text layers "
                         "plus whatever the OCR cache already holds); run no new OCR.")
    ap.add_argument("--workers", type=int, default=None,
                    help="OCR worker processes (default: one per core).")
    args = ap.parse_args()

    pdf_paths = [
        os.path.join(DATA_DIR, p.name)
        for p in sorted(Path(DATA_DIR).glob("*.pdf"))
    ]
    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}/", file=sys.stderr)
        return 1
    print(f"Found {len(pdf_paths)} PDFs in {DATA_DIR}/\n")

    # OCR first, as its own phase: it is the slow part and it is cached, so doing
    # it up front means an interrupted run loses no OCR work.
    if not args.skip_ocr:
        needing = [(p, ocr.pages_needing_ocr(p)) for p in pdf_paths]
        outstanding = sum(
            1 for path, pages in needing for pg in pages
            if not ocr.cache_path(path, pg).exists()
        )
        if outstanding:
            print(f"OCR: {outstanding} pages still need it. This is the slow phase; "
                  f"progress is cached per page, so interrupting is safe.\n")
            for path, pages in needing:
                if pages:
                    ocr.ocr_document(path, workers=args.workers)
            print()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    if args.reset:
        ids = vectorstore.get()["ids"]
        if ids:
            print(f"Reset: deleting {len(ids)} existing chunks\n")
            vectorstore.delete(ids=ids)
        already: set[str] = set()
    else:
        already = indexed_sources(vectorstore)
        if already:
            print(f"Already indexed: {', '.join(sorted(Path(s).name for s in already))}\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print("Loading and chunking...")
    total_added = 0
    for pdf_path in pdf_paths:
        if pdf_path in already:
            print(f"  {Path(pdf_path).name}: already indexed, skipping")
            continue

        pages = load_pages(pdf_path, use_ocr=True)
        if not pages:
            print(f"  {Path(pdf_path).name}: no text recovered, skipping")
            continue

        chunks = splitter.split_documents(pages)
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            vectorstore.add_documents(batch)
            print(f"    embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)} chunks",
                  flush=True)
        total_added += len(chunks)

    final = vectorstore.get(include=["metadatas"])
    sources = sorted({m.get("source") for m in final["metadatas"] if m})
    print(f"\nAdded {total_added} chunks this run.")
    print(f"Index now holds {len(final['ids'])} chunks across {len(sources)} sources:")
    for s in sources:
        n = sum(1 for m in final["metadatas"] if m and m.get("source") == s)
        print(f"  {s}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
