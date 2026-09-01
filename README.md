# Bihar Insights

A citation-aware, document-grounded Q&A app over Bihar's historical records — a history
of the freedom movement plus nine district gazetteers, **5,791 pages in total**. Every
answer cites the source PDF and page it came from, and those citations are verified
against the index before they reach you.

```
FastAPI + SSE   ·   Chroma + BM25 hybrid retrieval + cross-encoder rerank
LangChain tool-calling agent   ·   RapidOCR pipeline   ·   Next.js frontend(with help of claude and ui/ux-promax skill)
```

The goal is simple: answer only from the indexed documents, cite the supporting pages
inline, and say "I don't know" rather than guess.

## Why this project exists

This app is a side project of mine where i was experimenting and exploring with rag
architectures, guardrails, middlewares and better llms for generation , i created the
frontend with help of claude and other resources that i have (not so fond of frontend
work). explore the codebase or application , i would love any feedback .

It reduces hallucinations by:

- retrieving relevant passages before answering
- forcing the model to answer only from indexed sources
- validating citations against the document index
- refusing unsupported answers when evidence is missing

## The corpus

| | |
| --- | --- |
| Documents | 11 PDFs |
| Pages | 5,791 — all indexed |
| Chunks | 15,773 |
| Needed OCR | 4,942 pages (9 of the 11 PDFs are image scans with no text layer) |

The books are mid-century Indian print: a history of the Bihar freedom movement, and
district gazetteers for Patna, Gaya, Muzaffarpur and others. Expect period prose and
some OCR noise.

**Questions it answers well** — the 1857 revolt and Kunwar Singh, the Birsa Munda
uprising, the Non-cooperation and Home Rule movements, and district-level agriculture,
irrigation, rents, industries and administration.

**Questions it will honestly refuse** — anything needing modern census data, current
welfare schemes, or recent statistics. None of that is in these documents, and the app is
built to say so rather than improvise.

## Architecture

```text
data/*.pdf
    |
    +-- has a text layer? ---> PyMuPDF extraction
    +-- image scan?       ---> src/ocr.py  (RapidOCR, cached per page)
    |
    v
Per-page chunking (1000 chars, 200 overlap) - chunks never cross a page boundary
    |
    v
all-MiniLM-L6-v2 embeddings ---> Chroma
    |
    +--> semantic retrieval (k=5, fetch_k=20)
    +--> BM25 retrieval
            |
            v
      dedupe + cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
            |
            v
  LangGraph agent - 8 tools, model/tool call-limit middleware
            |
            v
  post-hoc groundedness + citation verification
            |
            v
     FastAPI SSE  --->  Next.js frontend
```

Chunking is deliberately **page-scoped**: a chunk never spans two pages, which is what
makes `[source: file, page N]` citations exact and lets every claim be checked against one
specific page.

## File layout

```text
src/
  agent.py         agent, 8 tools, injection filter, citation checker
  api.py           FastAPI endpoints, SSE streaming, rate limiting, guardrails
  config.py        startup credential check
  ingest.py        PDF -> text/OCR -> chunks -> Chroma
  llm.py           provider selection (Groq / HF / Llama / local fallback)
  main.py          CLI runner + vector store loaders
  ocr.py           parallel, cached, resumable OCR for scanned PDFs
  retrieval.py     hybrid search and reranking

evals/             evaluation harnesses and dataset

frontend/app/      Next.js UI (proxies /api/* to the backend)

data/*.pdf         source documents      (gitignored - large)
.chroma_db/        vector index          (generated)
.ocr_cache/        per-page OCR text     (generated)
```

## Quick start

Requires Python 3.10+ with [`uv`](https://docs.astral.sh/uv/), and Node 20+. No system OCR
install is needed — RapidOCR ships as a pip wheel.

```powershell
uv sync
copy .env.example .env       # then add a GROQ_API_KEY
```

`.env.example` documents every variable the code reads. Provider precedence is
**Groq → HuggingFace → Llama API → local fallback**; a blank value counts as unset. With no
key at all the app runs `Qwen2.5-1.5B-Instruct` on CPU — fine for a smoke test, far too
slow for real use.

Put your PDFs in `data/`, then build the index:

```powershell
uv run python src/ingest.py --reset
```

This reads each PDF page by page, OCRs anything without a text layer, chunks, embeds, and
writes to Chroma.

> **The first run is slow if your PDFs are scans.** OCR runs at roughly **12 pages/min** on
> a 12-core laptop CPU and saturates every core — the 4,942 scanned pages here took about
> 8 hours. It is CPU-bound, not memory-bound, so more RAM will not help, and on a thin
> laptop it is thermally bound, so mains power barely helps either.
>
> Every page is cached to `.ocr_cache/`, so Ctrl-C costs only the page in flight and
> re-running skips everything already done. See [OCR](#ocr) for running it in batches.

Start the backend:

```powershell
uv run uvicorn src.api:app --host 127.0.0.1 --port 8001
```

Cold start is about **17 seconds** at 15,773 chunks — embeddings, Chroma, BM25 and the
cross-encoder all load before it binds. Wait for `Application startup complete`.

> **Do not add `--reload`.** `/api/chat` is a long-lived SSE stream and the reloader kills
> the worker on every `.py` save — the browser then shows a bare `network error` with
> nothing in the server log. On Windows it also leaves orphaned children holding port 8001,
> so the next start fails with `[Errno 10048]`. To clear them:
>
> ```powershell
> Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
>   Where-Object { $_.CommandLine -like '*uvicorn*' -or $_.CommandLine -like '*multiprocessing*' } |
>   ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
> ```

Then the frontend, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The dev server proxies `/api/*` to `127.0.0.1:8001`; override
with `NEXT_PUBLIC_API_ORIGIN` in `frontend/.env.local`.

## OCR

`ingest.py` handles OCR automatically, but you can drive it directly:

```powershell
uv run python src/ocr.py                      # everything that needs it
uv run python src/ocr.py --max-pages 300      # a slice of each document
uv run python src/ocr.py data/bihar05.pdf     # one document
```

`--max-pages N` caps pages **per document per run**, which lets you go breadth-first — a
slice of every book makes all sources searchable far sooner than finishing one book at a
time, for the same total CPU. Re-run with a larger value to go deeper.

Two implementation notes that matter for speed:

- **One ONNX thread per worker, one worker per core.** Thread parallelism inside a single
  OCR call scales badly (12.7s/page at 1 thread vs 7.7s at 8 — only 1.65x for 8x the
  cores), and 16 threads is actually *slower* than 1 through oversubscription.
- **Pages render at the scan's native resolution.** RapidOCR downscales large inputs
  internally, so higher DPI costs almost nothing and fixes word-boundary errors on
  low-resolution scans.

After any OCR run, re-index with `uv run python src/ingest.py --reset --skip-ocr`.
`--skip-ocr` matters — without it, `ingest.py` starts *another* OCR job.

## API

**`GET /api/health`** → `{"status": "ok", "ready": true}`

**`POST /api/chat`** with `{"question": "..."}` → `text/event-stream`

| Event | Payload |
| --- | --- |
| `sources` | retrieved passages with source, page and excerpt |
| `token` | a piece of the answer text |
| `usage` | prompt/completion token counts |
| `notice` | a guardrail or groundedness message |
| `error` | generation failed |
| `done` | stream finished |

The stream also emits `: ping` comment frames every 10 seconds. These are not decoration —
the agent can work for over a minute before it has anything to say, and without bytes on
the wire an idle proxy will drop a connection that is perfectly healthy. Clients should
ignore any frame with no `data:` line, and treat a stream that ends **without** `done` as a
dropped connection rather than a finished answer.

## Grounding and guardrails

Before the agent runs: a 500-character question cap, a per-IP sliding-window rate limit
(20/min), and heuristic prompt-injection detection.

After it replies (`_enforce_groundedness` in `src/api.py`):

- every `[source: file, page N]` citation is checked against the live index
- citations whose page does not exist, or whose claim shares too little vocabulary with
  that page, are stripped and a `notice` is emitted
- if no retrieval tool was called at all, the answer is replaced with the fallback

```text
The literacy rate rose over the reported period [source: report.pdf, page 24].
```

The cited page is the **PDF page index**, not the number printed in the book — the offset
varies because of unnumbered plates. It is correct in a PDF viewer, which navigates by
index.

## Token budget

Worth understanding before you use the app in earnest, because it is a ceiling rather
than a tuning problem.

One question costs roughly **24,000 tokens** — five model calls whose context
accumulates, of which about 37% is the eight tool schemas being resent on every call.

The Groq free tier allows **8,000 tokens/minute** and **200,000 tokens/day**, which works
out to roughly **8 questions per day**. Any sustained use needs a paid tier.

The daily limit is reported **only in the body of a 429**, never in the `x-ratelimit-*`
headers, which makes it easy to misdiagnose as a per-minute pacing problem. If you hit
it: use a paid tier, ask fewer questions, or cut tokens per run.

## Troubleshooting

**`network error` in the chat UI while `/api/health` is fine.** The response stream was
severed after headers — usually `--reload` restarting the worker mid-stream.

**`[Errno 10048] only one usage of each socket address`.** An orphaned process still holds
port 8001; use the cleanup command above, or run on `--port 8002`.

**Empty answer, or "the agent reached its tool/model call limit".** The agent spent its five
model calls on tool calls without writing an answer. Ask something narrower, or raise the
limits in `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` in `src/agent.py`.

**Retrieval only returns one document.** The index is out of date with the OCR cache — run
`uv run python src/ingest.py --reset --skip-ocr`.

**401 / no API key.** Check what is actually loaded:

```powershell
uv run python -c "import os;from dotenv import load_dotenv;load_dotenv();print({k:bool((os.getenv(k) or '').strip()) for k in ['GROQ_API_KEY','HF_TOKEN','LLAMA_API_KEY']})"
```

**Check what is in the index:**

```powershell
uv run python -c "import sqlite3,collections;c=sqlite3.connect('.chroma_db/chroma.sqlite3').cursor();c.execute(\"select string_value from embedding_metadata where key='source'\");print(collections.Counter(r[0] for r in c.fetchall()))"
```

