# History of Bihar

History of Bihar is a citation-aware document question-answering application for exploring Bihar's history and public reports. It combines a FastAPI backend, a Next.js interface, local PDF ingestion, hybrid retrieval, and a streaming language-model response.

Answers are grounded in the indexed documents and include the source filename, page number, and excerpt used for retrieval.

## Highlights

- Hybrid search using Chroma semantic retrieval and BM25 keyword retrieval.
- Cross-encoder reranking for the final context selection.
- Streaming answers over Server-Sent Events (SSE).
- Source citations and excerpts displayed with every response.
- Hugging Face Router support with a local-model fallback.
- Responsive Next.js chat interface with health status and suggested questions.

## Metrics

| Metric | Current value |
| --- | ---: |
| Source PDFs | 3 |
| Local source corpus | ~541 MiB |
| Retrieval candidates | 20 |
| Reranked context documents | 5 |
| Chunk size / overlap | 1,000 / 200 characters |
| API routes | 2 |
| Stream event types | 4 |

The repository does not currently include a labeled evaluation set or benchmark run. Accuracy, faithfulness, retrieval recall, and response latency should be measured with a representative question set before being reported as quality metrics.

## Architecture

```text
PDF files in data/
				|
				v
	PyMuPDF + recursive chunking
				|
				v
 Chroma embeddings index
				|
				+--> semantic retrieval --+
				+--> BM25 retrieval ------+--> deduplicate --> cross-encoder rerank
																												|
																												v
																						 grounded prompt + LLM stream
																												|
																												v
																							FastAPI SSE --> Next.js UI
```

## Project layout

```text
src/
	api.py          FastAPI application and SSE endpoints
	chain.py        Grounded prompt and RAG chain helpers
	config.py       Environment configuration
	ingest.py       PDF loading, chunking, embeddings, and indexing
	llm.py          Hugging Face Router and local model setup
	main.py         CLI RAG runner
	retrieval.py    Hybrid retrieval and reranking
frontend/
	app/            Next.js user interface
data/             Local PDF files, supplied separately
```

## Requirements

- Python 3.10-3.13. Python 3.10 is recommended for the current `onnxruntime` dependency.
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 20 or newer and npm
- A Hugging Face access token for hosted generation, or enough local resources for the fallback model

## Setup

From the repository root:

```powershell
uv sync
```

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
HF_CHAT_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

Place the source files in `data/`:

```text
data/bihar01.pdf
data/bihar02.pdf
data/bihar03.pdf
```

The PDFs are intentionally excluded from Git because the collection includes files larger than GitHub's 100 MB file limit.

Build the local index:

```powershell
uv run python src/ingest.py
```

Start the backend in one terminal:

```powershell
uv run uvicorn src.api:app --host 127.0.0.1 --port 8001
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The Next.js proxy forwards `/api/*` requests to the backend on port `8001`. Set `NEXT_PUBLIC_API_ORIGIN` when using another backend URL.

## API

### Health check

```http
GET /api/health
```

Example response:

```json
{"status":"ok","ready":true}
```

### Ask a question

```http
POST /api/chat
Content-Type: application/json

{"question":"What role did Bihar play in the Revolt of 1857?"}
```

The response uses `text/event-stream` and emits these events:

- `sources`: retrieved documents, pages, and excerpts
- `token`: streamed answer text
- `error`: an error message, when applicable
- `done`: indicates that streaming is complete

## Configuration

The hosted model and generation parameters are configured in `src/llm.py`:

- `HF_CHAT_MODEL`: Hugging Face Router model name
- `temperature=0.3`: response variation
- `max_tokens=1200`: maximum generated answer length
- `top_p=0.9`: sampling control

For factual answers, keep temperature low and require citations from the provided context.

## CLI mode

After indexing, run the terminal interface with:

```powershell
uv run python src/main.py
```

Type `exit` or `quit` to stop.

## Runtime files and troubleshooting

- `.chroma_db/` is generated locally and can be rebuilt by running the ingestion command again.
- `.env`, virtual environments, logs, and source PDFs are excluded from Git.
- If port `8001` is already in use, stop the existing Uvicorn process or choose another port and set `NEXT_PUBLIC_API_ORIGIN` for the frontend.
