# Bihar Insights

Bihar Insights is a citation-aware, document-grounded Q&A app for Bihar’s public reports and historical PDFs. It combines:

- a FastAPI backend
- Chroma-based vector retrieval
- hybrid semantic + keyword search
- tool-calling agent logic
- a Next.js frontend
- citation checks against the indexed PDF pages

The goal is simple: answer questions using only the indexed Bihar documents and cite the supporting source pages inline.

## Why this project exists

This app is a side project of mine where i was experimenting and exploring with rag architectures, guardrails, middlewares and better llms for generation , i created the frontend with help of claude and other resources that i have (not so fond of frontend work). explore the codebase or application , i would love any feedback .

you can asks question like:

- agriculture
- welfare and government schemes
- literacy and demographics
- history and geography
- public reports and statistical summaries

It reduces hallucinations by:

- retrieving relevant passages before answering
- forcing the model to answer only from indexed sources
- validating citations against the document index
- refusing unsupported answers when evidence is missing

## Architecture overview

```text
PDF reports in data/
    |
    v
PyMuPDF loader
    |
    v
Recursive chunking
    |
    v
Sentence-transformer embeddings -> Chroma vector store
    |
    +--> semantic retrieval
    +--> BM25 retrieval
            |
            v
      deduplicate + rerank
            |
            v
  LangGraph tool-calling agent
            |
            v
    LLM answer generation
            |
            v
     FastAPI SSE API
            |
            v
       Next.js frontend
```

## File layout

```text
src/
  agent.py         tool-calling agent and guardrails
  api.py           FastAPI endpoints and SSE streaming
  config.py        environment/config helpers
  ingest.py        PDF ingestion, chunking, and Chroma indexing
  llm.py           provider selection for model backend
  main.py          CLI runner for local testing
  retrieval.py     search and reranking logic

frontend/
  app/             Next.js frontend
  package.json     frontend dependencies

public data/
  data/*.pdf       Bihar report PDFs

.chroma_db/
  generated local vector index
```

## Model setup

The app chooses the generation model in this order:

1. Groq, if `GROQ_API_KEY` is set
2. Hugging Face Router, if `HF_TOKEN` is set
3. Llama API, if `LLAMA_API_KEY` is set
4. Local fallback model, only for local development when no cloud provider is configured

The retrieval embedding model is:

- `sentence-transformers/all-MiniLM-L6-v2`

This embedding model is separate from the generation model and is used for semantic search in the vector database.

## Requirements

- Python 3.10+
- `uv`
- Node.js 20+
- npm

## Quick start

### 1) Install dependencies

```powershell
uv sync
```

### 2) Create `.env`

For local development, keep cloud keys empty unless you want to use a hosted provider.

Example local setup:

```env
APP_ENV=development
# keep cloud keys blank for now
```

Example Groq setup:

```env
APP_ENV=development
GROQ_API_KEY=your_key_here
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
```

Example Hugging Face setup:

```env
APP_ENV=development
HF_TOKEN=your_hf_token
HF_CHAT_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

Example Llama API setup:

```env
APP_ENV=development
LLAMA_API_KEY=your_key_here
LLAMA_CHAT_MODEL=llama-3.3-70b-instruct
```

For production, do not rely on local fallback alone. Use a valid cloud key or a properly configured local runtime.

### 3) Add PDFs

Put the Bihar report files in `data/`.

Example:

```text
data/hola.pdf
data/mola.pdf
data/cocacola.pdf
```

### 4) Build the Chroma index

```powershell
uv run python src/ingest.py
```

This loads the PDFs, splits them into chunks, embeds them, and stores them in `.chroma_db/`.

### 5) Start the backend

```powershell
uv run uvicorn src.api:app --host 127.0.0.1 --port 8001
```

### 6) Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## API

### Health check

```http
GET /api/health
```

### Chat endpoint

```http
POST /api/chat
Content-Type: application/json
```

Example request:

```json
{"question":"What welfare schemes are mentioned in the reports?"}
```

The app streams responses over SSE and emits events such as:

- `sources`: retrieved passages with source and page info
- `token`: streamed answer text
- `usage`: token usage metadata if the provider exposes it
- `notice`: groundedness or guardrail notices
- `error`: generation errors
- `done`: stream completion

## Grounding and citations

The app is designed to answer only from the retrieved document context.

Behavior includes:

- inline citation formatting
- verification of source/page pairs against the index
- refusal or fallback answers when the retrieved evidence is weak or missing

Example citation format:

```text
The literacy rate rose over the reported period [source: report.pdf, page 24].
```

## Guardrails included

The project includes a basic safety layer:

- question length cap
- rate limiting
- input injection detection
- groundedness checks
- citation verification
- no-context fallback for unsupported answers

## Troubleshooting

### 401 / no API key provided

This usually means the app is trying to call a cloud provider but no valid API key is available in the active environment.

Check which keys are set:

```powershell
uv run python -c "import os; print('GROQ', bool((os.getenv('GROQ_API_KEY') or '').strip())); print('HF', bool((os.getenv('HF_TOKEN') or '').strip())); print('LLAMA', bool((os.getenv('LLAMA_API_KEY') or '').strip()))"
```

If all are empty, the app will fall back to the local model path for development.

### Local fallback fails to load

This can happen if the fallback model is not a compatible instruct/chat model for the pipeline used by the app.

Common fixes:

- use a valid cloud provider key
- switch to a proper instruct model or Ollama-based local Llama
- avoid relying on fallback-only mode in production

### Port already in use

If port 8001 is busy:

```powershell
uv run uvicorn src.api:app --host 127.0.0.1 --port 8002
```

### Chroma index stale or missing

Rebuild the index:

```powershell
uv run python src/ingest.py
```

## Recommended workflow

For local development:

1. keep cloud keys blank if you want to test the local fallback
2. add your PDFs
3. build the index
4. start the app
5. test a few prompts

For deployment:

1. set a real provider key
2. keep `APP_ENV=production`
3. do not depend on a local fallback-only setup in a production environment

## Notes

- PDF files are often excluded from Git because of size limits.
- This project is meant for research / document-grounded Q&A and should not be treated as a general-purpose web search engine.
- The local fallback is useful for development, but production should use a proper model backend.
