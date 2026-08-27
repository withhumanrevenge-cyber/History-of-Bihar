# History of Bihar

History of Bihar is a document question-answering application for exploring Bihar's history. It combines a FastAPI backend, a Next.js frontend, local PDF ingestion, vector retrieval, and a Hugging Face language model. Answers are streamed to the browser and include the source documents and page numbers used by retrieval.

## Features

- Ask questions about the Bihar history PDF collection.
- Retrieve relevant passages with Chroma and BM25-based retrieval.
- Stream generated answers through Server-Sent Events.
- Show source filenames, page numbers, and excerpts with each answer.
- Use the Hugging Face Router when `HF_TOKEN` is available, with a local model fallback.

## Project Structure

```text
src/
  api.py          FastAPI application and streaming endpoints
  chain.py        RAG prompt and chain construction
  config.py       Environment configuration
  ingest.py       PDF loading, chunking, embeddings, and Chroma indexing
  llm.py          Hugging Face Router and local model setup
  main.py         RAG helpers and command-line interface
  retrieval.py    Retriever construction
frontend/
  app/            Next.js user interface
data/             Local PDF files used for ingestion
```

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 20 or newer and npm
- A Hugging Face access token for hosted generation, or enough local resources for the fallback model

## Setup

1. Install Python dependencies:

	```powershell
	uv sync
	```

2. Create a `.env` file in the project root:

	```env
	HF_TOKEN=your_huggingface_token
	HF_CHAT_MODEL=meta-llama/Llama-3.3-70B-Instruct
	```

	`HF_TOKEN` is required for the hosted Hugging Face Router path. The application falls back to `Qwen/Qwen2.5-1.5B-Instruct` when it is not set.

3. Put the source PDFs in `data/` with these names:

	```text
	data/bihar01.pdf
	data/bihar02.pdf
	data/bihar03.pdf
	```

	The PDFs are intentionally not tracked in Git because the collection includes files larger than GitHub's 100 MB file limit.

4. Build the Chroma index:

	```powershell
	uv run python src/ingest.py
	```

5. Start the backend:

	```powershell
	uv run uvicorn src.api:app --reload --host 127.0.0.1 --port 8001
	```

6. In a second terminal, start the frontend:

	```powershell
	cd frontend
	npm install
	npm run dev
	```

	Open [http://localhost:3000](http://localhost:3000). The frontend proxies API requests to the backend through its Next.js rewrites.

## API

### Health check

```http
GET /api/health
```

### Ask a question

```http
POST /api/chat
Content-Type: application/json

{"question":"What role did Bihar play in the Revolt of 1857?"}
```

The response is a streamed `text/event-stream` containing `sources`, `token`, `error`, and `done` events.

## Configuration

The model is configured in `src/llm.py`:

- `temperature=0.5` controls response variation.
- `max_tokens=800` limits generated response length.
- `top_p=0.9` controls token sampling.

For factual PDF answers, lower temperature values such as `0.2` to `0.3` are generally preferable.

## Command-Line Mode

After indexing, the RAG chain can also be used from the terminal:

```powershell
uv run python src/main.py
```

Type `exit` or `quit` to stop.

## Notes

- The generated `.chroma_db/` directory is local runtime data and can be rebuilt with the ingestion command.
- Keep `.env` out of version control and never commit API tokens.
