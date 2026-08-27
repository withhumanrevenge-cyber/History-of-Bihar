import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.chain import SYSTEM_PROMPT, USER_PROMPT, format_docs
from src.llm import get_llm
from src.main import load_chunks_from_chroma, load_vectorstore
from src.retrieval import create_retriever


state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    vectorstore = load_vectorstore()
    chunks = load_chunks_from_chroma(vectorstore)
    if not chunks:
        raise RuntimeError("Chroma is empty. Run `python src/ingest.py` first.")

    retriever = create_retriever(vectorstore, chunks, k=5, fetch_k=20)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    state["retriever"] = retriever
    state["llm"] = llm
    state["prompt"] = prompt
    yield
    state.clear()


app = FastAPI(title="Bihar Insights API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": bool(state)}


def sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


@app.post("/api/chat")
async def chat(q: Query):
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    retriever = state["retriever"]
    llm = state["llm"]
    prompt = state["prompt"]

    docs = retriever(question)
    sources = [
        {
            "id": i + 1,
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "excerpt": (d.page_content or "").strip()[:320],
        }
        for i, d in enumerate(docs)
    ]

    async def stream() -> AsyncIterator[bytes]:
        yield sse("sources", {"sources": sources})
        try:
            messages = prompt.format_messages(
                context=format_docs(docs), question=question
            )
            async for chunk in llm.astream(messages):
                text = getattr(chunk, "content", None) or ""
                if text:
                    yield sse("token", {"text": text})
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
        yield sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")
