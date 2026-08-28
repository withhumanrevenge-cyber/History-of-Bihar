import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent import build_agent


state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["agent"] = build_agent()
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


# The agent's model node returns the full final answer in one shot (see chat()
# below), so this fakes a per-token stream for the UI's typing effect by
# doling the text out word by word with a small delay between chunks.
TOKEN_STREAM_DELAY = 0.02


def _typing_chunks(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


# Tools whose JSON output contains retrieved passages worth surfacing as sources.
SOURCE_TOOLS = {"search_documents", "search_by_topic", "compare_documents", "build_timeline"}


def _passages_from_tool_output(tool_name: str, content: str) -> list[dict]:
    if tool_name not in SOURCE_TOOLS:
        return []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []

    # compare_documents returns {subject: [passages]}; the others return [passages] directly.
    groups = data.values() if isinstance(data, dict) else [data]
    passages = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict) or "source" not in item:
                continue
            passages.append({
                "source": item.get("source", "unknown"),
                "page": item.get("page"),
                "excerpt": (item.get("content") or item.get("text") or "").strip()[:320],
            })
    return passages


@app.post("/api/chat")
async def chat(q: Query):
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    agent = state["agent"]

    async def stream() -> AsyncIterator[bytes]:
        seen: set[tuple] = set()
        sources: list[dict] = []
        try:
            # create_agent's model node calls the LLM with a single ainvoke() per
            # turn rather than streaming tokens, so "updates" mode gives us one
            # event per completed tool call / model turn, not per token.
            async for update in agent.astream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="updates",
            ):
                for node_name, node_output in update.items():
                    for message in node_output.get("messages", []):
                        if node_name == "tools":
                            content = getattr(message, "content", None)
                            tool_name = getattr(message, "name", None)
                            if not isinstance(content, str) or not tool_name:
                                continue
                            added = False
                            for passage in _passages_from_tool_output(tool_name, content):
                                key = (passage["source"], passage["page"])
                                if key in seen:
                                    continue
                                seen.add(key)
                                sources.append({"id": len(sources) + 1, **passage})
                                added = True
                            if added:
                                yield sse("sources", {"sources": sources})
                        elif node_name == "model" and not getattr(message, "tool_calls", None):
                            text = getattr(message, "content", None) or ""
                            for piece in _typing_chunks(text):
                                yield sse("token", {"text": piece})
                                await asyncio.sleep(TOKEN_STREAM_DELAY)
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
        yield sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")
