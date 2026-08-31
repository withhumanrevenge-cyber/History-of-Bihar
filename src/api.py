import asyncio
import json
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent import CITATION_RE, NO_CONTEXT_MSG, build_agent, check_citations, detect_injection


state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["agent"], state["chunks"] = build_agent()
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


MAX_QUESTION_LEN = 500
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# Per-client sliding-window rate limit. In-memory and per-process, which is fine
# for a single-instance dev/demo deployment; a multi-worker deployment would need
# a shared store (e.g. Redis) instead.
_request_log: dict[str, deque] = {}


def _rate_limited(client_id: str) -> bool:
    now = time.monotonic()
    log = _request_log.setdefault(client_id, deque())
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    log.append(now)
    return False


def sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


# The agent's model node returns the full final answer in one shot (see chat()
# below), so this fakes a per-token stream for the UI's typing effect by
# doling the text out word by word with a small delay between chunks.
TOKEN_STREAM_DELAY = 0.0


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


def _extract_usage(message) -> dict | None:
    """Best-effort extraction of token usage from a LangChain message."""
    metadata = getattr(message, "response_metadata", None) or {}
    usage = (
        metadata.get("usage_metadata")
        or metadata.get("token_usage")
        or metadata.get("usage")
        or getattr(message, "usage_metadata", None)
    )

    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and "input_tokens" in usage:
        prompt_tokens = usage.get("input_tokens")
    if completion_tokens is None and "output_tokens" in usage:
        completion_tokens = usage.get("output_tokens")
    if total_tokens is None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    fields = {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }
    if all(v is None for v in fields.values()):
        return None
    return fields


def _enforce_groundedness(text: str, sources: list[dict], chunks) -> tuple[str, list[str]]:
    """Post-hoc check on the agent's final answer: strips citations that don't
    match a real (source, page) pair in the index, and replaces the whole answer
    with the standard fallback if no retrieval tool was ever called for a
    non-trivial answer. Returns (possibly-rewritten text, user-facing notices)."""
    if not text.strip() or text.strip() == NO_CONTEXT_MSG:
        return text, []

    notices = []
    findings = check_citations(text, chunks)
    bad_pairs = {(f["source"], f["page"]) for f in findings if f.get("cited") and not f.get("supported")}
    if bad_pairs:
        def _mask(m: re.Match) -> str:
            pair = (m.group(1).strip(), m.group(2).strip())
            return "[unverified citation removed]" if pair in bad_pairs else m.group(0)

        text = CITATION_RE.sub(_mask, text)
        notices.append("One or more citations could not be verified against the document index and were removed.")

    if not sources:
        text = NO_CONTEXT_MSG
        notices.append("Answer blocked: no supporting passages were retrieved for this question.")

    return text, notices


@app.post("/api/chat")
async def chat(q: Query, request: Request):
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(status_code=400, detail=f"Question too long (max {MAX_QUESTION_LEN} characters).")

    client_id = request.client.host if request.client else "unknown"
    if _rate_limited(client_id):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")

    if detect_injection(question):
        async def refuse() -> AsyncIterator[bytes]:
            yield sse("notice", {"message": "Your message was blocked by the input safety filter."})
            for piece in _typing_chunks(NO_CONTEXT_MSG):
                yield sse("token", {"text": piece})
            yield sse("done", {})

        return StreamingResponse(refuse(), media_type="text/event-stream")

    agent = state["agent"]
    chunks = state["chunks"]

    async def stream() -> AsyncIterator[bytes]:
        seen: set[tuple] = set()
        sources: list[dict] = []
        usage_emitted = False
        try:
            # create_agent's model node calls the LLM with a single ainvoke() per
            # turn rather than streaming tokens, so "updates" mode gives us one
            # event per completed tool call / model turn, not per token.
            async for update in agent.astream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="updates",
            ):
                for node_name, node_output in update.items():
                    if node_name not in ("model", "tools") or not node_output:
                        continue
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
                            usage = _extract_usage(message)
                            if usage and not usage_emitted:
                                usage_emitted = True
                                yield sse("usage", usage)

                            text, notices = _enforce_groundedness(text, sources, chunks)
                            for notice in notices:
                                yield sse("notice", {"message": notice})
                            for piece in _typing_chunks(text):
                                yield sse("token", {"text": piece})
                                await asyncio.sleep(TOKEN_STREAM_DELAY)
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
        yield sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")
