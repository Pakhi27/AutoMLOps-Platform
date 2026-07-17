"""ML Chatbot API — multi-provider LLM + rule-based fallback."""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.core.logging_config import get_logger
from app.services.llm_provider import chat_supports_streaming, get_chat_status, resolve_llm_provider
from app.services.ml_chatbot_agent import get_chat_agent
from app.services.rules_chatbot import handle_rules_chat

router = APIRouter(prefix="/chat", tags=["chatbot"])
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = None


async def _sse_stream(message: str, thread_id: str) -> AsyncIterator[str]:
    def fmt(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

    yield fmt("start", {"thread_id": thread_id, "provider": resolve_llm_provider()})

    if resolve_llm_provider() == "rules":
        try:
            response = handle_rules_chat(message)
            yield fmt("token", {"content": response})
            yield fmt("done", {})
        except Exception as exc:
            logger.exception("Rules chat failed")
            yield fmt("error", {"message": str(exc)})
        return

    agent = get_chat_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async for chunk_event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = chunk_event["event"]
            if kind == "on_chat_model_stream":
                chunk = chunk_event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield fmt("token", {"content": chunk.content})
            elif kind == "on_tool_start":
                yield fmt("tool_call", {"tool": chunk_event["name"], "input": chunk_event["data"].get("input")})
            elif kind == "on_tool_end":
                output = chunk_event["data"].get("output")
                if isinstance(output, ToolMessage):
                    output = output.content
                yield fmt("tool_result", {"tool": chunk_event["name"], "output": output})
        yield fmt("done", {})
    except Exception as exc:
        logger.exception("Chat stream failed")
        yield fmt("error", {"message": str(exc)})


@router.get("/status")
async def chat_status() -> dict[str, Any]:
    return get_chat_status()


@router.post("/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_stream(payload.message, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id, "Cache-Control": "no-cache"},
    )


@router.post("/message")
async def chat_message(payload: ChatRequest) -> dict[str, Any]:
    thread_id = payload.thread_id or str(uuid.uuid4())
    provider = resolve_llm_provider()
    fallback_reason: str | None = None

    if provider != "rules":
        try:
            agent = get_chat_agent()
            config = {"configurable": {"thread_id": thread_id}}
            result = agent.invoke({"messages": [HumanMessage(content=payload.message)]}, config=config)
            final = result["messages"][-1]
            return {
                "thread_id": thread_id,
                "provider": provider,
                "response": final.content,
                "streaming": chat_supports_streaming(),
            }
        except Exception as exc:
            logger.warning("LLM chat failed (%s) — falling back to rules mode: %s", provider, exc)
            fallback_reason = str(exc)

    try:
        response = handle_rules_chat(payload.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "thread_id": thread_id,
        "provider": "rules",
        "response": response,
        "fallback": provider != "rules",
        "fallback_reason": fallback_reason if provider != "rules" else None,
    }
