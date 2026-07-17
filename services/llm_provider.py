"""LLM provider selection for chatbot and optional advisor synthesis."""
from __future__ import annotations

import os
from typing import Any

import httpx

from app.core.logging_config import get_logger

logger = get_logger(__name__)

PROVIDERS = ("openai", "gemini", "groq", "ollama", "rules")


def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _has_gemini() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def _ollama_reachable() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def resolve_llm_provider() -> str:
    """Pick chat/advisor LLM backend from LLM_PROVIDER or env auto-detect."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in PROVIDERS:
        if explicit == "openai" and not _has_openai():
            logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY missing — falling back.")
        elif explicit == "gemini" and not _has_gemini():
            logger.warning("LLM_PROVIDER=gemini but GOOGLE_API_KEY missing — falling back.")
        elif explicit == "groq" and not _has_groq():
            logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY missing — falling back.")
        elif explicit == "ollama" and not _ollama_reachable():
            logger.warning("LLM_PROVIDER=ollama but Ollama not reachable — falling back.")
        else:
            return explicit

    if _has_openai():
        return "openai"
    if _has_gemini():
        return "gemini"
    if _has_groq():
        return "groq"
    if _ollama_reachable():
        return "ollama"
    return "rules"


def chat_available() -> bool:
    """Chat is always available — rules mode works without any API key."""
    return True


def chat_supports_streaming() -> bool:
    return resolve_llm_provider() != "rules"


def get_provider_model_name() -> str:
    provider = resolve_llm_provider()
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2")
    return "keyword-router"


def get_chat_model_name() -> str:
    """Model for ReAct tool-calling chat — Groq needs a tool-capable model."""
    provider = resolve_llm_provider()
    if provider == "openai":
        return os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_CHAT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "groq":
        # llama-3.1-8b-instant often fails Groq tool_use — use 70b for chat tools
        return os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    if provider == "ollama":
        return os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2")
    return "keyword-router"


def get_advisor_model_name() -> str:
    """Faster/cheaper model for advisor narrative synthesis."""
    provider = resolve_llm_provider()
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2")
    return "keyword-router"


def get_chat_status() -> dict[str, Any]:
    provider = resolve_llm_provider()
    ready = provider == "rules"
    error = ""
    if provider != "rules":
        try:
            build_chat_llm()
            ready = True
        except Exception as exc:
            ready = False
            error = str(exc)
    hints = {
        "openai": "Set OPENAI_API_KEY",
        "gemini": "Set GOOGLE_API_KEY (https://aistudio.google.com/apikey)",
        "groq": "Set GROQ_API_KEY (https://console.groq.com)",
        "ollama": "Install Ollama and run: ollama pull llama3.2",
        "rules": "No LLM key needed — try: list jobs | status job_xxx | features job_xxx",
    }
    return {
        "available": True,
        "ready": ready,
        "provider": provider,
        "model": get_chat_model_name() if provider != "rules" else get_provider_model_name(),
        "advisor_model": get_advisor_model_name() if provider != "rules" else None,
        "streaming": chat_supports_streaming() and ready,
        "hint": error or hints.get(provider, ""),
    }


def build_chat_llm():
    """Return a LangChain chat model for ReAct agent, or None for rules mode."""
    provider = resolve_llm_provider()
    if provider == "rules":
        return None

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=get_chat_model_name(),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            streaming=True,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-google-genai for Gemini support.") from exc
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=get_chat_model_name(), google_api_key=key, temperature=0.1)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError("Install langchain-groq for Groq support.") from exc
        return ChatGroq(model=get_chat_model_name(), api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError as exc:
                raise RuntimeError("Install langchain-ollama for Ollama support.") from exc
        return ChatOllama(
            model=get_chat_model_name(),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )

    raise RuntimeError(f"Unknown LLM provider: {provider}")


def build_advisor_llm(api_key: str | None = None):
    """Optional LLM for advisor narrative (uses same provider resolution)."""
    provider = resolve_llm_provider()
    if provider == "rules":
        return None
    if provider == "openai" and (api_key or _has_openai()):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=get_advisor_model_name(),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.15,
        )
    if provider == "gemini" and _has_gemini():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return None
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=get_advisor_model_name(), google_api_key=key, temperature=0.15)
    if provider == "groq" and _has_groq():
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            return None
        return ChatGroq(model=get_advisor_model_name(), api_key=os.getenv("GROQ_API_KEY"), temperature=0.15)
    if provider == "ollama" and _ollama_reachable():
        try:
            return build_chat_llm()
        except Exception:
            return None
    try:
        return build_chat_llm()
    except Exception:
        return None
