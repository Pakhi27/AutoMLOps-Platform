"""Tests for rule-based chat and LLM provider resolution."""
import os

from app.services.llm_provider import resolve_llm_provider
from app.services.rules_chatbot import handle_rules_chat


def test_rules_chat_help():
    out = handle_rules_chat("help")
    assert "list jobs" in out.lower()


def test_rules_chat_list_jobs(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    out = handle_rules_chat("list jobs")
    assert "No jobs found" in out or "job_" in out


def test_resolve_provider_rules_when_no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    assert resolve_llm_provider() == "rules"
