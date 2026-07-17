"""External evidence sources for Corrective RAG (Tavily, arXiv, Kaggle)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.logging_config import get_logger

logger = get_logger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_TAVILY_API = "https://api.tavily.com/search"


@dataclass
class ExternalDoc:
    source: str
    title: str
    content: str
    score: float
    chunk_type: str  # "web" | "arxiv" | "kaggle"
    tags: list[str]


def _clip(text: str, n: int = 800) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:n]


class TavilySearchClient:
    def __init__(self, api_key: str | None = None, timeout: float = 12.0) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[ExternalDoc]:
        if not self.enabled:
            return []
        try:
            resp = httpx.post(
                _TAVILY_API,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return []

        docs: list[ExternalDoc] = []
        for r in data.get("results", []):
            docs.append(
                ExternalDoc(
                    source=r.get("url", "web"),
                    title=r.get("title", "Untitled"),
                    content=_clip(r.get("content", "")),
                    score=float(r.get("score", 0.5)),
                    chunk_type="web",
                    tags=["web_search"],
                )
            )
        return docs


class ArxivSearchClient:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int = 3) -> list[ExternalDoc]:
        try:
            resp = httpx.get(
                _ARXIV_API,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
            return []
        return self._parse_atom(resp.text)

    @staticmethod
    def _parse_atom(xml_text: str) -> list[ExternalDoc]:
        import xml.etree.ElementTree as ET

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        docs: list[ExternalDoc] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return docs

        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = entry.findtext("atom:id", default="", namespaces=ns) or "arxiv"
            if not title:
                continue
            docs.append(
                ExternalDoc(
                    source=link,
                    title=title,
                    content=_clip(summary),
                    score=0.55,
                    chunk_type="arxiv",
                    tags=["arxiv", "academic"],
                )
            )
        return docs


class KaggleEvidenceClient:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        self._api = None

    def _client(self):
        if self._api is not None:
            return self._api
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

            api = KaggleApi()
            api.authenticate()
            self._api = api
        except Exception as exc:
            logger.warning("Kaggle API unavailable: %s", exc)
            self._api = None
        return self._api

    def search_competitions(self, domain: str, task_type: str, max_results: int = 3) -> list[ExternalDoc]:
        if not self.enabled:
            return []
        api = self._client()
        if api is None:
            return []
        try:
            search_term = f"{domain} {task_type}".strip() or task_type
            comps = api.competitions_list(search=search_term)[:max_results]
        except Exception as exc:
            logger.warning("Kaggle competition search failed: %s", exc)
            return []

        docs: list[ExternalDoc] = []
        for c in comps:
            content = (
                f"Kaggle competition '{c.title}' — category={getattr(c, 'category', '')}, "
                f"reward={getattr(c, 'reward', '')}, evaluation metric context relevant to "
                f"{task_type} on {domain or 'general'} data."
            )
            docs.append(
                ExternalDoc(
                    source=f"https://www.kaggle.com/c/{c.ref}" if hasattr(c, "ref") else "kaggle",
                    title=c.title,
                    content=_clip(content),
                    score=0.5,
                    chunk_type="kaggle",
                    tags=["kaggle", "leaderboard"],
                )
            )
        return docs


class EvidenceRouter:
    def __init__(self) -> None:
        self.tavily = TavilySearchClient()
        self.arxiv = ArxivSearchClient()
        self.kaggle = KaggleEvidenceClient()

    @staticmethod
    def internal_retrieval_is_weak(internal_docs: list[dict[str, Any]], threshold: float = 0.3) -> bool:
        if not internal_docs:
            return True
        avg_score = sum(d.get("score", 0.0) for d in internal_docs) / len(internal_docs)
        return avg_score < threshold

    def route(
        self,
        query_intent: str,
        query_text: str,
        domain: str = "general",
        task_type: str = "classification",
    ) -> list[ExternalDoc]:
        results: list[ExternalDoc] = []
        if query_intent == "benchmark":
            results.extend(self.kaggle.search_competitions(domain, task_type))
            results.extend(self.arxiv.search(query_text, max_results=2))
        elif query_intent == "model_choice":
            results.extend(self.arxiv.search(query_text, max_results=2))
            results.extend(self.tavily.search(query_text, max_results=3))
        else:
            results.extend(self.tavily.search(query_text, max_results=4))
            if not results:
                results.extend(self.arxiv.search(query_text, max_results=2))
        return results


def to_retrieved_doc_dicts(docs: list[ExternalDoc], category: str = "external_evidence") -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "source": d.source,
            "title": d.title,
            "content": d.content,
            "score": d.score,
            "chunk_type": d.chunk_type,
            "tags": d.tags,
        }
        for d in docs
    ]
