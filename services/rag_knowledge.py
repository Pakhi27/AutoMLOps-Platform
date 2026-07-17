"""RAG knowledge retrieval: hybrid TF-IDF + keyword/metadata boosting + MMR diversity."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import BASE_DIR, get_settings
from app.services.advisor_relevance import score_run_memory_relevance
from app.services.dataset_fingerprint import build_dataset_signature, row_size_bucket
from app.utils.io_utils import load_json


@dataclass
class RetrievedChunk:
    source: str
    title: str
    content: str
    score: float
    chunk_type: str = "playbook"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    task_type: str = ""
    target_column: str = ""
    feature_columns: list[str] = field(default_factory=list)
    top_features: list[str] = field(default_factory=list)
    dataset_signature: str = ""
    dataset_id: str = ""
    n_rows: int = 0
    row_bucket: str = ""


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2}


def _chunk_markdown(text: str, source: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = source.replace(".md", "").replace("_", " ").title()
    current_lines: list[str] = []
    tags: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append(_section(source, current_title, current_lines, tags))
            current_title = line[3:].strip()
            current_lines = []
            tags = _infer_tags(current_title, source)
        elif line.startswith("### "):
            if current_lines:
                sections.append(_section(source, current_title, current_lines, tags))
            current_title = f"{current_title} — {line[4:].strip()}"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(_section(source, current_title, current_lines, tags))
    return [s for s in sections if s["content"]]


def _section(source: str, title: str, lines: list[str], tags: list[str]) -> dict[str, Any]:
    content = "\n".join(lines).strip()
    return {
        "source": source,
        "title": title,
        "content": content,
        "chunk_type": "playbook",
        "tags": list(set(tags + _infer_tags(title, source) + _infer_tags(content, source))),
    }


def _infer_tags(text: str, source: str = "") -> list[str]:
    """Technique tags inferred from playbook text — not business-domain labels."""
    t = text.lower()
    tags: list[str] = []
    mapping = {
        "classification": ("classif", "binary", "multiclass", "softmax", "logistic"),
        "regression": ("regress", "rmse", "mae", "r2", "mse"),
        "imbalance": ("imbalance", "imbalanced", "minority", "class weight", "stratified"),
        "drift": ("drift", "monitor", "production", "reference", "evidently"),
        "tuning": ("optuna", "hyperparameter", "trial", "grid search"),
        "features": ("feature", "encoding", "categorical", "datetime", "interaction", "engineering"),
        "missing": ("missing", "impute", "imputation", "null"),
    }
    for tag, keywords in mapping.items():
        if any(k in t for k in keywords):
            tags.append(tag)
    return tags


def _run_to_chunk(job_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics", {})
    metric_str = ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )
    leaderboard = entry.get("baseline_scores") or {}
    lb_str = ""
    if isinstance(leaderboard, dict):
        lb_str = ", ".join(
            f"{m}={s:.4f}" if isinstance(s, float) else f"{m}={s}" for m, s in leaderboard.items()
        )

    fi = entry.get("top_features") or []
    fi_names = [f["feature"] for f in fi[:8] if isinstance(f, dict) and f.get("feature")]
    fi_str = ", ".join(f"{f['feature']}={f.get('importance', 0):.3f}" for f in fi[:5]) if fi else ""

    task = entry.get("task_type", "")
    target = entry.get("target_column", "")
    feature_columns = entry.get("feature_columns") or []
    n_rows = int(entry.get("n_rows") or 0)
    dataset_id = entry.get("dataset_id") or ""
    signature = entry.get("dataset_signature") or build_dataset_signature(
        list(feature_columns) + ([target] if target else []), target
    )

    content = (
        f"Historical run {job_id}: task={task}, target={target}, "
        f"model={entry.get('model_name')}, rows={n_rows}, dataset={dataset_id}. "
        f"Test metrics: {metric_str}. CV leaderboard: {lb_str}. "
        f"Features: {', '.join(feature_columns[:12])}. Top drivers: {fi_str}."
    )
    tags = [t for t in (task, entry.get("row_bucket")) if t]
    if entry.get("is_imbalanced"):
        tags.append("imbalance")
    return {
        "source": f"run:{job_id}",
        "title": f"Past run — {entry.get('model_name')} on {target}",
        "content": content,
        "chunk_type": "run_memory",
        "task_type": task,
        "target_column": target,
        "feature_columns": list(feature_columns),
        "top_features": fi_names,
        "dataset_signature": signature,
        "dataset_id": dataset_id,
        "n_rows": n_rows,
        "row_bucket": entry.get("row_bucket") or row_size_bucket(n_rows),
        "tags": tags,
    }


class RAGKnowledgeBase:
    """Hybrid retrieval over playbooks + model registry run memory."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        settings = get_settings()
        self.knowledge_dir = knowledge_dir or (BASE_DIR / settings.agent.get("knowledge_dir", "data/knowledge"))
        self.registry_file = settings.registry_file
        self._chunks: list[dict[str, Any]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        self._chunks = []
        if self.knowledge_dir.exists():
            for path in sorted(self.knowledge_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                self._chunks.extend(_chunk_markdown(text, path.name))

        registry = load_json(self.registry_file, default={})
        for job_id, entry in registry.items():
            if isinstance(entry, dict) and entry.get("model_name"):
                self._chunks.append(_run_to_chunk(job_id, entry))

        if not self._chunks:
            return

        corpus = [self._corpus_text(c) for c in self._chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        self._matrix = self._vectorizer.fit_transform(corpus)

    @staticmethod
    def _corpus_text(chunk: dict[str, Any]) -> str:
        tags = " ".join(chunk.get("tags", []))
        return f"{chunk['title']} {tags} {chunk['content']}"

    def reload(self) -> None:
        get_knowledge_base.cache_clear()
        self._load()

    def list_topics(self) -> list[dict[str, str]]:
        return [{"source": c["source"], "title": c["title"]} for c in self._chunks]

    def _hybrid_score(
        self,
        idx: int,
        tfidf_score: float,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> float:
        chunk = self._chunks[idx]
        text = self._corpus_text(chunk).lower()
        query_tokens = _tokenize(query)
        overlap = sum(0.06 for t in query_tokens if t in text)
        score = float(tfidf_score) + overlap

        ctx = context or {}
        task = ctx.get("task_type", "")
        ctx = context or {}
        task = ctx.get("task_type", "")
        if task and chunk.get("task_type") == task:
            score += 0.15
        if ctx.get("is_imbalanced") and "imbalance" in chunk.get("tags", []):
            score += 0.12
        if chunk.get("chunk_type") == "run_memory":
            rel = score_run_memory_relevance(chunk, ctx)
            score += rel * 0.4
            if rel < float((get_settings().agent.get("relevance") or {}).get("run_retrieval_min", 0.32)):
                score -= 0.6
        return score

    def _mmr_select(
        self,
        candidates: list[tuple[int, float]],
        top_k: int,
        lambda_div: float = 0.7,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        selected: list[tuple[int, float]] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda x: x[1])
                selected.append(best)
                remaining.remove(best)
                continue
            best_item = None
            best_mmr = -1.0
            for idx, rel_score in remaining:
                chunk_a = set(_tokenize(self._chunks[idx]["content"]))
                max_sim = 0.0
                for sel_idx, _ in selected:
                    chunk_b = set(_tokenize(self._chunks[sel_idx]["content"]))
                    if chunk_a and chunk_b:
                        inter = len(chunk_a & chunk_b)
                        union = len(chunk_a | chunk_b)
                        max_sim = max(max_sim, inter / union if union else 0)
                mmr = lambda_div * rel_score - (1 - lambda_div) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_item = (idx, rel_score)
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                break
        return selected

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        chunk_type: str | None = None,
        context: dict[str, Any] | None = None,
        use_mmr: bool = True,
    ) -> list[RetrievedChunk]:
        if not self._chunks or self._vectorizer is None or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        tfidf_scores = (self._matrix * query_vec.T).toarray().ravel()
        candidates: list[tuple[int, float]] = []
        for idx, tfidf in enumerate(tfidf_scores):
            if tfidf <= 0:
                continue
            chunk = self._chunks[idx]
            if chunk_type and chunk.get("chunk_type") != chunk_type:
                continue
            hybrid = self._hybrid_score(idx, float(tfidf), query, context)
            candidates.append((idx, hybrid))

        candidates.sort(key=lambda x: x[1], reverse=True)
        pool = candidates[: max(top_k * 4, 12)]
        picked = self._mmr_select(pool, top_k) if use_mmr else pool[:top_k]

        results: list[RetrievedChunk] = []
        for idx, score in picked:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    source=chunk["source"],
                    title=chunk["title"],
                    content=chunk["content"],
                    score=round(score, 4),
                    chunk_type=chunk.get("chunk_type", "playbook"),
                    tags=list(chunk.get("tags", [])),
                    task_type=chunk.get("task_type") or "",
                    target_column=chunk.get("target_column") or "",
                    feature_columns=list(chunk.get("feature_columns") or []),
                    top_features=list(chunk.get("top_features") or []),
                    dataset_signature=chunk.get("dataset_signature") or "",
                    dataset_id=chunk.get("dataset_id") or "",
                    n_rows=int(chunk.get("n_rows") or 0),
                    row_bucket=chunk.get("row_bucket") or "",
                )
            )
        return results

    def multi_query_retrieve(
        self,
        queries: dict[str, str],
        top_k: int = 3,
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[RetrievedChunk]]:
        return {name: self.retrieve(q, top_k=top_k, context=context) for name, q in queries.items()}

    def retrieve_merged(
        self,
        queries: dict[str, str],
        top_k_per_query: int = 3,
        max_total: int = 12,
        context: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve across multiple queries, dedupe by source+title, rank by score."""
        seen: set[str] = set()
        merged: list[RetrievedChunk] = []
        for category, query in queries.items():
            for chunk in self.retrieve(query, top_k=top_k_per_query, context=context):
                key = f"{chunk.source}|{chunk.title}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    RetrievedChunk(
                        source=chunk.source,
                        title=chunk.title,
                        content=chunk.content,
                        score=chunk.score,
                        chunk_type=chunk.chunk_type,
                        category=category,
                        tags=chunk.tags,
                        task_type=chunk.task_type,
                        target_column=chunk.target_column,
                        feature_columns=chunk.feature_columns,
                        top_features=chunk.top_features,
                        dataset_signature=chunk.dataset_signature,
                        dataset_id=chunk.dataset_id,
                        n_rows=chunk.n_rows,
                        row_bucket=chunk.row_bucket,
                    )
                )
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[:max_total]

    def build_query_from_profile(self, profile: dict[str, Any], target_column: str | None = None) -> str:
        from app.services.dataset_fingerprint import build_dataset_context

        ctx = build_dataset_context(profile, target_column)
        parts = [
            f"rows {ctx.n_rows}",
            f"columns {ctx.n_columns}",
            f"task {ctx.task_type}",
            f"target {ctx.target_column}",
            f"features {' '.join(ctx.feature_columns[:8])}",
        ]
        if ctx.is_imbalanced:
            parts.append("class imbalance f1 roc-auc stratified")
        if ctx.has_missing:
            parts.append("missing values imputation")
        if ctx.has_datetime:
            parts.append("datetime feature engineering")
        if ctx.task_type == "regression":
            parts.append("regression mae rmse feature engineering")
        return " ".join(parts)


@lru_cache
def get_knowledge_base() -> RAGKnowledgeBase:
    return RAGKnowledgeBase()
