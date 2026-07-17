"""Config-driven relevance scoring — compares runs and tips using dataset structure only."""
from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.services.dataset_fingerprint import DatasetContext, _split_column_tokens, _tokens, row_size_bucket

# Technique vocabulary shared across all tabular ML problems (not business-domain specific)
_ML_TECHNIQUE_TOKENS = frozenset(
    {
        "accuracy", "auc", "baseline", "binary", "boosting", "categorical", "class",
        "classification", "cross", "dataset", "encoding", "ensemble", "feature", "features",
        "fold", "f1", "gradient", "holdout", "hyperparameter", "impute", "imputation",
        "leaderboard", "log", "metric", "metrics", "missing", "model", "models", "numeric",
        "optuna", "pipeline", "precision", "predict", "recall", "regression", "regularization",
        "roc", "scale", "score", "scores", "shap", "sklearn", "split", "stratified", "target",
        "test", "train", "training", "trial", "trials", "tune", "tuning", "validation", "value",
        "values", "weight", "weighted", "mlflow", "overfitting", "underfitting", "cv",
    }
)


def _relevance_config() -> dict[str, Any]:
    return get_settings().agent.get("relevance") or {}


def _norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def _target_similarity(a: str | None, b: str | None) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.75
    ta, tb = _tokens(na), _tokens(nb)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _chunk_job_id(chunk: dict[str, Any]) -> str | None:
    source = chunk.get("source", "")
    if source.startswith("run:"):
        return source.split(":", 1)[1]
    return None


def _chunk_feature_tokens(chunk: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for col in chunk.get("feature_columns") or []:
        tokens |= _split_column_tokens([col])
    for col in chunk.get("top_features") or []:
        tokens |= _split_column_tokens([col])
    sig = chunk.get("dataset_signature") or ""
    if sig:
        tokens |= _split_column_tokens(sig.split("|"))
    return tokens


def score_run_memory_relevance(
    chunk: dict[str, Any],
    context: DatasetContext | dict[str, Any],
) -> float:
    """Score 0–1 how comparable a historical run is — structure only, no domain labels."""
    if chunk.get("chunk_type") != "run_memory":
        return 1.0

    if isinstance(context, dict):
        current_job_id = context.get("current_job_id")
        ctx = DatasetContext(
            task_type=context.get("task_type") or "classification",
            target_column=context.get("target_column") or "",
            feature_columns=list(context.get("feature_columns") or []),
            column_tokens=set(context.get("column_tokens") or []),
            dataset_signature=context.get("dataset_signature") or "",
            n_rows=int(context.get("n_rows") or 0),
            row_bucket=context.get("row_bucket") or row_size_bucket(int(context.get("n_rows") or 0)),
            is_imbalanced=bool(context.get("is_imbalanced")),
            dataset_id=context.get("dataset_id") or "",
        )
    else:
        ctx = context
        current_job_id = None

    job_id = _chunk_job_id(chunk)
    if current_job_id and job_id == current_job_id:
        return 0.0

    weights = _relevance_config().get("weights") or {}
    w_task = float(weights.get("task_type_match", 0.25))
    w_target = float(weights.get("target_match", 0.30))
    w_features = float(weights.get("feature_overlap", 0.30))
    w_rows = float(weights.get("row_bucket_match", 0.10))
    w_dataset = float(weights.get("same_dataset", 0.15))

    score = 0.05

    if ctx.task_type and chunk.get("task_type") == ctx.task_type:
        score += w_task

    target_sim = _target_similarity(ctx.target_column, chunk.get("target_column"))
    score += w_target * target_sim

    run_feats = _chunk_feature_tokens(chunk)
    feat_sim = _jaccard(ctx.column_tokens, run_feats)
    score += w_features * feat_sim

    chunk_bucket = chunk.get("row_bucket") or row_size_bucket(int(chunk.get("n_rows") or 0))
    if chunk_bucket == ctx.row_bucket:
        score += w_rows

    if ctx.dataset_id and chunk.get("dataset_id") == ctx.dataset_id:
        score += w_dataset

    sig_a = ctx.dataset_signature
    sig_b = chunk.get("dataset_signature") or ""
    if sig_a and sig_b and sig_a == sig_b:
        score = min(1.0, score + 0.2)

    comparable = (
        target_sim >= 0.5
        or feat_sim >= 0.12
        or (sig_a and sig_b and sig_a == sig_b)
        or (ctx.dataset_id and chunk.get("dataset_id") == ctx.dataset_id)
    )
    if not comparable:
        score = min(score, 0.22)

    return max(0.0, min(1.0, score))


def filter_run_memory_docs(
    docs: list[dict[str, Any]],
    context: DatasetContext | dict[str, Any],
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    cfg = _relevance_config()
    threshold = min_score if min_score is not None else float(cfg.get("run_retrieval_min", 0.32))
    kept: list[dict[str, Any]] = []
    for doc in docs:
        if doc.get("chunk_type") != "run_memory":
            kept.append(doc)
            continue
        rel = score_run_memory_relevance(doc, context)
        if rel < threshold:
            continue
        enriched = dict(doc)
        enriched["run_relevance"] = round(rel, 4)
        kept.append(enriched)
    return kept


def comparable_run_actions(
    docs: list[dict[str, Any]],
    max_actions: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    cfg = _relevance_config()
    limit = max_actions if max_actions is not None else int(cfg.get("max_compare_actions", 2))
    threshold = min_score if min_score is not None else float(cfg.get("run_compare_min", 0.48))

    candidates: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for doc in docs:
        if doc.get("chunk_type") != "run_memory":
            continue
        rel = float(doc.get("run_relevance") or 0)
        if rel < threshold:
            continue
        key = doc.get("source") or doc.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        candidates.append((rel, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "action": f"Compare with comparable past run: {doc['title']} (similarity {rel:.0%})",
            "source": doc["source"],
        }
        for rel, doc in candidates[:limit]
    ]


def score_tip_relevance(tip: str, column_tokens: set[str]) -> float:
    """How applicable a playbook tip is to this dataset's vocabulary."""
    tip_tokens = _tokens(tip)
    if not tip_tokens:
        return 0.0

    applicable = 0
    for token in tip_tokens:
        if token in _ML_TECHNIQUE_TOKENS:
            applicable += 1
            continue
        if any(token in col or col in token for col in column_tokens):
            applicable += 1

    return applicable / len(tip_tokens)


def filter_tips_for_dataset(
    tips: list[str],
    column_tokens: set[str],
    min_relevance: float | None = None,
) -> list[str]:
    cfg = _relevance_config()
    threshold = min_relevance if min_relevance is not None else float(cfg.get("tip_min_relevance", 0.15))

    filtered: list[str] = []
    seen: set[str] = set()
    scored = [(score_tip_relevance(t, column_tokens), t) for t in tips]
    scored.sort(key=lambda x: x[0], reverse=True)

    for rel, tip in scored:
        if rel < threshold:
            continue
        if tip in seen:
            continue
        seen.add(tip)
        filtered.append(tip)
    return filtered


def dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for action in actions:
        key = action.get("action", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out
