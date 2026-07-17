"""Evidence-grounded LangGraph ML advisor with hybrid RAG, job artifacts, and critic retry."""
from __future__ import annotations

import os
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.core.logging_config import get_logger
from app.core.config import get_settings
from app.services.advisor_relevance import (
    comparable_run_actions,
    dedupe_actions,
    filter_run_memory_docs,
    filter_tips_for_dataset,
)
from app.services.dataset_fingerprint import DatasetContext, build_dataset_context, row_size_bucket
from app.services.model_selector import ModelSelector
from app.services.rag_knowledge import get_knowledge_base
from app.services.web_retrieval_tools import EvidenceRouter, to_retrieved_doc_dicts

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    mode: Literal["pre_train", "post_train"]
    profile: dict[str, Any]
    fingerprint: dict[str, Any]
    dataset_context: dict[str, Any]
    target_column: str | None
    task_type: str | None
    current_job_id: str | None
    job_result: dict[str, Any] | None
    job_evidence: dict[str, Any]
    rag_queries: dict[str, str]
    retrieved_docs: list[dict[str, Any]]
    playbook_excerpts: list[str]
    data_insights: list[str]
    model_recommendations: list[dict[str, Any]]
    preprocessing_tips: list[str]
    top_actions: list[dict[str, Any]]
    risks: list[str]
    confidence: float
    critic_passed: bool
    critic_retried: bool
    web_evidence_used: bool
    narrative_report: str
    llm_used: bool
    llm_provider: str | None


def _rag_context(state: AgentState) -> dict[str, Any]:
    ctx = state.get("dataset_context")
    if ctx:
        out = dict(ctx)
        out["current_job_id"] = state.get("current_job_id")
        out["mode"] = state.get("mode")
        return out
    profile = state.get("profile", {})
    ds = build_dataset_context(profile, state.get("target_column"))
    return ds.to_rag_context(current_job_id=state.get("current_job_id"), mode=state.get("mode") or "pre_train")


def _apply_run_memory_filter(state: AgentState) -> None:
    state["retrieved_docs"] = filter_run_memory_docs(
        state.get("retrieved_docs", []),
        _rag_context(state),
    )


def build_fingerprint(profile: dict[str, Any], target_column: str | None, job_result: dict | None) -> dict[str, Any]:
    ctx = build_dataset_context(profile, target_column)
    fp: dict[str, Any] = {
        "n_rows": ctx.n_rows,
        "n_columns": ctx.n_columns,
        "n_numeric": len(profile.get("numeric_columns", [])),
        "n_categorical": len(profile.get("categorical_columns", [])),
        "n_datetime": len(profile.get("datetime_columns", [])),
        "task_type": ctx.task_type,
        "target_column": ctx.target_column,
        "row_bucket": ctx.row_bucket,
        "feature_columns": ctx.feature_columns,
        "dataset_signature": ctx.dataset_signature,
        "has_missing": ctx.has_missing,
        "has_datetime": ctx.has_datetime,
    }
    eda_target = profile.get("target_analysis") or {}
    if eda_target.get("is_imbalanced"):
        fp["is_imbalanced"] = True
        fp["imbalance_ratio"] = eda_target.get("imbalance_ratio", 0)
    if eda_target.get("type"):
        fp["task_type"] = eda_target["type"]
    if job_result:
        fp["winner_model"] = job_result.get("model_name")
        fp["metrics"] = job_result.get("metrics", {})
        fp["leaderboard"] = dict(job_result.get("baseline_scores") or {})
        fp["best_params"] = job_result.get("best_params", {})
        if job_result.get("feature_importance"):
            fp["feature_importance"] = job_result["feature_importance"][:5]
    return fp


def _dedupe_retrieved_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        key = f"{doc.get('source', '')}|{doc.get('title', '')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _plan_rag_queries(state: AgentState) -> dict[str, str]:
    fp = state["fingerprint"]
    ctx = state.get("dataset_context") or {}
    target = state.get("target_column") or fp.get("target_column") or "target"
    task = state.get("task_type") or fp.get("task_type") or "classification"
    features = " ".join((ctx.get("feature_columns") or fp.get("feature_columns") or [])[:6])
    n_rows = fp.get("n_rows", 0)

    queries = {
        "models": f"{task} model selection tabular {n_rows} rows {features}",
        "preprocessing": f"feature engineering missing values categorical numeric {features}",
        "tuning": f"optuna hyperparameter tuning {task} cross validation",
        "drift": f"data drift monitoring production deployment features {features}",
        "similar_runs": f"historical run {task} target {target} features {features}",
    }
    if fp.get("is_imbalanced"):
        queries["imbalance"] = "class imbalance stratified f1 roc-auc precision recall"
    if task == "regression":
        queries["regression"] = "regression mae rmse r2 target transform skew"
    if fp.get("has_missing"):
        queries["missing"] = "missing values imputation numeric categorical"
    if fp.get("has_datetime") or fp.get("n_datetime", 0) > 0:
        queries["datetime"] = "datetime feature engineering temporal columns"
    if state.get("mode") == "post_train" and fp.get("winner_model"):
        queries["winner"] = f"why {fp['winner_model']} won {task} tuning evidence"
    return queries


def _extract_playbook_tips(docs: list[dict[str, Any]], column_tokens: set[str], max_tips: int = 4) -> list[str]:
    tips: list[str] = []
    for doc in docs:
        if doc.get("chunk_type") != "playbook":
            continue
        for line in doc.get("content", "").splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "•")) or re.match(r"^\d+\.", line):
                tip = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
                if len(tip) > 20 and tip not in tips:
                    tips.append(tip)
                if len(tips) >= max_tips * 2:
                    break
    return filter_tips_for_dataset(tips, column_tokens)[:max_tips]


def _node_fingerprint(state: AgentState) -> AgentState:
    profile = state["profile"]
    target = state.get("target_column")
    job_result = state.get("job_result")
    ctx = build_dataset_context(profile, target)
    state["dataset_context"] = ctx.to_rag_context(
        current_job_id=state.get("current_job_id"),
        mode="post_train" if job_result else "pre_train",
    )
    state["fingerprint"] = build_fingerprint(profile, target, job_result)
    state["mode"] = "post_train" if job_result else "pre_train"

    if not state.get("task_type"):
        if job_result and job_result.get("task_type"):
            state["task_type"] = job_result["task_type"]
        elif target:
            eda = profile.get("target_analysis", {})
            state["task_type"] = eda.get("type", "classification")
        else:
            state["task_type"] = "classification"
    return state


def _node_multi_rag(state: AgentState) -> AgentState:
    kb = get_knowledge_base()
    fp = state["fingerprint"]
    ctx = state.get("dataset_context") or {}
    queries = _plan_rag_queries(state)
    state["rag_queries"] = queries

    rag_cfg = get_settings().agent.get("rag") or {}
    merged = kb.retrieve_merged(
        queries,
        top_k_per_query=int(rag_cfg.get("top_k_per_query", 3)),
        max_total=int(rag_cfg.get("max_total", 14)),
        context=_rag_context(state),
    )

    state["retrieved_docs"] = [
        {
            "category": c.category,
            "source": c.source,
            "title": c.title,
            "content": c.content,
            "score": c.score,
            "chunk_type": c.chunk_type,
            "tags": c.tags,
            "task_type": c.task_type,
            "target_column": c.target_column,
            "feature_columns": c.feature_columns,
            "top_features": c.top_features,
            "dataset_signature": c.dataset_signature,
            "dataset_id": c.dataset_id,
            "n_rows": c.n_rows,
            "row_bucket": c.row_bucket,
        }
        for c in merged
    ]
    _apply_run_memory_filter(state)
    tokens = set(ctx.get("column_tokens") or [])
    state["playbook_excerpts"] = _extract_playbook_tips(state["retrieved_docs"], tokens)
    return state


def _node_expand_rag(state: AgentState) -> AgentState:
    """Second-pass retrieval when critic flags issues — broader search."""
    kb = get_knowledge_base()
    fp = state["fingerprint"]
    task = state.get("task_type", "classification")
    extra = {
        "risk_mitigation": f"{task} overfitting small data regularization alternatives",
        "imbalance_fix": "imbalanced classification precision recall threshold tuning",
        "ensemble": "ensemble stacking top models close leaderboard gap",
    }
    if fp.get("n_rows", 0) < 1000:
        extra["small_data"] = "small dataset simple models regularization catboost random forest"

    rag_context = _rag_context(state)
    existing_keys = {f"{d['source']}|{d['title']}" for d in state.get("retrieved_docs", [])}
    for chunk in kb.retrieve_merged(extra, top_k_per_query=2, max_total=6, context=rag_context):
        key = f"{chunk.source}|{chunk.title}"
        if key not in existing_keys:
            state["retrieved_docs"].append(
                {
                    "category": "critic_retry",
                    "source": chunk.source,
                    "title": chunk.title,
                    "content": chunk.content,
                    "score": chunk.score,
                    "chunk_type": chunk.chunk_type,
                    "tags": chunk.tags,
                    "task_type": chunk.task_type,
                    "target_column": chunk.target_column,
                    "feature_columns": chunk.feature_columns,
                    "top_features": chunk.top_features,
                    "dataset_signature": chunk.dataset_signature,
                    "dataset_id": chunk.dataset_id,
                    "n_rows": chunk.n_rows,
                    "row_bucket": chunk.row_bucket,
                }
            )
            existing_keys.add(key)

    _apply_run_memory_filter(state)
    ctx = state.get("dataset_context") or {}
    state["playbook_excerpts"] = _extract_playbook_tips(
        state.get("retrieved_docs", []),
        set(ctx.get("column_tokens") or []),
        max_tips=6,
    )
    state["critic_retried"] = True
    return state


_evidence_router = EvidenceRouter()


def _classify_query_intent(state: AgentState) -> str:
    fp = state["fingerprint"]
    if state.get("mode") == "post_train" and fp.get("winner_model"):
        return "benchmark"
    if fp.get("n_rows", 0) < 1000 or fp.get("is_imbalanced"):
        return "model_choice"
    return "general"


def _node_web_search(state: AgentState) -> AgentState:
    fp = state["fingerprint"]
    task = state.get("task_type", "classification")
    features = " ".join(fp.get("feature_columns") or [])[:80]
    intent = _classify_query_intent(state)
    query_text = (
        f"best {task} model tabular data {fp.get('n_rows', 0)} rows target {fp.get('target_column')} "
        f"features {features} {'imbalanced' if fp.get('is_imbalanced') else ''}"
    ).strip()

    logger.info("Internal RAG weak — fetching external evidence (intent=%s)", intent)
    external_docs = _evidence_router.route(intent, query_text, domain="", task_type=task)
    new_docs = to_retrieved_doc_dicts(external_docs)

    existing_keys = {f"{d['source']}|{d['title']}" for d in state.get("retrieved_docs", [])}
    for doc in new_docs:
        key = f"{doc['source']}|{doc['title']}"
        if key not in existing_keys:
            state["retrieved_docs"].append(doc)
            existing_keys.add(key)

    state["web_evidence_used"] = bool(new_docs)
    _apply_run_memory_filter(state)
    ctx = state.get("dataset_context") or {}
    state["playbook_excerpts"] = _extract_playbook_tips(
        state.get("retrieved_docs", []),
        set(ctx.get("column_tokens") or []),
        max_tips=6,
    )
    state["critic_retried"] = True
    return state


def _node_evidence_merge(state: AgentState) -> AgentState:
    fp = state["fingerprint"]
    profile = state["profile"]
    job_result = state.get("job_result") or {}
    job_evidence = state.get("job_evidence") or {}
    task_type = state.get("task_type", "classification")
    insights: list[str] = []
    recommendations: list[dict[str, Any]] = []
    tips: list[str] = list(state.get("playbook_excerpts", []))
    risks: list[str] = []
    actions: list[dict[str, Any]] = []

    insights.append(
        f"Dataset: {fp['n_rows']:,} rows × {fp['n_columns']} columns "
        f"({fp['n_numeric']} numeric, {fp['n_categorical']} categorical, target={fp.get('target_column', '—')})."
    )

    if fp.get("is_imbalanced"):
        ratio = fp.get("imbalance_ratio", 0)
        insights.append(f"Class imbalance — majority class is {ratio * 100:.0f}% of data.")
        risks.append("Accuracy can be misleading — prioritize F1-weighted and ROC-AUC.")
        actions.append({"action": "Use class-weighted boosting or tune decision threshold", "source": "playbook:imbalance"})

    missing_cols = [c for c, v in profile.get("missing_values", {}).items() if v.get("n_missing", 0) > 0]
    if missing_cols:
        insights.append(f"{len(missing_cols)} columns have missing values.")
        if not any("imput" in t.lower() for t in tips):
            tips.append("Median imputation for numeric, mode for categorical columns.")

    if profile.get("datetime_columns"):
        cols = ", ".join(profile["datetime_columns"])
        tips.append(f"Expand datetime columns ({cols}) into year, month, day-of-week, tenure features.")

    corr_target = profile.get("correlation_with_target") or []
    if corr_target:
        top = corr_target[0]
        insights.append(f"Strongest numeric predictor: {top['feature']} (r={top['correlation']}).")

    top_features = job_evidence.get("feature_importance") or job_result.get("feature_importance") or []
    if top_features:
        names = [f["feature"] for f in top_features[:3]]
        insights.append(f"Top model drivers (feature importance): {', '.join(names)}.")

    if job_result:
        winner = job_result.get("model_name", "unknown")
        metrics = job_result.get("metrics", {})
        leaderboard = job_result.get("baseline_scores", {})
        sorted_lb = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)

        insights.append(f"Training complete — winner: **{winner}** after baseline CV + Optuna tuning.")
        if sorted_lb:
            second = sorted_lb[1] if len(sorted_lb) > 1 else None
            gap = sorted_lb[0][1] - second[1] if second else 0
            insights.append(
                f"CV leaderboard (cross-validation): {sorted_lb[0][0]}={sorted_lb[0][1]:.4f}"
                + (f", runner-up {second[0]}={second[1]:.4f} (gap {gap:.4f})" if second else "")
            )
            recommendations.append(
                {
                    "model": winner,
                    "confidence": min(0.95, 0.62 + gap * 2.5),
                    "rationale": f"Won CV with {sorted_lb[0][1]:.4f}; tuned via Optuna on holdout-safe pipeline.",
                    "source": "evidence:leaderboard",
                }
            )
            if second and gap < 0.02:
                risks.append(f"{second[0]} within {gap:.4f} CV of winner — consider ensemble or more trials.")
                actions.append({"action": f"Ensemble {winner} + {second[0]} or run 40+ Optuna trials", "source": "evidence"})

        best_params = job_result.get("best_params") or fp.get("best_params") or {}
        if best_params:
            param_str = ", ".join(f"{k}={v}" for k, v in list(best_params.items())[:4])
            insights.append(f"Best hyperparameters: {param_str}.")

        if metrics.get("roc_auc"):
            insights.append(f"Holdout test ROC-AUC: {metrics['roc_auc']:.4f}")
        if metrics.get("f1_weighted"):
            insights.append(f"Holdout test F1 (weighted): {metrics['f1_weighted']:.4f}")
        elif metrics.get("r2"):
            insights.append(f"Holdout test R²: {metrics['r2']:.4f}")

        actions.extend(comparable_run_actions(state.get("retrieved_docs", []), max_actions=2))
    else:
        available = ModelSelector.available_models(task_type)
        n_rows = fp.get("n_rows", 0)
        if n_rows < 1000:
            preferred = ["catboost", "random_forest", "logistic_regression"] if task_type == "classification" else ["ridge", "elastic_net", "random_forest"]
        elif fp.get("is_imbalanced"):
            preferred = ["catboost", "lightgbm", "xgboost", "hist_gradient_boosting"]
        else:
            preferred = ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"]
        for m in preferred:
            if m in available:
                rationale = f"Strong default for {n_rows:,}-row {task_type} tabular data"
                recommendations.append(
                    {"model": m, "confidence": 0.68, "rationale": rationale, "source": "heuristic+rag"}
                )

    for doc in state.get("retrieved_docs", []):
        if doc.get("category") == "drift":
            actions.append({"action": "Set up weekly drift checks on top 5 features", "source": doc["source"]})
            break

    state["data_insights"] = insights
    state["model_recommendations"] = recommendations[:5]
    ctx = state.get("dataset_context") or {}
    state["preprocessing_tips"] = filter_tips_for_dataset(
        tips, set(ctx.get("column_tokens") or [])
    )[:8]
    state["top_actions"] = dedupe_actions(actions)[:6]
    state["risks"] = risks
    state["confidence"] = recommendations[0]["confidence"] if recommendations else 0.55
    return state


def _node_critic(state: AgentState) -> AgentState:
    fp = state["fingerprint"]
    recs = state.get("model_recommendations", [])
    risks = list(state.get("risks", []))
    passed = True

    for rec in recs:
        model = rec.get("model", "")
        if fp.get("n_rows", 0) < 500 and model in ("svm", "knn"):
            rec["confidence"] = max(0.25, rec.get("confidence", 0.5) - 0.25)
            risks.append(f"Downgraded {model}: only {fp['n_rows']} rows — high overfitting risk.")
            passed = False
        if fp.get("n_categorical", 0) >= 4 and model == "svm":
            rec["confidence"] = max(0.3, rec.get("confidence", 0.5) - 0.15)
            risks.append("SVM struggles with many categoricals — prefer CatBoost/LightGBM.")

    if fp.get("is_imbalanced") and state.get("mode") == "post_train":
        metrics = fp.get("metrics", {})
        if metrics.get("accuracy", 0) > 0.9 and metrics.get("roc_auc", 1) < 0.75:
            risks.append("High accuracy but low AUC — model may be predicting majority class only.")
            passed = False

    if fp.get("n_rows", 0) < 100:
        risks.append("Very small dataset (<100 rows) — use repeated stratified CV and simpler models.")
        passed = False
    elif fp.get("n_rows", 0) < 200:
        metrics = fp.get("metrics", {})
        strong = metrics.get("roc_auc", 0) >= 0.9 or metrics.get("r2", 0) >= 0.85
        if strong and not fp.get("is_imbalanced"):
            risks.append("Small dataset (150–200 rows) — prefer stratified k-fold CV over a single holdout split.")
        else:
            risks.append("Small dataset — validate with repeated stratified CV, not a single holdout.")
            passed = False

    state["model_recommendations"] = recs
    state["risks"] = list(dict.fromkeys(risks))
    state["critic_passed"] = passed
    if not passed:
        state["confidence"] = max(0.35, state.get("confidence", 0.5) - 0.12)
    return state


def _route_after_critic(state: AgentState) -> str:
    if state.get("critic_passed") or state.get("critic_retried"):
        return "synthesize"
    if EvidenceRouter.internal_retrieval_is_weak(state.get("retrieved_docs", [])):
        return "web_search"
    return "expand_rag"


def _node_synthesize(state: AgentState) -> AgentState:
    llm_used = False
    llm_provider: str | None = None
    report = _structured_report(state)

    settings_enable = True
    try:
        from app.core.config import get_settings
        from app.services.llm_provider import build_advisor_llm, resolve_llm_provider

        llm_provider = resolve_llm_provider()
        settings_enable = get_settings().agent.get("enable_llm", True)
        if settings_enable and llm_provider != "rules":
            llm = build_advisor_llm()
            if llm is not None:
                try:
                    report = _llm_report(state, llm)
                    llm_used = True
                except Exception as exc:
                    logger.warning("LLM synthesis failed: %s", exc)
    except Exception as exc:
        logger.warning("Advisor LLM setup failed: %s", exc)

    state["narrative_report"] = report
    state["llm_used"] = llm_used
    state["llm_provider"] = llm_provider if llm_used else None
    state["retrieved_docs"] = _dedupe_retrieved_docs(state.get("retrieved_docs", []))
    return state


def _docs_for_report(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer playbooks and relevant run memory for narrative evidence."""
    playbooks = [d for d in docs if d.get("chunk_type") == "playbook"]
    runs = sorted(
        [d for d in docs if d.get("chunk_type") == "run_memory"],
        key=lambda d: d.get("run_relevance", 0),
        reverse=True,
    )
    other = [d for d in docs if d.get("chunk_type") not in ("playbook", "run_memory")]
    return (playbooks + runs[:2] + other)[:10]


def _structured_report(state: AgentState) -> str:
    mode = state.get("mode", "pre_train")
    fp = state.get("fingerprint", {})
    report_docs = _docs_for_report(state.get("retrieved_docs", []))
    lines = [
        f"# ML Advisor Report ({'Post-Training Evidence' if mode == 'post_train' else 'Pre-Training'})",
        f"\n**Confidence:** {state.get('confidence', 0.5):.0%} · **Critic:** {'passed' if state.get('critic_passed') else 'flagged — expanded search'}"
        + (" (web/arXiv/Kaggle evidence)" if state.get("web_evidence_used") else ""),
        f"**Target:** {fp.get('target_column', '—')} · **Task:** {state.get('task_type', '—')} · **Size:** {fp.get('row_bucket', '—')}",
        "\n## Key Insights",
        *[f"- {i}" for i in state.get("data_insights", [])],
        "\n## Model Recommendations",
    ]
    for rec in state.get("model_recommendations", []):
        lines.append(
            f"- **{rec['model']}** ({rec.get('confidence', 0):.0%}) — {rec.get('rationale', '')} [{rec.get('source', '')}]"
        )

    if state.get("risks"):
        lines.extend(["\n## Risks", *[f"- ⚠ {r}" for r in state["risks"]]])
    if state.get("top_actions"):
        lines.extend(["\n## Recommended Actions", *[f"- {a['action']} _(source: {a['source']})_" for a in state["top_actions"]]])
    if state.get("preprocessing_tips"):
        lines.extend(["\n## Preprocessing & Playbook Tips", *[f"- {t}" for t in state["preprocessing_tips"]]])

    lines.extend(["\n## Evidence Sources (RAG)"])
    for doc in report_docs[:8]:
        excerpt = doc["content"][:120].replace("\n", " ")
        rel = f", relevance={doc['run_relevance']:.0%}" if doc.get("run_relevance") is not None else ""
        lines.append(
            f"- **[{doc.get('category', 'general')}]** {doc['title']} — _{excerpt}…_ "
            f"(score={doc['score']}{rel}, {doc['source']})"
        )

    lines.append("\n---\n*LangGraph: fingerprint → hybrid RAG → evidence → critic [→ expand/web] → report*")
    return "\n".join(lines)


def _llm_report(state: AgentState, llm) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    doc_blocks = []
    for d in _docs_for_report(state.get("retrieved_docs", [])):
        doc_blocks.append(f"[{d.get('category')}] {d['title']} ({d['source']}):\n{d['content'][:400]}")

    fp = state.get("fingerprint", {})
    prompt = f"""Write a concise senior MLOps executive summary grounded ONLY in this evidence.

Mode: {state.get('mode')}
Target: {fp.get('target_column')} · Task: {state.get('task_type')} · Rows: {fp.get('n_rows')}
Fingerprint: {fp}
Insights: {state.get('data_insights')}
Recommendations: {state.get('model_recommendations')}
Risks: {state.get('risks')}
Actions: {state.get('top_actions')}
Preprocessing: {state.get('preprocessing_tips')}

RAG documents (only cite past runs with high structural similarity to this dataset):
{chr(10).join(doc_blocks)}

Use markdown with ONLY these sections (no duplicates):
## Executive Summary
## Why This Model
## What To Do Next

250-350 words total. In post-train mode clearly separate CV scores from holdout test metrics.
Do NOT repeat the full insights list, preprocessing tips, or RAG source list — the UI shows those separately.
Do NOT add "Models", "Similar Runs", "Preprocessing", or "RAG Documents" sections.
Cite sources inline as [playbook:name] or [run:job_id]. Do not invent metrics."""

    response = llm.invoke([
        SystemMessage(content="You are a senior MLOps engineer. Never invent metrics or model scores not in the evidence."),
        HumanMessage(content=prompt),
    ])
    return str(response.content)


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("build_fingerprint", _node_fingerprint)
    graph.add_node("multi_rag", _node_multi_rag)
    graph.add_node("expand_rag", _node_expand_rag)
    graph.add_node("web_search", _node_web_search)
    graph.add_node("evidence_merge", _node_evidence_merge)
    graph.add_node("critic", _node_critic)
    graph.add_node("synthesize", _node_synthesize)

    graph.set_entry_point("build_fingerprint")
    graph.add_edge("build_fingerprint", "multi_rag")
    graph.add_edge("multi_rag", "evidence_merge")
    graph.add_edge("evidence_merge", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"expand_rag": "expand_rag", "web_search": "web_search", "synthesize": "synthesize"},
    )
    graph.add_edge("expand_rag", "evidence_merge")
    graph.add_edge("web_search", "evidence_merge")
    graph.add_edge("synthesize", END)
    return graph.compile()


_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = _build_graph()
    return _agent_graph


class MLAdvisorAgent:
    def analyze(
        self,
        profile: dict[str, Any],
        target_column: str | None = None,
        task_type: str | None = None,
        job_result: dict[str, Any] | None = None,
        job_evidence: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        graph = get_agent_graph()
        initial: AgentState = {
            "profile": profile,
            "target_column": target_column,
            "task_type": task_type,
            "current_job_id": job_id or (job_evidence or {}).get("job_id"),
            "job_result": job_result,
            "job_evidence": job_evidence or {},
            "retrieved_docs": [],
            "playbook_excerpts": [],
            "data_insights": [],
            "model_recommendations": [],
            "preprocessing_tips": [],
            "top_actions": [],
            "risks": [],
            "confidence": 0.5,
            "critic_passed": True,
            "critic_retried": False,
            "web_evidence_used": False,
            "narrative_report": "",
            "llm_used": False,
        }
        result = graph.invoke(initial)
        recs = result.get("model_recommendations", [])
        return {
            "mode": result.get("mode", "pre_train"),
            "task_type": result.get("task_type"),
            "fingerprint": result.get("fingerprint", {}),
            "data_insights": result.get("data_insights", []),
            "model_recommendations": [r["model"] if isinstance(r, dict) else r for r in recs],
            "model_recommendations_detail": recs,
            "preprocessing_tips": result.get("preprocessing_tips", []),
            "retrieved_docs": result.get("retrieved_docs", []),
            "top_actions": result.get("top_actions", []),
            "risks": result.get("risks", []),
            "confidence": result.get("confidence", 0.5),
            "critic_passed": result.get("critic_passed", True),
            "web_evidence_used": result.get("web_evidence_used", False),
            "narrative_report": result.get("narrative_report", ""),
            "llm_used": result.get("llm_used", False),
            "llm_provider": result.get("llm_provider"),
        }
