"""Conversational ML chatbot — multi-turn tool-calling agent for the platform."""
from __future__ import annotations

import json
import threading
import uuid
from typing import Any

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import JobStatus
from app.services.drift_monitor import DriftMonitor
from app.services.feature_importance import load_or_compute
from app.services.ml_agent_graph import MLAdvisorAgent
from app.services.model_registry import get_model_registry
from app.store.job_store import get_job_store
from app.utils.io_utils import new_id, read_csv_safely

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the operating assistant for an AutoML platform. \
You can look up training jobs, feature importances, run drift checks, \
consult the ML advisor for grounded recommendations, and trigger retraining.

Rules:
- Never invent job ids, metrics, or feature names — only report what tools return.
- If the user references "that job" or "it", resolve from conversation context or ask to clarify.
- Prefer the most specific tool (get_feature_importance for "what drives predictions").
- Use ask_ml_advisor for open-ended "why" / "what should I do next" questions.
- Confirm before trigger_retrain unless the user explicitly asked to retrain.
- Summarize tool results in plain English — do not paste raw JSON to the user."""


def format_job_status(entry: dict[str, Any], job_id: str) -> str:
    """Human-readable job summary for chat UI and rule-based fallback."""
    model = entry.get("model_name") or "unknown"
    metrics = entry.get("metrics") or {}
    baseline = entry.get("baseline_scores") or {}
    params = entry.get("best_params") or {}

    lines = [
        f"**Winning model:** {model.replace('_', ' ')}",
        f"**Job:** `{job_id}` · **Target:** {entry.get('target_column', '—')} · **Task:** {entry.get('task_type', '—')}",
        f"**Dataset:** `{entry.get('dataset_id', '—')}`",
    ]

    holdout = []
    for key, label in (("roc_auc", "ROC-AUC"), ("f1_weighted", "F1"), ("accuracy", "Accuracy"), ("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE")):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            holdout.append(f"{label} **{val:.4f}**")
    if holdout:
        lines.append("**Holdout test:** " + " · ".join(holdout))

    if baseline:
        sorted_lb = sorted(baseline.items(), key=lambda x: x[1], reverse=True)
        lines.append("\n**CV leaderboard:**")
        for rank, (name, score) in enumerate(sorted_lb[:6], 1):
            tag = " ← winner" if name == model else ""
            lines.append(f"{rank}. {name.replace('_', ' ')} — **{score:.4f}**{tag}")

    if params:
        param_str = ", ".join(f"`{k}={v}`" for k, v in list(params.items())[:5])
        lines.append(f"\n**Best hyperparameters:** {param_str}")

    return "\n".join(lines)


def format_drift_result(result: dict[str, Any]) -> str:
    detected = result.get("dataset_drift_detected", False)
    share = result.get("drift_share", 0)
    drifted = result.get("number_of_drifted_columns", 0)
    total = result.get("number_of_columns", 0)
    status = "⚠ Drift detected" if detected else "✓ No significant drift"
    lines = [
        f"**{status}**",
        f"Drift share: **{share * 100:.1f}%** · Columns drifted: **{drifted}/{total}**",
    ]
    if result.get("retrain_message"):
        lines.append(result["retrain_message"])
    if result.get("retrain_job_id"):
        lines.append(f"New retrain job: `{result['retrain_job_id']}`")
    return "\n".join(lines)


@tool
def list_recent_jobs(limit: int = 10) -> str:
    """List recent training jobs with id, dataset, target, task type, model, and metrics."""
    registry = get_model_registry().list_all()
    items = list(registry.items())[-limit:]
    if not items:
        return "No jobs found in the registry yet."
    lines = []
    for job_id, entry in items:
        metrics = entry.get("metrics", {})
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:3] if isinstance(v, (int, float)))
        lines.append(
            f"- {job_id}: dataset={entry.get('dataset_id')}, target={entry.get('target_column')}, "
            f"task={entry.get('task_type')}, model={entry.get('model_name')}, metrics=[{metric_str}]"
        )
    return "\n".join(lines)


@tool
def get_job_status(job_id: str) -> str:
    """Get training status, winning model, hyperparameters, and holdout metrics for a job_id."""
    entry = get_model_registry().get(job_id)
    if entry is None:
        return f"No job found with id '{job_id}'. Use list_recent_jobs to see available ids."
    return format_job_status(entry, job_id)


@tool
def get_feature_importance(job_id: str) -> str:
    """Get top SHAP feature importances for a completed job."""
    try:
        importances = load_or_compute(job_id)
    except FileNotFoundError as exc:
        return str(exc)
    top = importances[:10]
    lines = [f"{i+1}. {f['feature']} — importance={f['importance']:.4f}" for i, f in enumerate(top)]
    return "\n".join(lines) if lines else "No feature importance available."


@tool
def check_drift(job_id: str, current_dataset_id: str) -> str:
    """Compare a job's training reference against a new dataset id for data drift."""
    settings = get_settings()
    entry = get_model_registry().get(job_id)
    if entry is None:
        return f"No job found with id '{job_id}'."

    ref_path = entry.get("reference_data_path") or str(settings.reference_dir / f"{job_id}.csv")
    current_path = settings.upload_dir / f"{current_dataset_id}.csv"
    if not current_path.exists():
        return f"Dataset '{current_dataset_id}' not found. Upload it first."

    reference = read_csv_safely(ref_path)
    current = read_csv_safely(current_path)
    monitor = DriftMonitor()
    output_html = settings.artifacts_dir / job_id / f"drift_vs_{current_dataset_id}.html"
    result = monitor.run_drift_report(reference, current, output_html)
    return format_drift_result(result)


@tool
def ask_ml_advisor(question: str, job_id: str = "") -> str:
    """RAG-grounded ML advisor for modeling advice. Pass job_id to use training evidence."""
    registry_entry = get_model_registry().get(job_id) if job_id else None
    profile: dict[str, Any] = {"n_rows": 0, "n_columns": 0, "numeric_columns": [], "categorical_columns": [], "datetime_columns": [], "missing_values": {}}
    job_result = None
    if registry_entry:
        job_result = {
            "model_name": registry_entry.get("model_name"),
            "task_type": registry_entry.get("task_type"),
            "metrics": registry_entry.get("metrics"),
            "baseline_scores": registry_entry.get("baseline_scores"),
            "best_params": registry_entry.get("best_params"),
        }
    advisor = MLAdvisorAgent()
    result = advisor.analyze(
        profile=profile,
        target_column=registry_entry.get("target_column") if registry_entry else None,
        task_type=registry_entry.get("task_type") if registry_entry else None,
        job_result=job_result,
    )
    header = f"Question context: {question}\n\n" if question else ""
    return header + result["narrative_report"]


def _run_retrain_bg(new_job_id: str, dataset_id: str, target_column: str, task_type: str | None, parent_job_id: str) -> None:
    from app.services.pipeline_orchestrator import PipelineOrchestrator

    store = get_job_store()
    store.create(new_job_id, dataset_id, target_column)
    store.update(new_job_id, status=JobStatus.running.value, parent_job_id=parent_job_id)
    try:
        result = PipelineOrchestrator().run(
            job_id=new_job_id,
            dataset_id=dataset_id,
            target_column=target_column,
            task_type=task_type,
        )
        store.update(new_job_id, status=JobStatus.success.value, result=result, task_type=result.get("task_type"))
    except Exception as exc:
        logger.exception("Chatbot retrain %s failed", new_job_id)
        store.update(new_job_id, status=JobStatus.failed.value, error=str(exc))


@tool
def trigger_retrain(job_id: str) -> str:
    """Start a fresh training run using the same dataset and target as an existing job."""
    entry = get_model_registry().get(job_id)
    if entry is None:
        return f"No job found with id '{job_id}'."
    new_job_id = new_id("job_")
    threading.Thread(
        target=_run_retrain_bg,
        args=(new_job_id, entry["dataset_id"], entry["target_column"], entry.get("task_type"), job_id),
        daemon=True,
    ).start()
    return f"Retraining started as '{new_job_id}'. Check status with get_job_status('{new_job_id}')."


ALL_TOOLS = [list_recent_jobs, get_job_status, get_feature_importance, check_drift, ask_ml_advisor, trigger_retrain]

_agent = None
_checkpointer = None


def chat_available() -> bool:
    from app.services.llm_provider import chat_available as _available

    return _available()


def is_rules_mode() -> bool:
    from app.services.llm_provider import resolve_llm_provider

    return resolve_llm_provider() == "rules"


def get_chat_agent():
    global _agent, _checkpointer
    from app.services.llm_provider import build_chat_llm, resolve_llm_provider

    if resolve_llm_provider() == "rules":
        raise RuntimeError("Rules mode active — use handle_rules_chat instead of get_chat_agent.")

    if _agent is None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        llm = build_chat_llm()
        if llm is None:
            raise RuntimeError("No LLM configured for chat agent.")
        _checkpointer = MemorySaver()
        _agent = create_react_agent(
            llm,
            tools=ALL_TOOLS,
            checkpointer=_checkpointer,
            state_modifier=SYSTEM_PROMPT,
        )
    return _agent
