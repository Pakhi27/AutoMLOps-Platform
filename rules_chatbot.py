"""Rule-based chat fallback — no LLM API required."""
from __future__ import annotations

import re

from app.services.ml_chatbot_agent import (
    ask_ml_advisor,
    check_drift,
    get_feature_importance,
    get_job_status,
    list_recent_jobs,
    trigger_retrain,
)
from app.services.model_registry import get_model_registry

_JOB_ID = re.compile(r"job_[a-f0-9]{8,12}", re.I)
_DS_ID = re.compile(r"ds_[a-f0-9]{8,12}", re.I)

HELP_TEXT = """Rule-based chat (no LLM). Try:
• list jobs
• status job_xxxxxxxx
• features job_xxxxxxxx
• drift job_xxx ds_yyyyyyyy  (needs both ids)
• advisor job_xxxxxxxx  — ML advisor report
• retrain job_xxxxxxxx
• help"""


def _latest_job_id() -> str | None:
    registry = get_model_registry().list_all()
    if not registry:
        return None
    return list(registry.keys())[-1]


def _extract_job(text: str) -> str | None:
    m = _JOB_ID.search(text)
    if m:
        return m.group(0)
    msg = text.lower()
    if any(k in msg for k in ("latest job", "last job", "recent job", "this job", "my job", "current job")):
        return _latest_job_id()
    return None


def _extract_dataset(text: str) -> str | None:
    m = _DS_ID.search(text)
    return m.group(0) if m else None


def handle_rules_chat(message: str) -> str:
    msg = message.lower().strip()
    if not msg:
        return HELP_TEXT
    if msg in ("help", "?", "commands"):
        return HELP_TEXT

    job_id = _extract_job(message)

    if any(k in msg for k in ("list job", "list jobs", "show jobs", "my jobs", "recent jobs")):
        return list_recent_jobs.invoke({"limit": 10})

    if job_id and any(k in msg for k in ("feature", "importance", "shap", "drivers")):
        return get_feature_importance.invoke({"job_id": job_id})

    if job_id and any(k in msg for k in ("drift", "shift", "production")):
        ds_id = _extract_dataset(message)
        if not ds_id:
            return (
                f"To check drift I need a new dataset id too.\n"
                f"Example: drift {job_id} ds_abc12345\n"
                f"(Upload the new CSV first in Step 1.)"
            )
        return check_drift.invoke({"job_id": job_id, "current_dataset_id": ds_id})

    if job_id and any(k in msg for k in ("advisor", "advice", "recommend", "why", "explain")):
        return ask_ml_advisor.invoke({"question": message, "job_id": job_id})

    if job_id and any(k in msg for k in ("retrain", "re-train", "retrain model")):
        return trigger_retrain.invoke({"job_id": job_id})

    if job_id and any(k in msg for k in ("status", "metrics", "model", "result", "info", "best")):
        return get_job_status.invoke({"job_id": job_id})

    if job_id:
        return get_job_status.invoke({"job_id": job_id})

    if any(k in msg for k in ("latest job", "last job", "recent job", "this job", "my job", "current job")):
        return "No training jobs found yet. Run Step 4 (Train) first, then ask again."

    return (
        HELP_TEXT
        + "\n\nTip: include a job id like job_abc12345, or say \"list jobs\" to see available jobs."
    )
