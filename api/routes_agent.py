"""AI Agent API: evidence-grounded LangGraph advisor with hybrid RAG."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import AgentAnalyzeRequest, AgentAnalyzeResponse, KnowledgeTopic
from app.services.modality.dataset_analysis import (
    build_image_profile,
    load_tabular_dataframe,
    resolve_dataset_path,
)
from app.services.modality.detector import load_modality_metadata
from app.services.data_profiler import DataProfiler
from app.services.eda_service import EDAService
from app.services.ml_agent_graph import MLAdvisorAgent
from app.services.model_registry import get_model_registry
from app.services.model_selector import ModelSelector
from app.services.rag_knowledge import get_knowledge_base
from app.store.job_store import get_job_store

router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(__name__)


def _enrich_profile(profile: dict, df, target_column: str) -> dict:
    eda = EDAService().analyze(df, target_column)
    profile["target_analysis"] = eda["target_analysis"]
    profile["correlation_with_target"] = eda["correlation_with_target"]
    return profile


def _load_job_evidence(job_id: str, job_result: dict, reg_entry: dict) -> dict:
    settings = get_settings()
    job_dir = settings.artifacts_dir / job_id
    evidence: dict = {}

    fi_path = job_dir / "feature_importance.json"
    if fi_path.exists():
        evidence["feature_importance"] = json.loads(fi_path.read_text(encoding="utf-8"))[:10]
    elif job_result.get("feature_importance"):
        evidence["feature_importance"] = job_result["feature_importance"][:10]

    tuning_path = job_dir / "tuning_history.json"
    if tuning_path.exists():
        history = json.loads(tuning_path.read_text(encoding="utf-8"))
        if history:
            best = max(history, key=lambda t: t.get("value", float("-inf")))
            evidence["best_trial"] = best

    if reg_entry.get("metrics"):
        evidence["registry_metrics"] = reg_entry["metrics"]
    return evidence


def _build_job_result(job: dict, reg_entry: dict) -> dict:
    result = dict(job.get("result") or {})
    result.setdefault("model_name", reg_entry.get("model_name"))
    result.setdefault("task_type", reg_entry.get("task_type"))
    result.setdefault("metrics", reg_entry.get("metrics", {}))
    result.setdefault("baseline_scores", reg_entry.get("baseline_scores", {}))
    result.setdefault("best_params", reg_entry.get("best_params", result.get("best_params", {})))
    result.setdefault("feature_importance", result.get("feature_importance", []))
    return result


@router.get("/models", summary="List available ML algorithms")
async def list_models() -> dict[str, list[str]]:
    return {
        "classification": ModelSelector.available_models("classification"),
        "regression": ModelSelector.available_models("regression"),
    }


@router.get("/knowledge", response_model=list[KnowledgeTopic], summary="List RAG knowledge topics")
async def list_knowledge() -> list[KnowledgeTopic]:
    kb = get_knowledge_base()
    return [KnowledgeTopic(source=t["source"], title=t["title"]) for t in kb.list_topics()]


@router.post("/knowledge/search", summary="Search RAG knowledge base")
async def search_knowledge(query: str = Query(...), top_k: int = 4) -> list[dict]:
    kb = get_knowledge_base()
    chunks = kb.retrieve(query, top_k=top_k)
    return [
        {"source": c.source, "title": c.title, "content": c.content, "score": round(c.score, 4)}
        for c in chunks
    ]


@router.post("/analyze", response_model=AgentAnalyzeResponse, summary="LangGraph ML advisor (pre or post-train)")
async def analyze_dataset(request: AgentAnalyzeRequest) -> AgentAnalyzeResponse:
    settings = get_settings()
    profiler = DataProfiler()
    agent = MLAdvisorAgent()
    job_result = None
    job_evidence = None
    target_column = request.target_column
    profile = request.profile

    if request.job_id:
        store = get_job_store()
        job = store.get(request.job_id)
        registry = get_model_registry()
        reg_entry = registry.get(request.job_id)
        if job is None or reg_entry is None:
            raise HTTPException(status_code=404, detail=f"Job '{request.job_id}' not found or not trained.")
        target_column = target_column or job["target_column"]
        dataset_id = job["dataset_id"]
        meta = load_modality_metadata(settings.upload_dir, dataset_id)
        modality = reg_entry.get("modality") or meta.get("modality", "tabular")

        if modality == "image":
            profile = build_image_profile(settings.upload_dir, dataset_id, target_column, meta)
        else:
            df = load_tabular_dataframe(settings.upload_dir, dataset_id)
            if df is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{dataset_id}' has no tabular file for advisor profiling.",
                )
            profile = profiler.profile(df)
            profile = _enrich_profile(profile, df, target_column)
        profile["modality"] = modality

        job_result = _build_job_result(job, reg_entry)
        job_evidence = _load_job_evidence(request.job_id, job_result, reg_entry)
        job_evidence["job_id"] = request.job_id
        get_knowledge_base().reload()
    elif request.profile:
        profile = request.profile
    elif request.dataset_id:
        meta = load_modality_metadata(settings.upload_dir, request.dataset_id)
        modality = meta.get("modality", "tabular")
        if modality == "image":
            profile = build_image_profile(
                settings.upload_dir, request.dataset_id, target_column or "label", meta
            )
        else:
            df = load_tabular_dataframe(settings.upload_dir, request.dataset_id)
            if df is None:
                path = resolve_dataset_path(settings.upload_dir, request.dataset_id)
                if path is None:
                    raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")
                raise HTTPException(
                    status_code=422,
                    detail=f"Dataset '{request.dataset_id}' is not tabular — advisor uses image/text summaries.",
                )
            profile = profiler.profile(df)
            if target_column:
                profile = _enrich_profile(profile, df, target_column)
        profile["modality"] = modality
    else:
        raise HTTPException(status_code=400, detail="Provide job_id, dataset_id, or profile.")

    logger.info("Running evidence-grounded advisor (mode=%s)", "post_train" if job_result else "pre_train")
    result = agent.analyze(
        profile=profile,
        target_column=target_column,
        task_type=request.task_type.value if request.task_type else None,
        job_result=job_result,
        job_evidence=job_evidence,
        job_id=request.job_id,
    )
    return AgentAnalyzeResponse(**result)


@router.post("/analyze/job/{job_id}", response_model=AgentAnalyzeResponse, summary="Post-train advisor for a job")
async def analyze_job(job_id: str) -> AgentAnalyzeResponse:
    return await analyze_dataset(AgentAnalyzeRequest(job_id=job_id))
