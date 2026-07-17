"""Auto-generate model card documentation after training."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelCardGenerator:
    def generate(
        self,
        *,
        job_id: str,
        registry_entry: dict[str, Any],
        profile: dict | None = None,
        feature_selection: dict | None = None,
        leakage_report: dict | None = None,
        model_review: dict | None = None,
        business_insights: dict | None = None,
        advisor_report: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        card = {
            "model_details": {
                "job_id": job_id,
                "model_name": registry_entry.get("model_name"),
                "task_type": registry_entry.get("task_type"),
                "target_column": registry_entry.get("target_column"),
                "dataset_id": registry_entry.get("dataset_id"),
                "created_at": registry_entry.get("created_at", now),
                "version": "1.0.0",
            },
            "dataset_summary": {
                "n_rows": profile.get("n_rows") if profile else registry_entry.get("n_rows"),
                "n_columns": profile.get("n_columns") if profile else None,
                "feature_columns": registry_entry.get("feature_columns", []),
            },
            "training": {
                "best_params": registry_entry.get("best_params"),
                "baseline_scores": registry_entry.get("baseline_scores"),
                "feature_selection": feature_selection,
                "elapsed_seconds": registry_entry.get("elapsed_seconds"),
            },
            "evaluation": {
                "metrics": registry_entry.get("metrics"),
            },
            "explainability": {
                "top_features": registry_entry.get("top_features", [])[:10],
            },
            "data_quality": {
                "leakage_report": leakage_report,
            },
            "model_review": model_review,
            "business_insights": business_insights,
            "limitations": _limitations(registry_entry, leakage_report),
            "bias_considerations": _bias_notes(profile, registry_entry),
            "deployment": {
                "predict_endpoint": f"/predict/{job_id}",
                "batch_endpoint": f"/predict/{job_id}/batch",
                "feature_importance_endpoint": f"/predict/{job_id}/feature-importance",
                "drift_endpoint": f"/monitor/drift/{job_id}",
            },
        }
        markdown = self._to_markdown(card, advisor_report)
        return {"card": card, "markdown": markdown}

    def save(self, job_dir: Path, card_data: dict[str, Any]) -> dict[str, str]:
        job_dir.mkdir(parents=True, exist_ok=True)
        json_path = job_dir / "model_card.json"
        md_path = job_dir / "model_card.md"
        json_path.write_text(json.dumps(card_data["card"], indent=2, default=str), encoding="utf-8")
        md_path.write_text(card_data["markdown"], encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    def _to_markdown(self, card: dict, advisor_report: str | None) -> str:
        m = card["model_details"]
        lines = [
            f"# Model Card — {m['job_id']}",
            "",
            f"**Model:** {m['model_name']} · **Task:** {m['task_type']} · **Target:** `{m['target_column']}`",
            f"**Dataset:** `{m['dataset_id']}` · **Version:** {m['version']}",
            "",
            "## Dataset Summary",
            f"- Rows: {card['dataset_summary'].get('n_rows', '—')}",
            f"- Features: {len(card['dataset_summary'].get('feature_columns', []))}",
            "",
            "## Evaluation Metrics",
        ]
        for k, v in (card["evaluation"].get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k}:** {v:.4f}")
        lines.extend(["", "## Hyperparameters"])
        for k, v in (card["training"].get("best_params") or {}).items():
            lines.append(f"- `{k}` = {v}")
        if card["explainability"].get("top_features"):
            lines.extend(["", "## Top Feature Drivers"])
            for f in card["explainability"]["top_features"][:5]:
                lines.append(f"- {f.get('feature')} (importance={f.get('importance', '—')})")
        if card.get("limitations"):
            lines.extend(["", "## Limitations"])
            for lim in card["limitations"]:
                lines.append(f"- {lim}")
        if card.get("bias_considerations"):
            lines.extend(["", "## Bias Considerations"])
            for b in card["bias_considerations"]:
                lines.append(f"- {b}")
        lines.extend(["", "## Deployment"])
        for k, v in card["deployment"].items():
            lines.append(f"- {k}: `{v}`")
        if advisor_report:
            lines.extend(["", "## Advisor Notes", advisor_report[:1500]])
        return "\n".join(lines)


def _limitations(entry: dict, leakage: dict | None) -> list[str]:
    lim = []
    n_rows = entry.get("n_rows") or 0
    if n_rows and n_rows < 1000:
        lim.append(f"Trained on {n_rows} rows — performance may not generalize to larger populations.")
    if leakage and leakage.get("leakage_detected"):
        lim.append("Leakage signals were present — validate holdout metrics on clean features.")
    lim.append("Model performance depends on reference data distribution — monitor drift in production.")
    return lim


def _bias_notes(profile: dict | None, entry: dict) -> list[str]:
    notes = ["Review feature fairness across demographic groups before deployment."]
    if profile and profile.get("categorical_columns"):
        notes.append("Categorical features may encode proxy variables — audit for protected attributes.")
    if entry.get("task_type") == "classification":
        notes.append("Check precision/recall across minority classes, not just overall accuracy.")
    return notes
