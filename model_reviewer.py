"""AI Model Reviewer — structured post-training analysis."""
from __future__ import annotations

from typing import Any

from app.services.llm_provider import build_advisor_llm, resolve_llm_provider


class ModelReviewer:
    def review(
        self,
        *,
        job_id: str,
        model_name: str,
        task_type: str,
        metrics: dict[str, Any],
        baseline_scores: dict[str, float],
        best_params: dict[str, Any],
        feature_importance: list[dict],
        n_train: int,
        n_test: int,
        feature_selection: dict | None = None,
        leakage_report: dict | None = None,
    ) -> dict[str, Any]:
        strengths: list[str] = []
        weaknesses: list[str] = []
        recommendations: list[str] = []

        sorted_lb = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
        winner_score = sorted_lb[0][1] if sorted_lb else 0
        runner_up = sorted_lb[1] if len(sorted_lb) > 1 else None
        gap = winner_score - runner_up[1] if runner_up else 0.1

        if gap >= 0.02:
            strengths.append(f"Clear CV winner — {gap:.4f} gap over runner-up")
        elif runner_up:
            weaknesses.append(f"Runner-up {runner_up[0]} within {gap:.4f} CV of winner — consider ensemble")
            recommendations.append(f"Try ensemble of {model_name} + {runner_up[0]}")

        if winner_score >= 0.85:
            strengths.append("Strong cross-validation performance")
        elif winner_score < 0.7:
            weaknesses.append("Moderate CV score — model may need more tuning or features")
            recommendations.append("Increase Optuna trials or try gradient boosting family")

        if task_type == "classification":
            acc = metrics.get("accuracy", 0)
            auc = metrics.get("roc_auc", 0)
            f1 = metrics.get("f1_weighted", 0)
            if acc > 0.95 and auc < 0.75:
                weaknesses.append("High accuracy but low AUC — possible majority-class bias")
                recommendations.append("Use class weights and optimize F1 / AUC instead of accuracy")
            if f1 >= 0.85:
                strengths.append("Solid F1-weighted on holdout")
            if auc >= 0.9:
                strengths.append("High holdout ROC-AUC")

        if n_train < 500:
            weaknesses.append(f"Small training set ({n_train} rows) — watch for overfitting")
            recommendations.append("Prefer stratified k-fold CV and simpler models")

        if feature_selection and feature_selection.get("removed_count", 0) > 0:
            strengths.append(
                f"Feature selection reduced {feature_selection['original_count']} → "
                f"{feature_selection['selected_count']} features"
            )

        if leakage_report and leakage_report.get("leakage_detected"):
            weaknesses.append("Data leakage signals were detected — verify holdout metrics are realistic")
            recommendations.append("Re-run after dropping leakage columns: " + ", ".join(
                leakage_report.get("recommended_drop", [])[:3]
            ))

        top_feats = [f["feature"] for f in feature_importance[:3]]
        if len(top_feats) >= 2:
            strengths.append(f"Top drivers: {', '.join(top_feats)}")

        # Overfitting heuristic: train vs test gap if available
        if metrics.get("train_score") and metrics.get("test_score"):
            diff = metrics["train_score"] - metrics["test_score"]
            if diff > 0.08:
                weaknesses.append(f"Possible overfitting (train-test gap {diff:.3f})")
                recommendations.append("Increase regularization or reduce model complexity")

        if not recommendations:
            recommendations.append("Monitor production drift weekly on top features")
            recommendations.append("Log this run to MLflow for experiment tracking")

        llm_narrative = self._llm_narrative(
            job_id, model_name, task_type, metrics, strengths, weaknesses, recommendations, best_params
        )

        return {
            "job_id": job_id,
            "model_name": model_name,
            "task_type": task_type,
            "overall_verdict": "pass" if len(weaknesses) <= 1 else "review",
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "narrative": llm_narrative,
        }

    def _llm_narrative(
        self,
        job_id: str,
        model_name: str,
        task_type: str,
        metrics: dict,
        strengths: list[str],
        weaknesses: list[str],
        recommendations: list[str],
        best_params: dict,
    ) -> str:
        if resolve_llm_provider() == "rules":
            return self._template_narrative(job_id, model_name, strengths, weaknesses, recommendations)

        llm = build_advisor_llm()
        if llm is None:
            return self._template_narrative(job_id, model_name, strengths, weaknesses, recommendations)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = f"""Write a concise model review for job {job_id}.
Model: {model_name} ({task_type})
Metrics: {metrics}
Best params: {best_params}
Strengths: {strengths}
Weaknesses: {weaknesses}
Recommendations: {recommendations}

Use markdown with sections: ## Model Review, ### Strengths, ### Weaknesses, ### Recommendations.
150-200 words. Do not invent metrics."""
            response = llm.invoke([
                SystemMessage(content="You are a senior ML engineer reviewing a trained model."),
                HumanMessage(content=prompt),
            ])
            return str(response.content)
        except Exception:
            return self._template_narrative(job_id, model_name, strengths, weaknesses, recommendations)

    @staticmethod
    def _template_narrative(
        job_id: str,
        model_name: str,
        strengths: list[str],
        weaknesses: list[str],
        recommendations: list[str],
    ) -> str:
        lines = [f"## Model Review — `{job_id}`", f"**Model:** {model_name.replace('_', ' ')}\n"]
        lines.append("### Strengths")
        lines.extend(f"- ✔ {s}" for s in strengths or ["No major strengths flagged"])
        lines.append("\n### Weaknesses")
        lines.extend(f"- ✖ {w}" for w in weaknesses or ["None identified"])
        lines.append("\n### Recommendations")
        lines.extend(f"- {r}" for r in recommendations)
        return "\n".join(lines)
