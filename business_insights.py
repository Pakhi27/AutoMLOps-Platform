"""Business-friendly insight generator from SHAP + EDA."""
from __future__ import annotations

from typing import Any

from app.services.llm_provider import build_advisor_llm, resolve_llm_provider


class BusinessInsightGenerator:
    def generate(
        self,
        *,
        target_column: str,
        task_type: str,
        feature_importance: list[dict],
        profile: dict | None = None,
        metrics: dict | None = None,
        eda: dict | None = None,
    ) -> dict[str, Any]:
        top_features = [f["feature"] for f in feature_importance[:5]]
        drivers = []
        for f in feature_importance[:5]:
            drivers.append({
                "feature": f["feature"],
                "importance": f.get("importance", 0),
                "rank": f.get("rank", 0),
            })

        bullets = self._rule_based_insights(top_features, target_column, task_type, eda, metrics)
        summary = self._llm_summary(target_column, task_type, drivers, bullets, metrics)

        return {
            "target_column": target_column,
            "task_type": task_type,
            "top_drivers": drivers,
            "business_bullets": bullets,
            "executive_summary": summary,
        }

    def _rule_based_insights(
        self,
        top_features: list[str],
        target: str,
        task_type: str,
        eda: dict | None,
        metrics: dict | None,
    ) -> list[str]:
        bullets: list[str] = []
        if top_features:
            bullets.append(
                f"**{top_features[0].replace('_', ' ').title()}** is the strongest driver of `{target}`."
            )
        if len(top_features) >= 2:
            bullets.append(
                f"**{top_features[1].replace('_', ' ').title()}** and "
                f"**{top_features[2].replace('_', ' ').title() if len(top_features) > 2 else top_features[1]}** "
                f"are the next most influential factors."
            )
        if eda and eda.get("correlation_with_target"):
            top_corr = eda["correlation_with_target"][0]
            bullets.append(
                f"Numeric feature `{top_corr['feature']}` shows r={top_corr['correlation']:.3f} with {target}."
            )
        if metrics and task_type == "classification" and metrics.get("roc_auc"):
            bullets.append(
                f"Model separates classes well (ROC-AUC {metrics['roc_auc']:.2f}) — suitable for ranking / prioritization use cases."
            )
        if not bullets:
            bullets.append("Run training with SHAP enabled to populate business drivers.")
        return bullets

    def _llm_summary(
        self,
        target: str,
        task_type: str,
        drivers: list[dict],
        bullets: list[str],
        metrics: dict | None,
    ) -> str:
        if resolve_llm_provider() == "rules":
            return "## Business Summary\n\n" + "\n".join(f"- {b}" for b in bullets)

        llm = build_advisor_llm()
        if llm is None:
            return "## Business Summary\n\n" + "\n".join(f"- {b}" for b in bullets)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = f"""Write a business-friendly summary for non-technical stakeholders.
Target: {target} ({task_type})
Top drivers: {drivers}
Key points: {bullets}
Metrics: {metrics}

Use markdown. 3-5 bullet points in plain language. No jargon like SHAP or hyperparameters."""
            response = llm.invoke([
                SystemMessage(content="You explain ML results to business managers."),
                HumanMessage(content=prompt),
            ])
            return str(response.content)
        except Exception:
            return "## Business Summary\n\n" + "\n".join(f"- {b}" for b in bullets)
