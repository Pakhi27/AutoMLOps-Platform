"""Generate matplotlib EDA charts (saved as PNG, served by the API)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.config import get_settings

MAX_NUMERIC_FEATURES = 6
MAX_CATEGORICAL_FEATURES = 6
MAX_TARGET_RELATIONSHIP_CHARTS = 3


class EDAChartGenerator:
    """Builds PNG charts for target + each feature vs target."""

    def __init__(self, dataset_id: str, target_column: str) -> None:
        self.dataset_id = dataset_id
        self.target_column = target_column
        settings = get_settings()
        self.out_dir = settings.artifacts_dir / "eda" / dataset_id / target_column
        self.out_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    @staticmethod
    def _safe_name(text: str) -> str:
        return re.sub(r"[^\w\-]", "_", text)[:80]

    def generate_all(self, df: pd.DataFrame, eda: dict[str, Any]) -> list[dict[str, str]]:
        charts: list[dict[str, str]] = []
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        task_type = eda["task_type"]
        eda = self._limit_eda_for_charts(eda)

        charts.append(self._chart_target(y, task_type))
        charts.extend(self._charts_numeric(X, y, task_type, eda))
        charts.extend(self._charts_categorical(X, y, task_type, eda))

        if eda.get("correlation_with_target"):
            charts.append(self._chart_correlations(eda["correlation_with_target"]))

        return [c for c in charts if c]

    @staticmethod
    def _limit_eda_for_charts(eda: dict[str, Any]) -> dict[str, Any]:
        """Cap chart count on wide datasets so EDA finishes in reasonable time."""
        limited = dict(eda)
        numeric = eda.get("numeric_features", {})
        if len(numeric) > MAX_NUMERIC_FEATURES:
            ranked = [c["feature"] for c in eda.get("correlation_with_target", []) if c["feature"] in numeric]
            for col in numeric:
                if col not in ranked:
                    ranked.append(col)
            keep = ranked[:MAX_NUMERIC_FEATURES]
            limited["numeric_features"] = {k: numeric[k] for k in keep if k in numeric}

        categorical = eda.get("categorical_features", {})
        if len(categorical) > MAX_CATEGORICAL_FEATURES:
            ranked = sorted(
                categorical,
                key=lambda c: categorical[c].get("n_unique", 0),
                reverse=True,
            )[:MAX_CATEGORICAL_FEATURES]
            limited["categorical_features"] = {k: categorical[k] for k in ranked}

        return limited

    def _save(self, fig: plt.Figure, name: str, title: str, chart_type: str) -> dict[str, str]:
        path = self.out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#1a2332")
        plt.close(fig)
        return {
            "id": name,
            "title": title,
            "chart_type": chart_type,
            "url": f"/datasets/{self.dataset_id}/eda/chart/{self.target_column}/{name}.png",
        }

    def _chart_target(self, y: pd.Series, task_type: str) -> dict[str, str]:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1a2332")
        ax.set_facecolor("#243044")
        if task_type == "classification":
            counts = y.value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
            ax.pie(
                counts.values,
                labels=counts.index.astype(str),
                autopct="%1.1f%%",
                colors=colors,
                textprops={"color": "white"},
            )
            ax.set_title(f"Target: {self.target_column} (class balance)", color="white", fontsize=11)
            return self._save(fig, "target_pie", f"Target — {self.target_column}", "pie")
        series = pd.to_numeric(y, errors="coerce").dropna()
        ax.hist(series, bins=min(25, max(8, series.nunique())), color="#3b82f6", edgecolor="white", alpha=0.85)
        ax.set_xlabel(self.target_column, color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title(f"Target distribution — {self.target_column}", color="white", fontsize=11)
        ax.tick_params(colors="white")
        return self._save(fig, "target_hist", f"Target — {self.target_column}", "histogram")

    def _charts_numeric(self, X: pd.DataFrame, y: pd.Series, task_type: str, eda: dict) -> list[dict[str, str]]:
        charts = []
        for idx, (col, info) in enumerate(eda.get("numeric_features", {}).items()):
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            if series.empty:
                continue

            # Histogram
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#1a2332")
            ax.set_facecolor("#243044")
            ax.hist(series, bins=min(20, max(6, series.nunique())), color="#60a5fa", edgecolor="#1e3a5f", alpha=0.9)
            ax.set_title(f"{col} — distribution", color="white", fontsize=10)
            ax.set_xlabel(col, color="white", fontsize=9)
            ax.tick_params(colors="white")
            charts.append(self._save(fig, f"num_{self._safe_name(col)}_hist", f"{col} histogram", "histogram"))

            if idx >= MAX_TARGET_RELATIONSHIP_CHARTS:
                continue

            if task_type == "classification" and info.get("mean_by_class"):
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#1a2332")
                ax.set_facecolor("#243044")
                classes = list(info["mean_by_class"].keys())
                means = list(info["mean_by_class"].values())
                ax.bar(classes, means, color="#22c55e", edgecolor="white")
                ax.set_title(f"{col} — mean by {self.target_column}", color="white", fontsize=10)
                ax.set_ylabel(f"Mean {col}", color="white")
                ax.tick_params(colors="white")
                charts.append(self._save(fig, f"num_{self._safe_name(col)}_by_target", f"{col} vs target (mean by class)", "bar"))

            if task_type == "regression":
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#1a2332")
                ax.set_facecolor("#243044")
                y_num = pd.to_numeric(y, errors="coerce")
                mask = series.index.intersection(y_num.dropna().index)
                ax.scatter(series.loc[mask], y_num.loc[mask], alpha=0.4, s=18, c="#f59e0b")
                ax.set_xlabel(col, color="white")
                ax.set_ylabel(self.target_column, color="white")
                ax.set_title(f"{col} vs {self.target_column}", color="white", fontsize=10)
                ax.tick_params(colors="white")
                charts.append(self._save(fig, f"num_{self._safe_name(col)}_scatter", f"{col} vs target", "scatter"))

        return charts

    def _charts_categorical(self, X: pd.DataFrame, y: pd.Series, task_type: str, eda: dict) -> list[dict[str, str]]:
        charts = []
        for idx, (col, info) in enumerate(eda.get("categorical_features", {}).items()):
            series = X[col].astype(str).fillna("missing")
            top = series.value_counts().head(8)
            if top.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a2332")
            ax.set_facecolor("#243044")
            ax.barh(top.index.astype(str)[::-1], top.values[::-1], color="#a78bfa")
            ax.set_title(f"{col} — top categories", color="white", fontsize=10)
            ax.tick_params(colors="white")
            charts.append(self._save(fig, f"cat_{self._safe_name(col)}_counts", f"{col} value counts", "bar"))

            if idx >= MAX_TARGET_RELATIONSHIP_CHARTS:
                continue

            if task_type == "classification" and info.get("target_rate_by_category"):
                rates = info["target_rate_by_category"]
                # Pick positive class rate if binary (last class or 'yes'/'1')
                plot_cats = list(top.index.astype(str))[:8]
                rate_vals = []
                for cat in plot_cats:
                    row = rates.get(cat, {})
                    if not row:
                        rate_vals.append(0)
                        continue
                    # prefer 'yes' or '1' or max class label
                    if "yes" in row:
                        rate_vals.append(row["yes"] * 100)
                    elif "1" in row:
                        rate_vals.append(row["1"] * 100)
                    else:
                        rate_vals.append(max(row.values()) * 100)
                fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a2332")
                ax.set_facecolor("#243044")
                ax.bar(plot_cats, rate_vals, color="#ef4444", edgecolor="white")
                ax.set_ylabel(f"Target rate (%)", color="white")
                ax.set_title(f"{col} — churn/target rate by category", color="white", fontsize=10)
                ax.tick_params(colors="white", axis="x", rotation=25)
                plt.setp(ax.get_xticklabels(), ha="right")
                charts.append(self._save(fig, f"cat_{self._safe_name(col)}_target_rate", f"{col} vs target rate", "bar"))

            if task_type == "regression" and info.get("mean_target_by_category"):
                items = list(info["mean_target_by_category"].items())[:8]
                fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a2332")
                ax.set_facecolor("#243044")
                ax.bar([k for k, _ in items], [v for _, v in items], color="#14b8a6")
                ax.set_ylabel(f"Mean {self.target_column}", color="white")
                ax.set_title(f"{col} — mean target by category", color="white", fontsize=10)
                ax.tick_params(colors="white", axis="x", rotation=25)
                charts.append(self._save(fig, f"cat_{self._safe_name(col)}_mean_target", f"{col} vs target", "bar"))

        return charts

    def _chart_correlations(self, correlations: list[dict]) -> dict[str, str]:
        top = correlations[:12]
        fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.35)), facecolor="#1a2332")
        ax.set_facecolor("#243044")
        names = [c["feature"] for c in top]
        vals = [c["correlation"] for c in top]
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="white", linewidth=0.8)
        ax.set_title("Numeric correlation with target", color="white", fontsize=10)
        ax.tick_params(colors="white")
        return self._save(fig, "correlations", "Feature correlations with target", "bar")
