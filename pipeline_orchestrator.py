"""The AutoML orchestrator: wires all pipeline stages together end to end.

Stages: profile -> clean -> outlier-handle -> feature-engineer -> select
model -> tune (Optuna) -> fit & evaluate -> explain (SHAP) -> track
(MLflow) -> register -> save drift reference (for Evidently).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.data_cleaner import DataCleanerTransformer
from app.services.data_profiler import DataProfiler
from app.services.experiment_tracker import ExperimentTracker
from app.services.explainability import ExplainabilityService
from app.services.feature_importance import extract_from_pipeline
from app.services.feature_engineer import FeatureEngineerTransformer
from app.services.hyperparameter_tuner import HyperparameterTuner
from app.services.dataset_fingerprint import build_dataset_signature, row_size_bucket
from app.services.feature_selector import ColumnSelector, FeatureSelector
from app.services.leakage_detector import LeakageDetector
from app.services.business_insights import BusinessInsightGenerator
from app.services.ml_agent_graph import MLAdvisorAgent, build_fingerprint
from app.services.model_card_generator import ModelCardGenerator
from app.services.model_registry import get_model_registry
from app.services.model_reviewer import ModelReviewer
from app.services.model_selector import ModelSelector
from app.services.outlier_detector import OutlierCapTransformer
from app.store.job_store import get_job_store
from app.utils.io_utils import read_csv_safely

logger = get_logger(__name__)

PIPELINE_STAGES = [
    ("profiling", "Profiling dataset", 8),
    ("leakage", "Checking data leakage", 12),
    ("preprocessing", "Cleaning & feature engineering", 22),
    ("feature_selection", "Automatic feature selection", 30),
    ("model_selection", "Baseline model selection", 42),
    ("tuning", "Optuna hyperparameter tuning", 60),
    ("training", "Fitting final pipeline", 72),
    ("evaluation", "Evaluating on test set", 78),
    ("explainability", "SHAP explainability", 84),
    ("review", "AI model review & business insights", 90),
    ("advisor", "AI advisor report", 93),
    ("model_card", "Generating model card", 96),
    ("tracking", "MLflow tracking & registry", 98),
    ("complete", "Pipeline complete", 100),
]


class PipelineOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.profiler = DataProfiler()
        self.registry = get_model_registry()

    def _progress(self, job_id: str, stage: str, message: str, pct: int) -> None:
        if job_id:
            get_job_store().set_progress(job_id, stage, message, pct)

    def run(
        self,
        job_id: str,
        dataset_id: str,
        target_column: str,
        task_type: Optional[str] = None,
        n_trials: Optional[int] = None,
        test_size: Optional[float] = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        settings = self.settings
        cfg_model = settings.modeling
        cfg_clean = settings.cleaning
        cfg_outlier = settings.outliers
        cfg_fe = settings.feature_engineering
        cfg_tune = settings.tuning

        n_trials = n_trials or cfg_tune.get("n_trials", 25)
        test_size = test_size or cfg_model.get("test_size", 0.2)
        random_state = cfg_model.get("random_state", 42)
        cv_folds = cfg_model.get("cv_folds", 5)

        dataset_path = settings.upload_dir / f"{dataset_id}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found at {dataset_path}")

        df = read_csv_safely(dataset_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

        # --- upfront, row-count-changing cleanup (safe here, before splitting) ---
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.dropna(subset=[target_column]).reset_index(drop=True)

        logger.info("[%s] Profiling dataset (%d rows, %d cols)", job_id, *df.shape)
        self._progress(job_id, "profiling", "Profiling dataset", 8)
        profile = self.profiler.profile(df)

        # --- leakage detection (before split) ---
        self._progress(job_id, "leakage", "Scanning for data leakage", 12)
        leak_cfg = settings.leakage
        leakage_report = LeakageDetector(
            target_corr_threshold=leak_cfg.get("target_corr_threshold", 0.98),
            duplicate_corr_threshold=leak_cfg.get("duplicate_corr_threshold", 0.999),
        ).detect(df, target_column)
        if leak_cfg.get("auto_drop", True) and leakage_report.get("recommended_drop"):
            drop_cols = [c for c in leakage_report["recommended_drop"] if c in df.columns and c != target_column]
            if drop_cols:
                logger.info("[%s] Auto-dropping leakage columns: %s", job_id, drop_cols)
                df = df.drop(columns=drop_cols)
                profile = self.profiler.profile(df)

        X = df.drop(columns=[target_column])
        y_raw = df[target_column]

        selector = ModelSelector(random_state=random_state, cv_folds=cv_folds)
        detected_task_type = task_type or selector.detect_task_type(y_raw)

        label_encoder: Optional[LabelEncoder] = None
        if detected_task_type == "classification":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index, name=y_raw.name)
            from app.utils.split_utils import validate_classification_target

            validate_classification_target(y)
        else:
            y = y_raw.astype(float)

        from app.utils.split_utils import stratify_vector

        stratify = stratify_vector(y, test_size=test_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # --- fit preprocessing chain (fit on train only -> avoids leakage) ---
        datetime_cols = [c for c in profile["datetime_columns"] if c in X.columns]

        cleaner = DataCleanerTransformer(
            missing_drop_threshold=cfg_clean.get("missing_drop_threshold", 0.9),
            numeric_impute_strategy=cfg_clean.get("numeric_impute_strategy", "median"),
            categorical_impute_strategy=cfg_clean.get("categorical_impute_strategy", "most_frequent"),
        )
        outlier_tf = OutlierCapTransformer(
            method=cfg_outlier.get("method", "iqr"),
            iqr_multiplier=cfg_outlier.get("iqr_multiplier", 1.5),
            random_state=random_state,
        )
        feature_eng = FeatureEngineerTransformer(
            datetime_columns=datetime_cols,
            skew_threshold=cfg_fe.get("skew_threshold", 1.0),
            onehot_max_cardinality=cfg_fe.get("onehot_max_cardinality", 10),
            enable_interactions=cfg_fe.get("enable_interactions", True),
            max_interactions=cfg_fe.get("max_interactions", 3),
            enable_binning=cfg_fe.get("enable_binning", True),
            binning_max_unique=cfg_fe.get("binning_max_unique", 20),
        )

        logger.info("[%s] Fitting preprocessing chain", job_id)
        self._progress(job_id, "preprocessing", "Cleaning & engineering features", 22)
        X_train_c = cleaner.fit_transform(X_train, y_train)
        X_train_o = outlier_tf.fit_transform(X_train_c, y_train)
        X_train_f = feature_eng.fit_transform(X_train_o, y_train)

        # --- automatic feature selection (train only) ---
        fs_cfg = settings.feature_selection
        feature_selection_report: dict[str, Any] = {}
        if fs_cfg.get("enabled", True):
            self._progress(job_id, "feature_selection", "Selecting best features", 30)
            selector_fs = FeatureSelector(
                method=fs_cfg.get("method", "mutual_info"),
                max_features=fs_cfg.get("max_features"),
            )
            X_train_f, feature_selection_report = selector_fs.select(X_train_f, y_train, detected_task_type)
            selected = feature_selection_report.get("selected_features", list(X_train_f.columns))
            X_train_f = X_train_f[selected]

        candidates = (
            cfg_model.get("classification_candidates")
            if detected_task_type == "classification"
            else cfg_model.get("regression_candidates")
        )

        logger.info("[%s] Baseline model selection over candidates=%s", job_id, candidates)
        self._progress(job_id, "model_selection", f"Selecting best model from {len(candidates)} candidates", 45)
        selection_result = selector.select(X_train_f, y_train, detected_task_type, candidates)

        logger.info(
            "[%s] Tuning '%s' with Optuna (%d trials)", job_id, selection_result.best_model_name, n_trials
        )
        self._progress(job_id, "tuning", f"Tuning {selection_result.best_model_name} ({n_trials} trials)", 60)

        def tuning_progress(trial_num: int, total_trials: int) -> None:
            pct = 60 + int(12 * trial_num / max(total_trials, 1))
            self._progress(
                job_id,
                "tuning",
                f"Optuna trial {trial_num}/{total_trials} — {selection_result.best_model_name}",
                min(pct, 72),
            )

        tuner = HyperparameterTuner(random_state=random_state, cv_folds=cv_folds)
        tuning_result = tuner.tune(
            X_train_f,
            y_train,
            detected_task_type,
            selection_result.best_model_name,
            n_trials=n_trials,
            timeout=cfg_tune.get("timeout_seconds", 600),
            progress_callback=tuning_progress,
        )

        final_model = ModelSelector.build_model(
            detected_task_type, selection_result.best_model_name, tuning_result.best_params, random_state
        )

        pipeline_steps: list[tuple[str, Any]] = [
            ("cleaner", cleaner),
            ("outlier", outlier_tf),
            ("features", feature_eng),
        ]
        if feature_selection_report.get("selected_features"):
            pipeline_steps.append(
                ("selector", ColumnSelector(columns=feature_selection_report["selected_features"]))
            )
        pipeline_steps.append(("model", final_model))
        full_pipeline = Pipeline(steps=pipeline_steps)
        logger.info("[%s] Fitting final pipeline on full training split", job_id)
        self._progress(job_id, "training", "Fitting final pipeline", 75)
        full_pipeline.fit(X_train, y_train)

        self._progress(job_id, "evaluation", "Evaluating on holdout test set", 82)

        metrics = self._evaluate(full_pipeline, X_test, y_test, detected_task_type)
        logger.info("[%s] Test metrics: %s", job_id, metrics)

        # --- persist pipeline artifact ---
        job_dir = settings.artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "leakage_report.json").write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")
        if feature_selection_report:
            (job_dir / "feature_selection.json").write_text(
                json.dumps(feature_selection_report, indent=2), encoding="utf-8"
            )

        pipeline_path = job_dir / "pipeline.joblib"
        joblib.dump(full_pipeline, pipeline_path)

        # --- SHAP explainability (black-box over the full pipeline) ---
        self._progress(job_id, "explainability", "Computing SHAP feature importance", 88)
        shap_plot_path = None
        feature_importance: list[dict] = []
        try:
            explainer_service = ExplainabilityService(sample_size=settings.explainability.get("shap_sample_size", 200))
            if detected_task_type == "classification" and hasattr(full_pipeline, "predict_proba"):
                predict_fn = lambda data: full_pipeline.predict_proba(data)  # noqa: E731
            else:
                predict_fn = lambda data: full_pipeline.predict(data)  # noqa: E731
            explainer_service.build_explainer(predict_fn, X_train)
            shap_plot_path = job_dir / "shap_summary.png"
            explainer_service.summary_plot(X_test, shap_plot_path)
            feature_importance = extract_from_pipeline(full_pipeline, X_train, detected_task_type)
            fi_path = job_dir / "feature_importance.json"
            fi_path.write_text(json.dumps(feature_importance, indent=2), encoding="utf-8")
        except Exception as exc:  # SHAP is best-effort; never fail the whole run because of it
            logger.warning("[%s] SHAP explainability failed: %s", job_id, exc)
            shap_plot_path = None

        # --- AI model reviewer + business insights ---
        self._progress(job_id, "review", "Running model review & business insights", 90)
        model_review = ModelReviewer().review(
            job_id=job_id,
            model_name=selection_result.best_model_name,
            task_type=detected_task_type,
            metrics=metrics,
            baseline_scores=selection_result.baseline_scores,
            best_params=tuning_result.best_params,
            feature_importance=feature_importance[:10] if feature_importance else [],
            n_train=len(X_train),
            n_test=len(X_test),
            feature_selection=feature_selection_report or None,
            leakage_report=leakage_report,
        )
        (job_dir / "model_review.json").write_text(json.dumps(model_review, indent=2), encoding="utf-8")

        business_insights = None
        try:
            from app.services.eda_service import EDAService

            eda_for_biz = EDAService().analyze(df, target_column)
            business_insights = BusinessInsightGenerator().generate(
                target_column=target_column,
                task_type=detected_task_type,
                feature_importance=feature_importance[:10] if feature_importance else [],
                profile=profile,
                metrics=metrics,
                eda=eda_for_biz,
            )
            (job_dir / "business_insights.json").write_text(
                json.dumps(business_insights, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[%s] Business insights failed: %s", job_id, exc)

        # --- save reference dataset for future drift monitoring ---
        reference_path = settings.reference_dir / f"{job_id}.csv"
        X_train.to_csv(reference_path, index=False)

        # --- LangGraph ML advisor report (post-train evidence-grounded) ---
        self._progress(job_id, "advisor", "Generating evidence-grounded AI report", 93)
        advisor_report = None
        try:
            from app.services.eda_service import EDAService
            from app.services.rag_knowledge import get_knowledge_base

            eda = EDAService().analyze(df, target_column)
            profile["target_analysis"] = eda["target_analysis"]
            profile["correlation_with_target"] = eda["correlation_with_target"]
            profile["fingerprint"] = build_fingerprint(profile, target_column, None)

            job_result_preview = {
                "model_name": selection_result.best_model_name,
                "task_type": detected_task_type,
                "metrics": metrics,
                "baseline_scores": selection_result.baseline_scores,
                "best_params": tuning_result.best_params if tuning_result else {},
                "feature_importance": feature_importance[:10] if feature_importance else [],
            }
            advisor = MLAdvisorAgent()
            advisor_report = advisor.analyze(
                profile=profile,
                target_column=target_column,
                task_type=detected_task_type,
                job_result=job_result_preview,
                job_evidence={"feature_importance": feature_importance[:10]} if feature_importance else {},
            )
            advisor_path = job_dir / "advisor_report.md"
            advisor_path.write_text(advisor_report["narrative_report"], encoding="utf-8")
            get_knowledge_base().reload()
        except Exception as exc:
            logger.warning("[%s] ML advisor report failed: %s", job_id, exc)

        # --- model card ---
        self._progress(job_id, "model_card", "Generating model card documentation", 96)
        model_card_paths: dict[str, str] = {}
        try:
            card_entry = {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "target_column": target_column,
                "task_type": detected_task_type,
                "model_name": selection_result.best_model_name,
                "best_params": tuning_result.best_params,
                "metrics": metrics,
                "baseline_scores": selection_result.baseline_scores,
                "feature_columns": X.columns.tolist(),
                "top_features": feature_importance[:10] if feature_importance else [],
                "n_rows": int(profile.get("n_rows", len(df))),
                "elapsed_seconds": round(time.time() - t0, 2),
            }
            card_data = ModelCardGenerator().generate(
                job_id=job_id,
                registry_entry=card_entry,
                profile=profile,
                feature_selection=feature_selection_report or None,
                leakage_report=leakage_report,
                model_review=model_review,
                business_insights=business_insights,
                advisor_report=advisor_report.get("narrative_report") if advisor_report else None,
            )
            model_card_paths = ModelCardGenerator().save(job_dir, card_data)
        except Exception as exc:
            logger.warning("[%s] Model card generation failed: %s", job_id, exc)

        # --- MLflow tracking ---
        self._progress(job_id, "tracking", "Logging to MLflow & registry", 98)
        tracker = ExperimentTracker()
        mlflow_run_id = ""
        with tracker.start_run(run_name=job_id):
            tracker.set_tags({"job_id": job_id, "dataset_id": dataset_id, "task_type": detected_task_type})
            tracker.log_params(
                {
                    "target_column": target_column,
                    "task_type": detected_task_type,
                    "model_name": selection_result.best_model_name,
                    "n_trials": n_trials,
                    "test_size": test_size,
                    **{f"best_{k}": v for k, v in tuning_result.best_params.items()},
                }
            )
            tracker.log_metrics(metrics)
            tracker.log_dict_artifact(profile, "profiling_report.json")
            tracker.log_dict_artifact(cleaner.get_report(), "cleaning_report.json")
            tracker.log_dict_artifact(outlier_tf.get_report(), "outlier_report.json")
            tracker.log_dict_artifact(feature_eng.get_report(), "feature_engineering_report.json")
            tracker.log_dict_artifact(
                {"baseline_scores": selection_result.baseline_scores, "scoring": selection_result.scoring},
                "model_selection_report.json",
            )
            if advisor_report:
                tracker.log_dict_artifact(advisor_report, "advisor_report.json")
            tracker.log_dict_artifact(leakage_report, "leakage_report.json")
            if feature_selection_report:
                tracker.log_dict_artifact(feature_selection_report, "feature_selection.json")
            tracker.log_dict_artifact(model_review, "model_review.json")
            if business_insights:
                tracker.log_dict_artifact(business_insights, "business_insights.json")
            if model_card_paths.get("markdown"):
                tracker.log_artifact(Path(model_card_paths["markdown"]))
            tracker.log_dict_artifact(
                {"best_params": tuning_result.best_params, "best_score": tuning_result.best_score,
                 "trials": tuning_result.trials_history},
                "tuning_history.json",
            )
            if shap_plot_path and Path(shap_plot_path).exists():
                tracker.log_artifact(shap_plot_path)
            tracker.log_model(full_pipeline, artifact_path="model", registered_model_name=None)
            mlflow_run_id = tracker.active_run_id()

        elapsed = round(time.time() - t0, 2)

        registry_entry = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "task_type": detected_task_type,
            "model_name": selection_result.best_model_name,
            "best_params": tuning_result.best_params,
            "metrics": metrics,
            "baseline_scores": selection_result.baseline_scores,
            "pipeline_path": str(pipeline_path),
            "reference_data_path": str(reference_path),
            "shap_plot_path": str(shap_plot_path) if shap_plot_path else None,
            "feature_columns": X.columns.tolist(),
            "label_classes": label_encoder.classes_.tolist() if label_encoder is not None else None,
            "mlflow_run_id": mlflow_run_id,
            "training_seconds": elapsed,
            "n_rows": int(profile.get("n_rows", len(df))),
            "dataset_signature": build_dataset_signature(
                list(X.columns) + ([target_column] if target_column not in X.columns else []),
                target_column,
            ),
            "row_bucket": row_size_bucket(int(profile.get("n_rows", len(df)))),
            "top_features": feature_importance[:10] if feature_importance else [],
            "is_imbalanced": bool((profile.get("target_analysis") or {}).get("is_imbalanced")),
            "model_card_path": model_card_paths.get("markdown"),
            "leakage_detected": leakage_report.get("leakage_detected", False),
            "feature_selection": feature_selection_report,
        }
        self.registry.register(job_id, registry_entry)
        self._progress(job_id, "complete", "Pipeline complete", 100)

        return {
            "job_id": job_id,
            "task_type": detected_task_type,
            "model_name": selection_result.best_model_name,
            "baseline_scores": selection_result.baseline_scores,
            "model_leaderboard": sorted(
                selection_result.baseline_scores.items(), key=lambda x: x[1], reverse=True
            ),
            "best_params": tuning_result.best_params,
            "metrics": metrics,
            "mlflow_run_id": mlflow_run_id,
            "pipeline_path": str(pipeline_path),
            "shap_plot_path": str(shap_plot_path) if shap_plot_path else None,
            "reference_data_path": str(reference_path),
            "training_seconds": elapsed,
            "advisor_report": advisor_report,
            "feature_importance": feature_importance[:10] if feature_importance else [],
            "leakage_report": leakage_report,
            "feature_selection": feature_selection_report,
            "model_review": model_review,
            "business_insights": business_insights,
            "model_card_path": model_card_paths.get("markdown"),
        }

    @staticmethod
    def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> dict[str, float]:
        y_pred = pipeline.predict(X_test)
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
            try:
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                        )
            except Exception:
                pass
            return metrics

        return {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(root_mean_squared_error(y_test, y_pred)),
        }
