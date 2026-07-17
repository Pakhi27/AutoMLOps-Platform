"""Stage 6: Hyperparameter tuning with Optuna."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from app.services.model_selector import ModelSelector

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    trials_history: list[dict[str, Any]] = field(default_factory=list)


def _suggest_params(trial: optuna.Trial, model_name: str, task_type: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
        }
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
    if model_name == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        if task_type == "classification":
            params["eval_metric"] = "logloss"
        return params
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "verbosity": -1,
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }
    if model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        }
    if model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "verbose": 0,
            "allow_writing_files": False,
        }
    raise ValueError(f"Unknown model_name: {model_name}")


class HyperparameterTuner:
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        model_name: str,
        n_trials: int = 25,
        timeout: int | None = None,
        scoring: str | None = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TuningResult:
        scoring = scoring or ("f1_weighted" if task_type == "classification" else "r2")
        from app.utils.split_utils import cross_validator

        cv, n_splits = cross_validator(y, self.cv_folds, task_type, self.random_state)
        history: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, model_name, task_type)
            model = ModelSelector.build_model(task_type, model_name, params, self.random_state)
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                score = float(np.mean(scores))
            except Exception:
                score = float("-inf")
            history.append({"trial": trial.number, "params": params, "score": score})
            if progress_callback is not None:
                progress_callback(trial.number + 1, n_trials)
            return score

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            trials_history=history,
        )
