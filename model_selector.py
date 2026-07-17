"""Stage 5: Task detection & baseline model selection.

Detects classification vs. regression from the target column, then quickly
cross-validates candidate model families with default hyperparameters to pick
the most promising one for Optuna tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    HAS_CATBOOST = False

# Models that benefit from feature scaling before fit.
_SCALED_MODELS = {"svm", "knn"}


def _catboost_classifier(rs: int) -> Any:
    return CatBoostClassifier(
        random_state=rs, iterations=200, verbose=0, allow_writing_files=False
    )


def _catboost_regressor(rs: int) -> Any:
    return CatBoostRegressor(
        random_state=rs, iterations=200, verbose=0, allow_writing_files=False
    )


CLASSIFICATION_MODELS: dict[str, Callable[[int], Any]] = {
    "logistic_regression": lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
    "xgboost": lambda rs: XGBClassifier(random_state=rs, eval_metric="logloss", n_estimators=200),
    "lightgbm": lambda rs: LGBMClassifier(random_state=rs, n_estimators=200, verbosity=-1),
    "gradient_boosting": lambda rs: GradientBoostingClassifier(random_state=rs, n_estimators=150),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingClassifier(
        random_state=rs, max_iter=200
    ),
    "svm": lambda rs: SVC(probability=True, random_state=rs),
    "knn": lambda rs: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

REGRESSION_MODELS: dict[str, Callable[[int], Any]] = {
    "ridge": lambda rs: Ridge(random_state=rs),
    "elastic_net": lambda rs: ElasticNet(random_state=rs, max_iter=5000),
    "random_forest": lambda rs: RandomForestRegressor(n_estimators=200, random_state=rs, n_jobs=-1),
    "xgboost": lambda rs: XGBRegressor(random_state=rs, n_estimators=200),
    "lightgbm": lambda rs: LGBMRegressor(random_state=rs, n_estimators=200, verbosity=-1),
    "gradient_boosting": lambda rs: GradientBoostingRegressor(random_state=rs, n_estimators=150),
    "hist_gradient_boosting": lambda rs: HistGradientBoostingRegressor(
        random_state=rs, max_iter=200
    ),
    "svm": lambda rs: SVR(),
    "knn": lambda rs: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}

if HAS_CATBOOST:
    CLASSIFICATION_MODELS["catboost"] = _catboost_classifier
    REGRESSION_MODELS["catboost"] = _catboost_regressor


@dataclass
class ModelSelectionResult:
    task_type: str
    best_model_name: str
    scoring: str
    baseline_scores: dict[str, float]


class ModelSelector:
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds

    def detect_task_type(self, y: pd.Series) -> str:
        if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == bool:
            return "classification"
        n_unique = y.nunique()
        if pd.api.types.is_integer_dtype(y) and n_unique <= max(20, int(0.05 * len(y))):
            return "classification"
        if n_unique <= 2:
            return "classification"
        return "regression"

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        candidates: list[str] | None = None,
    ) -> ModelSelectionResult:
        model_zoo = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        candidates = candidates or list(model_zoo.keys())
        scoring = "f1_weighted" if task_type == "classification" else "r2"

        from app.utils.split_utils import cross_validator

        cv, n_splits = cross_validator(y, self.cv_folds, task_type, self.random_state)

        scores: dict[str, float] = {}
        for name in candidates:
            if name not in model_zoo:
                continue
            try:
                model = self.build_model(task_type, name, {}, self.random_state)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                scores[name] = float(np.mean(cv_scores))
            except Exception:
                scores[name] = float("-inf")

        best_model_name = max(scores, key=scores.get) if scores else candidates[0]

        return ModelSelectionResult(
            task_type=task_type,
            best_model_name=best_model_name,
            scoring=scoring,
            baseline_scores=scores,
        )

    @staticmethod
    def build_model(task_type: str, model_name: str, params: dict[str, Any], random_state: int) -> Any:
        model_zoo = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        factory = model_zoo[model_name]
        model = factory(random_state)
        if params:
            model.set_params(**params)
        if model_name in _SCALED_MODELS:
            return SkPipeline([("scaler", StandardScaler()), ("model", model)])
        return model

    @staticmethod
    def available_models(task_type: str) -> list[str]:
        zoo = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        return list(zoo.keys())
