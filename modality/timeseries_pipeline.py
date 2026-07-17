"""Time-series forecasting pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from app.core.config import get_settings
from app.services.modality.base import BaseModalityPipeline
from app.utils.io_utils import read_csv_safely


class TimeSeriesModalityPipeline(BaseModalityPipeline):
    modality = "timeseries"
    pipeline_type = "timeseries_forecast"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        datetime_column: Optional[str] = None,
        test_size: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        t0 = time.time()
        settings = get_settings()
        test_size = test_size if test_size is not None else settings.modeling.get("test_size", 0.2)
        df = read_csv_safely(Path(dataset_path))
        dt_col = datetime_column or metadata.get("datetime_column")
        if not dt_col:
            for c in df.columns:
                if "date" in str(c).lower() or "time" in str(c).lower():
                    dt_col = c
                    break
        if not dt_col or dt_col not in df.columns:
            raise ValueError("datetime_column required for time-series pipeline.")

        self.report("preprocessing", "Parsing timestamps and sorting", 12)
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col, target_column]).sort_values(dt_col).reset_index(drop=True)
        y = pd.to_numeric(df[target_column], errors="coerce")
        df = df.loc[y.notna()].copy()
        y = y.loc[y.notna()]

        self.report("preprocessing", "Engineering lag and rolling features", 28)
        features = self._build_features(df, dt_col, target_column)
        X = features.dropna()
        y_aligned = y.loc[X.index]
        if len(X) < 30:
            raise ValueError("Not enough rows for time-series training (need 30+).")

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_aligned.iloc[:split_idx], y_aligned.iloc[split_idx:]

        candidates = {
            "ridge_lags": Ridge(alpha=1.0),
            "lgbm_lags": LGBMRegressor(n_estimators=400, learning_rate=0.05, verbose=-1),
            "xgboost_lags": XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=-1,
            ),
            "random_forest_lags": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "hist_gbr_lags": HistGradientBoostingRegressor(max_iter=200, random_state=42),
        }

        self.report("training", "Training forecast models", 55)
        best_name, best_model, best_rmse = "", None, float("inf")
        leaderboard: dict[str, float] = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            leaderboard[name] = round(-rmse, 4)  # higher is better for display
            if rmse < best_rmse:
                best_rmse, best_name, best_model = rmse, name, model

        pred = best_model.predict(X_test)
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "r2": round(float(r2_score(y_test, pred)), 4),
        }
        if y_test.abs().mean() > 0:
            metrics["mape"] = round(float(np.mean(np.abs((y_test - pred) / y_test.replace(0, np.nan)))), 4)

        self.report("explainability", "Lag feature importance", 82)
        importance = self._feature_importance(best_model, list(X.columns))

        out_dir = settings.artifacts_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": best_model,
            "feature_columns": list(X.columns),
            "datetime_column": dt_col,
            "target_column": target_column,
        }
        model_path = out_dir / "timeseries_model.joblib"
        joblib.dump(bundle, model_path)
        (out_dir / "timeseries_explainability.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")

        self.report("complete", "Time-series pipeline complete", 100)
        return {
            "job_id": job_id,
            "modality": self.modality,
            "pipeline_type": self.pipeline_type,
            "task_type": "regression",
            "model_name": best_name,
            "datetime_column": dt_col,
            "target_column": target_column,
            "metrics": metrics,
            "baseline_scores": leaderboard,
            "explainability": importance,
            "drift_notes": ["Value distribution shift", "Seasonality change", "Anomaly rate drift"],
            "pipeline_path": str(model_path),
            "training_seconds": round(time.time() - t0, 1),
        }

    @staticmethod
    def _build_features(df: pd.DataFrame, dt_col: str, target_col: str) -> pd.DataFrame:
        s = pd.to_numeric(df[target_col], errors="coerce")
        out = pd.DataFrame(index=df.index)
        for lag in (1, 2, 3, 7, 14):
            out[f"lag_{lag}"] = s.shift(lag)
        out["roll_mean_7"] = s.shift(1).rolling(7, min_periods=1).mean()
        out["roll_std_7"] = s.shift(1).rolling(7, min_periods=1).std()
        dt = df[dt_col]
        out["hour"] = dt.dt.hour
        out["dow"] = dt.dt.dayofweek
        out["month"] = dt.dt.month
        return out

    @staticmethod
    def _feature_importance(model: Any, columns: list[str]) -> dict[str, Any]:
        if hasattr(model, "feature_importances_"):
            pairs = sorted(zip(columns, model.feature_importances_), key=lambda x: -x[1])
            return {"method": "feature_importance", "features": [{"name": n, "importance": round(float(v), 4)} for n, v in pairs[:15]]}
        if hasattr(model, "coef_"):
            pairs = sorted(zip(columns, np.abs(model.coef_)), key=lambda x: -x[1])
            return {"method": "coefficients", "features": [{"name": n, "importance": round(float(v), 4)} for n, v in pairs[:15]]}
        return {"method": "none", "features": []}
