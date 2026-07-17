"""Data leakage detection before training."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_ID_PATTERN = re.compile(r"(^id$|_id$|^id_|customer.?id|user.?id|account.?id|record.?id)", re.I)
_FUTURE_PATTERN = re.compile(r"(cancel|termination|end.?date|closed.?date|outcome|result|approved.?date|decision.?date)", re.I)


class LeakageDetector:
    def __init__(
        self,
        *,
        target_corr_threshold: float = 0.98,
        duplicate_corr_threshold: float = 0.999,
        id_unique_ratio: float = 0.95,
    ):
        self.target_corr_threshold = target_corr_threshold
        self.duplicate_corr_threshold = duplicate_corr_threshold
        self.id_unique_ratio = id_unique_ratio

    def detect(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target '{target_column}' not in dataset")

        issues: list[dict[str, Any]] = []
        recommended_drop: list[str] = []
        n_rows = len(df)
        features = [c for c in df.columns if c != target_column]
        y = df[target_column]

        # ID columns
        for col in features:
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / n_rows if n_rows else 0
            if _ID_PATTERN.search(col.replace(" ", "_")) or ratio >= self.id_unique_ratio:
                issues.append({
                    "column": col,
                    "type": "id_column",
                    "severity": "high",
                    "reason": f"Near-unique identifier ({nunique}/{n_rows} unique)",
                    "action": "remove",
                })
                recommended_drop.append(col)

        # Target leakage via correlation (numeric)
        numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) and pd.api.types.is_numeric_dtype(y):
            for col in numeric:
                if col in recommended_drop:
                    continue
                corr = df[col].corr(y)
                if corr is not None and abs(corr) >= self.target_corr_threshold:
                    issues.append({
                        "column": col,
                        "type": "target_leakage",
                        "severity": "critical",
                        "reason": f"Extremely high correlation with target (r={corr:.4f})",
                        "action": "remove",
                    })
                    recommended_drop.append(col)
        elif len(numeric):
            # Encode target temporarily for correlation
            y_enc = pd.factorize(y)[0]
            for col in numeric:
                if col in recommended_drop:
                    continue
                corr = np.corrcoef(df[col].fillna(df[col].median()), y_enc)[0, 1]
                if abs(corr) >= self.target_corr_threshold:
                    issues.append({
                        "column": col,
                        "type": "target_leakage",
                        "severity": "critical",
                        "reason": f"Extremely high correlation with target (r={corr:.4f})",
                        "action": "remove",
                    })
                    recommended_drop.append(col)

        # Duplicate / copy of target
        for col in features:
            if col in recommended_drop:
                continue
            if df[col].equals(y) or df[col].astype(str).equals(y.astype(str)):
                issues.append({
                    "column": col,
                    "type": "duplicate_target",
                    "severity": "critical",
                    "reason": "Column is identical to target",
                    "action": "remove",
                })
                recommended_drop.append(col)
                continue
            if col.lower().replace("_", "") == target_column.lower().replace("_", "") + "copy":
                issues.append({
                    "column": col,
                    "type": "duplicate_target",
                    "severity": "critical",
                    "reason": "Likely duplicate/copy of target column",
                    "action": "remove",
                })
                recommended_drop.append(col)

        # Near-duplicate features
        num_for_dup = [c for c in numeric if c not in recommended_drop]
        if len(num_for_dup) >= 2:
            corr_matrix = df[num_for_dup].corr().abs()
            seen_pairs: set[tuple[str, str]] = set()
            for i, a in enumerate(num_for_dup):
                for b in num_for_dup[i + 1:]:
                    if corr_matrix.loc[a, b] >= self.duplicate_corr_threshold:
                        pair = tuple(sorted((a, b)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        issues.append({
                            "column": b,
                            "type": "duplicate_feature",
                            "severity": "medium",
                            "reason": f"Near-perfect duplicate of '{a}' (r={corr_matrix.loc[a, b]:.4f})",
                            "action": "review",
                        })

        # Future information (heuristic on column names + datetime)
        datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        for col in features:
            if col in recommended_drop:
                continue
            if _FUTURE_PATTERN.search(col):
                issues.append({
                    "column": col,
                    "type": "future_information",
                    "severity": "high",
                    "reason": "Column name suggests post-outcome / future information",
                    "action": "remove",
                })
                recommended_drop.append(col)
            elif col in datetime_cols:
                issues.append({
                    "column": col,
                    "type": "future_information",
                    "severity": "medium",
                    "reason": "Datetime column may contain future information relative to prediction time",
                    "action": "review",
                })

        recommended_drop = list(dict.fromkeys(recommended_drop))
        leakage_detected = any(i["severity"] in ("critical", "high") for i in issues)

        return {
            "leakage_detected": leakage_detected,
            "n_issues": len(issues),
            "issues": issues,
            "recommended_drop": recommended_drop,
            "summary": self._summary(issues, recommended_drop),
        }

    @staticmethod
    def _summary(issues: list[dict], drops: list[str]) -> str:
        if not issues:
            return "No leakage signals detected."
        parts = [f"{len(issues)} potential leakage signal(s)"]
        if drops:
            parts.append(f"{len(drops)} column(s) recommended for removal")
        return " · ".join(parts)
