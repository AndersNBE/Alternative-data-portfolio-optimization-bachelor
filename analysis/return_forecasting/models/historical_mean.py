from __future__ import annotations

from typing import Any

import pandas as pd


MODEL_ID = "historical_mean"


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    min_train_months = int(config.get("min_train_months", 36))
    rows: list[dict[str, Any]] = []
    panel = panel.sort_values(["date", "country"]).reset_index(drop=True)
    global_mean = float(panel["target_return"].dropna().mean()) if panel["target_return"].notna().any() else 0.0

    for _, row in panel.iterrows():
        history = panel[(panel["country"] == row["country"]) & (panel["date"] < row["date"])]
        history = history.dropna(subset=["target_return"])
        if len(history) >= min_train_months:
            pred = float(history["target_return"].mean())
            train_size = int(len(history))
        else:
            pooled = panel[panel["date"] < row["date"]].dropna(subset=["target_return"])
            pred = float(pooled["target_return"].mean()) if not pooled.empty else global_mean
            train_size = int(len(pooled))
        rows.append(
            {
                "model_id": MODEL_ID,
                "country": row["country"],
                "date": row["date"],
                "month_str": row["month_str"],
                "actual_return": float(row["target_return"]),
                "predicted_return": pred,
                "train_size": train_size,
            }
        )

    meta = {
        "model_id": MODEL_ID,
        "method_name": "Country historical mean with pooled fallback",
        "research_basis": [
            "Historical-average benchmark used in out-of-sample return-prediction studies.",
            "Welch and Goyal (2008) emphasize comparing predictive regressions against the historical mean.",
        ],
        "feature_columns": [],
        "config": dict(config),
    }
    return pd.DataFrame(rows), meta

