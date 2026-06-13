from __future__ import annotations

from typing import Any

import pandas as pd


MODEL_ID = "always_positive"


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Direction-only benchmark that always predicts a positive return."""

    work = panel.sort_values(["date", "country"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        rows.append(
            {
                "model_id": MODEL_ID,
                "country": row.country,
                "date": row.date,
                "month_str": row.month_str,
                "actual_return": float(row.target_return),
                "predicted_return": 1.0,
                "train_size": 0,
            }
        )

    meta = {
        "model_id": MODEL_ID,
        "method_name": "Always-positive directional baseline",
        "research_basis": [
            "Simple class-imbalance benchmark for directional return prediction.",
            "A useful edge check because equity index returns are positive in many monthly samples.",
        ],
        "feature_columns": [],
        "config": dict(config),
    }
    return pd.DataFrame(rows), meta
