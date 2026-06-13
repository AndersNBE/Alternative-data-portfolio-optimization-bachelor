from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    values = pd.concat([a, b], axis=1).dropna()
    if len(values) < 3:
        return float("nan")
    if values.iloc[:, 0].std() == 0 or values.iloc[:, 1].std() == 0:
        return float("nan")
    return float(values.iloc[:, 0].corr(values.iloc[:, 1]))


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, group in predictions.groupby("model_id"):
        g = group.dropna(subset=["actual_return", "predicted_return"]).copy()
        if g.empty:
            continue
        err = g["predicted_return"] - g["actual_return"]
        direction_actual = np.sign(g["actual_return"])
        direction_pred = np.sign(g["predicted_return"])
        active = direction_pred != 0
        strategy_return = direction_pred * g["actual_return"]
        rows.append(
            {
                "model_id": model_id,
                "n_predictions": int(len(g)),
                "n_countries": int(g["country"].nunique()),
                "date_min": str(pd.to_datetime(g["date"]).min().date()),
                "date_max": str(pd.to_datetime(g["date"]).max().date()),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "corr": safe_corr(g["actual_return"], g["predicted_return"]),
                "directional_accuracy": float(np.mean(direction_actual == direction_pred)),
                "active_directional_accuracy": float(
                    np.mean(direction_actual[active] == direction_pred[active])
                )
                if bool(active.any())
                else float("nan"),
                "predicted_positive_rate": float(np.mean(g["predicted_return"] > 0)),
                "actual_positive_rate": float(np.mean(g["actual_return"] > 0)),
                "mean_actual_return": float(g["actual_return"].mean()),
                "mean_predicted_return": float(g["predicted_return"].mean()),
                "strategy_mean_monthly_return": float(strategy_return.mean()),
                "strategy_monthly_vol": float(strategy_return.std(ddof=0)),
                "strategy_annualized_sharpe": float(
                    np.sqrt(12) * strategy_return.mean() / strategy_return.std(ddof=0)
                )
                if strategy_return.std(ddof=0) > 0
                else float("nan"),
            }
        )
    columns = [
        "model_id",
        "n_predictions",
        "n_countries",
        "date_min",
        "date_max",
        "rmse",
        "mae",
        "bias",
        "corr",
        "directional_accuracy",
        "active_directional_accuracy",
        "predicted_positive_rate",
        "actual_positive_rate",
        "mean_actual_return",
        "mean_predicted_return",
        "strategy_mean_monthly_return",
        "strategy_monthly_vol",
        "strategy_annualized_sharpe",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["rmse", "mae"], ascending=[True, True])
        .reset_index(drop=True)
    )


def add_relative_metrics(metrics: pd.DataFrame, baseline_model_id: str = "historical_mean") -> pd.DataFrame:
    out = metrics.copy()
    baseline = out[out["model_id"] == baseline_model_id]
    if baseline.empty:
        out["rmse_improvement_vs_historical_mean"] = np.nan
        return out
    baseline_rmse = float(baseline.iloc[0]["rmse"])
    if baseline_rmse <= 0:
        out["rmse_improvement_vs_historical_mean"] = np.nan
        return out
    out["rmse_improvement_vs_historical_mean"] = 1.0 - (out["rmse"] / baseline_rmse)
    return out
