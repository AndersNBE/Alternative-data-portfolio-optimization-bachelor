"""Conservative expanding-window OLS forecasts for country index returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MODEL_ID = "ols_predictive"

DEFAULT_FEATURE_COLUMNS = [
    "gnc_lag_1",
    "gnc_lag_2",
    "gnc_lag_3",
]

REQUIRED_COLUMNS = {
    "country",
    "date",
    "month_str",
    "target_return",
}


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run country-level expanding-window predictive regressions.

    The model follows the standard return-predictability setup: each prediction
    for country c at date t is estimated only with rows for country c where
    date < t. Forecasts are a robust combination of valid univariate OLS
    forecasts over lagged predictors, then shrunk toward the historical mean.
    """

    cfg = _normalise_config(config)
    _validate_panel(panel)

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    feature_columns = _select_feature_columns(work, cfg)

    predictions: list[dict[str, Any]] = []
    for country, country_panel in work.sort_values(["country", "date"]).groupby(
        "country", sort=True
    ):
        country_panel = country_panel.sort_values("date").reset_index(drop=True)
        for _, row in country_panel.iterrows():
            train = country_panel.loc[country_panel["date"] < row["date"]]
            train_y = pd.to_numeric(train["target_return"], errors="coerce").dropna()
            historical_mean = float(train_y.mean()) if len(train_y) else cfg["default_forecast"]

            if len(train_y) < cfg["min_train_size"]:
                forecast = historical_mean
            else:
                forecast = _combined_univariate_forecast(
                    train=train,
                    row=row,
                    feature_columns=feature_columns,
                    min_feature_obs=cfg["min_feature_obs"],
                    fallback=historical_mean,
                    variance_epsilon=cfg["variance_epsilon"],
                    combination_method=cfg["combination_method"],
                )
                forecast = _shrink_to_mean(
                    forecast=forecast,
                    historical_mean=historical_mean,
                    train_size=len(train_y),
                    config=cfg,
                )

            forecast = _apply_forecast_restrictions(
                forecast=forecast,
                historical_mean=historical_mean,
                historical_std=float(train_y.std(ddof=1)) if len(train_y) > 1 else np.nan,
                config=cfg,
            )

            predictions.append(
                {
                    "model_id": MODEL_ID,
                    "country": country,
                    "date": row["date"],
                    "month_str": row["month_str"],
                    "actual_return": _as_float_or_nan(row["target_return"]),
                    "predicted_return": forecast,
                    "train_size": int(len(train_y)),
                }
            )

    prediction_df = pd.DataFrame(
        predictions,
        columns=[
            "model_id",
            "country",
            "date",
            "month_str",
            "actual_return",
            "predicted_return",
            "train_size",
        ],
    )

    meta = {
        "model_id": MODEL_ID,
        "method_name": "Conservative expanding-window univariate OLS combination",
        "research_basis": [
            {
                "citation": "Yu, Hao, Wu, Zhao and Wang (2023)",
                "doi": "10.1057/s41599-023-01891-9",
                "role": "Container-based predictors and simple forecast combination.",
            },
            {
                "citation": "Welch and Goyal (2008)",
                "doi": "10.1093/rfs/hhm014",
                "role": "Strict out-of-sample predictive-regression evaluation.",
            },
            {
                "citation": "Campbell and Thompson (2008)",
                "doi": "10.1093/rfs/hhm055",
                "role": "Motivation for conservative forecast restrictions.",
            },
        ],
        "feature_columns": feature_columns,
        "config": cfg,
    }
    return prediction_df, meta


def _normalise_config(config: dict | None) -> dict[str, Any]:
    raw = dict(config or {})
    cfg: dict[str, Any] = {
        "feature_columns": raw.get("feature_columns"),
        "min_train_size": int(raw.get("min_train_size", 36)),
        "min_feature_obs": int(raw.get("min_feature_obs", 30)),
        "default_forecast": float(raw.get("default_forecast", 0.0)),
        "variance_epsilon": float(raw.get("variance_epsilon", 1e-12)),
        "forecast_bound_sigma": raw.get("forecast_bound_sigma", 1.5),
        "nonnegative_forecast": bool(raw.get("nonnegative_forecast", False)),
        "shrinkage_weight": raw.get("shrinkage_weight", 0.35),
        "combination_method": raw.get("combination_method", "median"),
    }
    if cfg["forecast_bound_sigma"] is not None:
        cfg["forecast_bound_sigma"] = float(cfg["forecast_bound_sigma"])
    if cfg["shrinkage_weight"] is not None:
        cfg["shrinkage_weight"] = float(cfg["shrinkage_weight"])
        if not 0.0 <= cfg["shrinkage_weight"] <= 1.0:
            raise ValueError("shrinkage_weight must be between 0 and 1")
    if cfg["combination_method"] not in {"mean", "median"}:
        raise ValueError("combination_method must be 'mean' or 'median'")
    return cfg


def _validate_panel(panel: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")


def _select_feature_columns(panel: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    requested = config.get("feature_columns") or DEFAULT_FEATURE_COLUMNS
    available = [col for col in requested if col in panel.columns]
    non_lagged = [col for col in available if "_lag_" not in col]
    if non_lagged:
        raise ValueError(
            "ols_predictive only accepts lagged predictors; "
            f"non-lagged columns requested: {non_lagged}"
        )
    if not available:
        raise ValueError("no requested lagged feature columns are present in panel")
    return available


def _combined_univariate_forecast(
    train: pd.DataFrame,
    row: pd.Series,
    feature_columns: list[str],
    min_feature_obs: int,
    fallback: float,
    variance_epsilon: float,
    combination_method: str,
) -> float:
    forecasts: list[float] = []
    for feature in feature_columns:
        x_next = _as_float_or_nan(row[feature])
        if not np.isfinite(x_next):
            continue

        xy = train[[feature, "target_return"]].apply(pd.to_numeric, errors="coerce")
        xy = xy.replace([np.inf, -np.inf], np.nan).dropna()
        if len(xy) < min_feature_obs:
            continue
        x = xy[feature].to_numpy(dtype=float)
        y = xy["target_return"].to_numpy(dtype=float)
        if np.nanvar(x) <= variance_epsilon:
            continue

        design = np.column_stack([np.ones(len(x)), x])
        try:
            intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        prediction = float(intercept + slope * x_next)
        if np.isfinite(prediction):
            forecasts.append(prediction)

    if not forecasts:
        return float(fallback)
    if combination_method == "median":
        return float(np.median(forecasts))
    return float(np.mean(forecasts))


def _shrink_to_mean(
    forecast: float,
    historical_mean: float,
    train_size: int,
    config: dict[str, Any],
) -> float:
    if not np.isfinite(forecast):
        return float(historical_mean)

    fixed_weight = config["shrinkage_weight"]
    if fixed_weight is None:
        feature_obs = max(config["min_feature_obs"], 1)
        fixed_weight = max(0.0, min(1.0, 1.0 - feature_obs / max(train_size, 1)))

    return float(historical_mean + fixed_weight * (forecast - historical_mean))


def _apply_forecast_restrictions(
    forecast: float,
    historical_mean: float,
    historical_std: float,
    config: dict[str, Any],
) -> float:
    if not np.isfinite(forecast):
        forecast = historical_mean

    if config["nonnegative_forecast"]:
        forecast = max(0.0, forecast)

    bound_sigma = config["forecast_bound_sigma"]
    if bound_sigma is not None and np.isfinite(historical_std) and historical_std > 0:
        lower = historical_mean - bound_sigma * historical_std
        upper = historical_mean + bound_sigma * historical_std
        forecast = min(max(forecast, lower), upper)

    return float(forecast)


def _as_float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(result):
        return float("nan")
    return result
