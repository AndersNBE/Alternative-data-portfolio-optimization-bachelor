"""Expanding-window random forest panel model for monthly country returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_ID = "random_forest_panel"


BASE_FEATURE_COLUMNS = [
    "gnc_lag_1",
    "gnc_lag_2",
    "gnc_lag_3",
    "gnc_lag_4",
    "gnc_lag_5",
    "gnc_lag_6",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "return_lag_6",
    "gnc_ma_3",
    "gnc_ma_6",
    "gnc_delta_1",
    "gnc_delta_3",
    "rolling_vol_3",
    "rolling_vol_6",
    "n_ports",
    "mean_NC",
    "mean_observations_per_port",
]

OPTIONAL_FEATURE_COLUMNS = [
    "gnc",
]


DEFAULT_CONFIG = {
    "min_train_size": 48,
    "n_estimators": 300,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
    "winsorize_features": True,
    "winsorize_target": True,
    "winsor_lower": 0.02,
    "winsor_upper": 0.98,
    "prediction_clip_lower": 0.02,
    "prediction_clip_upper": 0.98,
    "shrinkage_to_mean": 0.60,
    "include_current_gnc": False,
    "feature_columns": None,
    "fallback": "country_then_global_mean",
}


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run a pooled panel random forest with strict expanding-window forecasts.

    The model is refit once per forecast month using only observations with
    ``date`` strictly before the prediction date. If the training window is too
    small, or if a row has incomplete predictors, the prediction falls back to a
    historical mean computed from the same past-only training window.
    """

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    _validate_config(cfg)

    configured_features = cfg.get("feature_columns")
    if configured_features:
        base_feature_columns = [str(col) for col in configured_features if str(col) in panel.columns]
    else:
        base_feature_columns = list(BASE_FEATURE_COLUMNS)
        if bool(cfg["include_current_gnc"]):
            base_feature_columns += OPTIONAL_FEATURE_COLUMNS

    if not base_feature_columns:
        raise ValueError("No random forest feature columns are available.")

    required_columns = {
        "country",
        "date",
        "month_str",
        "target_return",
        *base_feature_columns,
    }
    missing = sorted(required_columns.difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date", "country"]).reset_index(drop=True)
    work = _add_known_panel_features(work)

    feature_columns = list(base_feature_columns)
    feature_columns += [
        "country_code",
        "month_number",
        "month_sin",
        "month_cos",
        "year",
        "time_index",
    ]

    predictions: list[dict[str, Any]] = []
    for prediction_date in sorted(work["date"].dropna().unique()):
        train_window = work[work["date"] < prediction_date].copy()
        prediction_window = work[work["date"] == prediction_date].copy()

        valid_train = _complete_rows(train_window, feature_columns, include_target=True)
        train_size = int(len(valid_train))

        model: RandomForestRegressor | None = None
        feature_bounds: tuple[pd.Series, pd.Series] | None = None
        target_bounds: tuple[float, float] | None = None
        prediction_bounds: tuple[float, float] | None = None

        if train_size >= int(cfg["min_train_size"]):
            x_train = valid_train[feature_columns].astype(float)
            y_train = valid_train["target_return"].astype(float)
            raw_y_train = y_train.copy()

            if bool(cfg["winsorize_features"]):
                x_train, feature_bounds = _winsorize_frame(
                    x_train,
                    lower=float(cfg["winsor_lower"]),
                    upper=float(cfg["winsor_upper"]),
                )

            if bool(cfg["winsorize_target"]):
                y_train, target_bounds = _winsorize_series(
                    y_train,
                    lower=float(cfg["winsor_lower"]),
                    upper=float(cfg["winsor_upper"]),
                )

            prediction_bounds = _series_quantile_bounds(
                raw_y_train,
                lower=float(cfg["prediction_clip_lower"]),
                upper=float(cfg["prediction_clip_upper"]),
            )

            model = RandomForestRegressor(
                n_estimators=int(cfg["n_estimators"]),
                max_depth=cfg["max_depth"],
                min_samples_leaf=int(cfg["min_samples_leaf"]),
                max_features=cfg["max_features"],
                bootstrap=bool(cfg["bootstrap"]),
                random_state=int(cfg["random_state"]),
                n_jobs=int(cfg["n_jobs"]),
            )
            model.fit(x_train, y_train)

        for _, row in prediction_window.iterrows():
            fallback_prediction = _historical_mean_fallback(row, train_window)
            prediction = fallback_prediction
            if model is not None and not row[feature_columns].isna().any():
                x_pred = row[feature_columns].to_frame().T.astype(float)
                if feature_bounds is not None:
                    x_pred = x_pred.clip(lower=feature_bounds[0], upper=feature_bounds[1], axis=1)
                model_prediction = float(model.predict(x_pred)[0])
                if target_bounds is not None:
                    model_prediction = float(np.clip(model_prediction, target_bounds[0], target_bounds[1]))
                prediction = _shrink_prediction(
                    model_prediction=model_prediction,
                    fallback_prediction=fallback_prediction,
                    shrinkage=float(cfg["shrinkage_to_mean"]),
                )
                if prediction_bounds is not None:
                    prediction = float(
                        np.clip(prediction, prediction_bounds[0], prediction_bounds[1])
                    )

            predictions.append(
                {
                    "model_id": MODEL_ID,
                    "country": row["country"],
                    "date": row["date"],
                    "month_str": row["month_str"],
                    "actual_return": row["target_return"],
                    "predicted_return": prediction,
                    "train_size": train_size,
                }
            )

    predictions_df = pd.DataFrame(
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
        "method_name": "Expanding-window pooled RandomForestRegressor panel",
        "feature_columns": feature_columns,
        "config": _json_safe_config(cfg),
    }
    return predictions_df, meta


def _add_known_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    countries = sorted(out["country"].dropna().astype(str).unique())
    country_codes = {country: idx for idx, country in enumerate(countries)}
    out["country_code"] = out["country"].astype(str).map(country_codes).astype(float)

    month_number = out["date"].dt.month.astype(float)
    out["month_number"] = month_number
    out["month_sin"] = np.sin(2.0 * np.pi * month_number / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month_number / 12.0)
    out["year"] = out["date"].dt.year.astype(float)

    first_month = out["date"].min().to_period("M")
    out["time_index"] = out["date"].dt.to_period("M").apply(
        lambda value: (value.year - first_month.year) * 12 + value.month - first_month.month
    )
    out["time_index"] = out["time_index"].astype(float)
    return out


def _complete_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    include_target: bool,
) -> pd.DataFrame:
    columns = feature_columns + (["target_return"] if include_target else [])
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return clean.dropna(subset=columns)


def _winsorize_frame(
    frame: pd.DataFrame,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, tuple[pd.Series, pd.Series]]:
    lower_bounds = frame.quantile(lower)
    upper_bounds = frame.quantile(upper)
    return frame.clip(lower=lower_bounds, upper=upper_bounds, axis=1), (lower_bounds, upper_bounds)


def _winsorize_series(
    series: pd.Series,
    lower: float,
    upper: float,
) -> tuple[pd.Series, tuple[float, float]]:
    lower_bound = float(series.quantile(lower))
    upper_bound = float(series.quantile(upper))
    return series.clip(lower=lower_bound, upper=upper_bound), (lower_bound, upper_bound)


def _series_quantile_bounds(series: pd.Series, lower: float, upper: float) -> tuple[float, float]:
    return float(series.quantile(lower)), float(series.quantile(upper))


def _shrink_prediction(
    model_prediction: float,
    fallback_prediction: float,
    shrinkage: float,
) -> float:
    return (1.0 - shrinkage) * model_prediction + shrinkage * fallback_prediction


def _historical_mean_fallback(row: pd.Series, train_window: pd.DataFrame) -> float:
    valid_history = train_window.dropna(subset=["target_return"])
    country_history = valid_history[valid_history["country"] == row["country"]]
    if not country_history.empty:
        return float(country_history["target_return"].mean())
    if not valid_history.empty:
        return float(valid_history["target_return"].mean())
    return 0.0


def _validate_config(config: dict) -> None:
    if int(config["min_train_size"]) < 1:
        raise ValueError("min_train_size must be at least 1")
    if int(config["n_estimators"]) < 1:
        raise ValueError("n_estimators must be at least 1")
    if int(config["min_samples_leaf"]) < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    lower = float(config["winsor_lower"])
    upper = float(config["winsor_upper"])
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("winsor_lower and winsor_upper must satisfy 0 <= lower < upper <= 1")
    clip_lower = float(config["prediction_clip_lower"])
    clip_upper = float(config["prediction_clip_upper"])
    if not 0.0 <= clip_lower < clip_upper <= 1.0:
        raise ValueError(
            "prediction_clip_lower and prediction_clip_upper must satisfy 0 <= lower < upper <= 1"
        )
    shrinkage = float(config["shrinkage_to_mean"])
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage_to_mean must be between 0 and 1")


def _json_safe_config(config: dict) -> dict:
    safe: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
