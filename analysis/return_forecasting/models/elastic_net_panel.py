"""Regularized panel return forecasting model.

The public entry point is ``run_model(panel, config)``.  It fits a pooled
country-month model in an expanding-window out-of-sample loop and uses only
rows strictly before each prediction date.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_ID = "elastic_net_panel"

DEFAULT_FEATURE_COLUMNS = [
    "gnc_lag_1",
    "gnc_lag_2",
    "gnc_lag_3",
    "return_lag_1",
    "return_lag_2",
    "rolling_vol_3",
]


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run expanding-window OOS forecasts for country index returns.

    Parameters
    ----------
    panel:
        Monthly country panel. Required columns are described in the common
        interface. ``date`` must be parseable by ``pandas.to_datetime``.
    config:
        Optional JSON-like settings. Important keys:
        ``feature_columns`` (list[str]), ``penalty`` ("elastic_net", "ridge",
        "lasso"), ``use_cv`` (bool), ``alpha`` (float), ``l1_ratio`` (float),
        ``alpha_grid`` (list[float]), ``l1_ratio_grid`` (list[float]),
        ``min_train_size`` (int), and ``cv_splits`` (int).
    """

    cfg = _default_config()
    cfg.update(config or {})

    required = {"country", "date", "month_str", "target_return"}
    missing_required = required.difference(panel.columns)
    if missing_required:
        raise ValueError(f"panel is missing required columns: {sorted(missing_required)}")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date", "country"]).reset_index(drop=True)
    work["country"] = work["country"].astype("string").fillna("__missing_country__")
    work["target_return"] = pd.to_numeric(work["target_return"], errors="coerce")

    feature_columns = _resolve_feature_columns(work, cfg)
    for column in feature_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    predictions: list[dict[str, Any]] = []
    unique_dates = list(pd.Series(work["date"].dropna().unique()).sort_values())

    for prediction_date in unique_dates:
        train = work[(work["date"] < prediction_date) & work["target_return"].notna()].copy()
        test = work[(work["date"] == prediction_date) & work["target_return"].notna()].copy()
        if test.empty:
            continue

        train_size = int(len(train))
        train_months = int(train["date"].nunique())
        fitted_model = None
        if (
            train_size >= int(cfg["min_train_size"])
            and train_months >= int(cfg["min_train_months"])
            and feature_columns
        ):
            try:
                fitted_model = _fit_model(train, feature_columns, cfg)
            except Exception:
                fitted_model = None

        fallback_values = _historical_mean_predictions(train, test, cfg)

        if fitted_model is None:
            predicted = fallback_values
        else:
            try:
                raw_predicted = fitted_model.predict(test[feature_columns + ["country"]])
                raw_predicted = np.asarray(raw_predicted, dtype=float)
                invalid = ~np.isfinite(raw_predicted)
                if invalid.any():
                    raw_predicted[invalid] = fallback_values[invalid]
                predicted = _shrink_to_fallback(raw_predicted, fallback_values, train_months, cfg)
            except Exception:
                predicted = fallback_values

        for row, pred in zip(test.itertuples(index=False), predicted):
            predictions.append(
                {
                    "model_id": MODEL_ID,
                    "country": str(row.country),
                    "date": pd.Timestamp(row.date),
                    "month_str": str(row.month_str),
                    "actual_return": float(row.target_return),
                    "predicted_return": float(pred),
                    "train_size": train_size,
                }
            )

    output = pd.DataFrame(
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
        "method_name": "Expanding-window regularized panel regression with country fixed effects",
        "feature_columns": list(feature_columns),
        "config": _json_safe(cfg),
    }
    return output, meta


def _default_config() -> dict[str, Any]:
    return {
        "penalty": "elastic_net",
        "use_cv": True,
        "alpha": 1.0,
        "l1_ratio": 0.2,
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "l1_ratio_grid": [0.05, 0.2, 0.5],
        "cv_splits": 3,
        "min_train_size": 36,
        "min_train_months": 36,
        "min_country_mean_obs": 36,
        "forecast_signal_weight": 0.25,
        "signal_weight_min_train_months": 36,
        "signal_weight_full_train_months": 120,
        "max_signal_deviation": 0.08,
        "feature_columns": None,
        "include_current_gnc": False,
        "random_state": 20260423,
        "max_iter": 50000,
    }


def _resolve_feature_columns(panel: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    configured = config.get("feature_columns")
    if configured:
        candidates = [str(column) for column in configured]
    else:
        candidates = list(DEFAULT_FEATURE_COLUMNS)
        if bool(config.get("include_current_gnc", False)):
            candidates.append("gnc")

    return [column for column in candidates if column in panel.columns]


def _fit_model(train: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> Pipeline:
    train_sorted = train.sort_values(["date", "country"]).reset_index(drop=True)
    x_train = train_sorted[feature_columns + ["country"]]
    y_train = train_sorted["target_return"].astype(float).to_numpy()

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_columns,
            ),
            ("country_fe", _make_one_hot_encoder(), ["country"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    estimator = _make_estimator(train_sorted, config)
    model = Pipeline(steps=[("features", transformer), ("model", estimator)])
    model.fit(x_train, y_train)
    return model


def _make_estimator(train: pd.DataFrame, config: dict[str, Any]):
    penalty = str(config.get("penalty", "elastic_net")).lower()
    use_cv = bool(config.get("use_cv", True))
    alpha_grid = np.asarray(config.get("alpha_grid", [0.1]), dtype=float)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid) & (alpha_grid > 0)]
    if alpha_grid.size == 0:
        alpha_grid = np.asarray([float(config.get("alpha", 0.1))])

    cv = _make_time_series_cv(train, int(config.get("cv_splits", 3)))
    max_iter = int(config.get("max_iter", 50000))

    if use_cv and cv is not None:
        if penalty == "ridge":
            return RidgeCV(alphas=alpha_grid, cv=cv)
        if penalty == "lasso":
            return LassoCV(alphas=alpha_grid, cv=cv, max_iter=max_iter)
        return ElasticNetCV(
            alphas=alpha_grid,
            l1_ratio=list(config.get("l1_ratio_grid", [0.5])),
            cv=cv,
            max_iter=max_iter,
            random_state=int(config.get("random_state", 20260423)),
        )

    alpha = float(config.get("alpha", 0.1))
    if penalty == "ridge":
        return Ridge(alpha=alpha)
    if penalty == "lasso":
        return Lasso(alpha=alpha, max_iter=max_iter)
    return ElasticNet(
        alpha=alpha,
        l1_ratio=float(config.get("l1_ratio", 0.5)),
        max_iter=max_iter,
        random_state=int(config.get("random_state", 20260423)),
    )


def _make_time_series_cv(train: pd.DataFrame, requested_splits: int):
    unique_dates = np.asarray(sorted(train["date"].dropna().unique()))
    n_splits = min(max(requested_splits, 2), len(unique_dates) - 1)
    if n_splits < 2:
        return None

    splits = []
    for train_date_idx, val_date_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        train_dates = set(unique_dates[train_date_idx])
        val_dates = set(unique_dates[val_date_idx])
        train_idx = np.flatnonzero(train["date"].isin(train_dates).to_numpy())
        val_idx = np.flatnonzero(train["date"].isin(val_dates).to_numpy())
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits or None


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _historical_mean_predictions(
    train: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    if train.empty:
        global_mean = 0.0
        country_means = pd.Series(dtype=float)
        country_counts = pd.Series(dtype=int)
    else:
        global_mean = float(train["target_return"].mean())
        country_grouped = train.groupby("country")["target_return"]
        country_means = country_grouped.mean()
        country_counts = country_grouped.count()

    min_country_obs = int(config.get("min_country_mean_obs", 6))
    preds = []
    for country in test["country"]:
        if country in country_means.index and int(country_counts.loc[country]) >= min_country_obs:
            preds.append(float(country_means.loc[country]))
        else:
            preds.append(global_mean)
    return np.asarray(preds, dtype=float)


def _shrink_to_fallback(
    raw_predictions: np.ndarray,
    fallback_values: np.ndarray,
    train_months: int,
    config: dict[str, Any],
) -> np.ndarray:
    base_weight = float(config.get("forecast_signal_weight", 0.25))
    min_size = int(config.get("signal_weight_min_train_months", 36))
    full_size = int(config.get("signal_weight_full_train_months", 120))
    if full_size <= min_size:
        sample_weight = 1.0
    else:
        sample_weight = (train_months - min_size) / float(full_size - min_size)
        sample_weight = float(np.clip(sample_weight, 0.0, 1.0))

    weight = float(np.clip(base_weight * sample_weight, 0.0, 1.0))
    deviation = raw_predictions - fallback_values
    max_deviation = config.get("max_signal_deviation")
    if max_deviation is not None:
        bound = abs(float(max_deviation))
        deviation = np.clip(deviation, -bound, bound)
    return fallback_values + weight * deviation


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
