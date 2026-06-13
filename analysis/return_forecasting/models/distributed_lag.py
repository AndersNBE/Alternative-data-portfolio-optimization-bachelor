"""ARX/distributed-lag return forecaster using lagged container GNC signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MODEL_ID = "distributed_lag"


DEFAULT_CONFIG: dict[str, Any] = {
    "return_lags": 2,
    "gnc_lags": 6,
    "ridge_alpha": 250.0,
    "min_pooled_train": 36,
    "min_mean_train": 6,
    "country_mean_shrink_k": 24.0,
    "model_shrinkage": 0.25,
    "forecast_bound_sigma": 1.0,
    "geometric_decays": [0.70],
    "include_almon_terms": False,
    "include_country_fixed_effects": True,
}


def run_model(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run expanding-window out-of-sample forecasts.

    The model is deliberately conservative: own return lags handle AR structure,
    GNC lags enter through one constrained distributed-lag summary, and the
    regression forecast is shrunk back toward an expanding historical mean.
    """

    cfg = _merge_config(config)
    work = _prepare_panel(panel)
    work, feature_columns = _add_features(work, cfg)
    if not feature_columns:
        raise ValueError("No lagged predictor columns found for distributed_lag model.")

    predictions: list[dict[str, Any]] = []
    for prediction_date in sorted(work["date"].dropna().unique()):
        predictions.extend(
            _predict_date(
                work=work,
                prediction_date=pd.Timestamp(prediction_date),
                cfg=cfg,
                feature_columns=feature_columns,
            )
        )

    pred_df = pd.DataFrame(
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
        "method_name": "Conservative pooled ARX distributed-lag ridge",
        "feature_columns": feature_columns,
        "config": cfg,
    }
    return pred_df, meta


def _merge_config(config: dict | None) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    cfg["return_lags"] = int(cfg["return_lags"])
    cfg["gnc_lags"] = int(cfg["gnc_lags"])
    cfg["ridge_alpha"] = float(cfg["ridge_alpha"])
    cfg["min_pooled_train"] = int(cfg["min_pooled_train"])
    cfg["min_mean_train"] = int(cfg["min_mean_train"])
    cfg["country_mean_shrink_k"] = float(cfg["country_mean_shrink_k"])
    cfg["model_shrinkage"] = float(cfg["model_shrinkage"])
    cfg["forecast_bound_sigma"] = (
        None if cfg["forecast_bound_sigma"] is None else float(cfg["forecast_bound_sigma"])
    )
    cfg["geometric_decays"] = [float(x) for x in cfg["geometric_decays"]]
    cfg["include_almon_terms"] = bool(cfg["include_almon_terms"])
    cfg["include_country_fixed_effects"] = bool(cfg["include_country_fixed_effects"])
    return cfg


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"country", "date", "month_str", "target_return"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["country", "date"]).reset_index(drop=True)
    return work


def _add_features(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    work = panel.copy()
    feature_columns: list[str] = []

    for lag in range(1, cfg["return_lags"] + 1):
        col = f"return_lag_{lag}"
        if col in work.columns:
            feature_columns.append(col)

    gnc_cols = [f"gnc_lag_{lag}" for lag in range(1, cfg["gnc_lags"] + 1) if f"gnc_lag_{lag}" in work.columns]

    if gnc_cols:
        lags = np.arange(1, len(gnc_cols) + 1, dtype=float)
        for decay in cfg["geometric_decays"]:
            name = f"gnc_geom_{decay:g}"
            weights = decay ** (lags - 1.0)
            weights = weights / weights.sum()
            work[name] = work[gnc_cols].to_numpy(dtype=float) @ weights
            feature_columns.append(name)

        if cfg["include_almon_terms"]:
            lag_mean = lags.mean()
            lag_scale = np.max(np.abs(lags - lag_mean)) or 1.0
            lag_centered = (lags - lag_mean) / lag_scale
            almon_terms = {
                "gnc_almon_level": np.ones_like(lags) / len(lags),
                "gnc_almon_slope": lag_centered,
                "gnc_almon_curve": lag_centered**2 - np.mean(lag_centered**2),
            }
            gnc_values = work[gnc_cols].to_numpy(dtype=float)
            for name, weights in almon_terms.items():
                work[name] = gnc_values @ weights
                feature_columns.append(name)

    return work, _dedupe(feature_columns)


def _predict_date(
    work: pd.DataFrame,
    prediction_date: pd.Timestamp,
    cfg: dict[str, Any],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    train_pool = work[work["date"] < prediction_date].copy()
    test = work[work["date"] == prediction_date].sort_values("country")
    pooled_train = _complete_cases(train_pool, feature_columns)

    ridge_model = None
    if len(pooled_train) >= cfg["min_pooled_train"]:
        ridge_model = _fit_ridge_model(
            train=pooled_train,
            feature_columns=feature_columns,
            alpha=cfg["ridge_alpha"],
            country_fixed_effects=cfg["include_country_fixed_effects"],
        )

    predictions: list[dict[str, Any]] = []
    for _, row in test.iterrows():
        country = str(row["country"])
        anchor, anchor_std, anchor_size = _historical_anchor(train_pool, country, cfg)
        row_df = row.to_frame().T
        row_complete = _feature_complete_cases(row_df, feature_columns)
        if ridge_model is not None and len(row_complete) == 1:
            model_pred = _predict_ridge_model(
                model=ridge_model,
                row=row_complete.iloc[0],
                feature_columns=feature_columns,
            )
            predicted = _combine_and_restrict(
                model_pred=model_pred,
                anchor=anchor,
                historical_std=anchor_std,
                cfg=cfg,
            )
            train_size = len(pooled_train)
        else:
            predicted = anchor
            train_size = anchor_size

        predictions.append(
            {
                "model_id": MODEL_ID,
                "country": country,
                "date": row["date"],
                "month_str": row["month_str"],
                "actual_return": _to_float(row["target_return"]),
                "predicted_return": _to_float(predicted),
                "train_size": int(train_size),
            }
        )
    return predictions


def _fit_ridge_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    alpha: float,
    country_fixed_effects: bool,
) -> dict[str, Any]:
    X = train[feature_columns].to_numpy(dtype=float)

    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    X = (X - mean) / scale

    fe_countries: list[str] = []
    if country_fixed_effects:
        countries = sorted(train["country"].astype(str).unique())
        baseline = countries[0] if countries else None
        fe_countries = [c for c in countries if c != baseline]
        if fe_countries:
            fe_train = np.column_stack([(train["country"].astype(str) == c).to_numpy(float) for c in fe_countries])
            X = np.column_stack([X, fe_train])

    X_design = np.column_stack([np.ones(len(X)), X])
    y = train["target_return"].to_numpy(dtype=float)

    penalty = np.eye(X_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X_design.T @ X_design + penalty) @ X_design.T @ y
    return {
        "beta": beta,
        "mean": mean,
        "scale": scale,
        "fe_countries": fe_countries,
    }


def _predict_ridge_model(model: dict[str, Any], row: pd.Series, feature_columns: list[str]) -> float:
    x_pred = row[feature_columns].to_numpy(dtype=float)
    x_pred = (x_pred - model["mean"]) / model["scale"]

    fe_countries = model["fe_countries"]
    if fe_countries:
        fe_pred = np.array([float(str(row["country"]) == c) for c in fe_countries])
        x_pred = np.concatenate([x_pred, fe_pred])

    x_design = np.concatenate([[1.0], x_pred])
    return float(x_design @ model["beta"])


def _complete_cases(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    cols = ["target_return", "country", *feature_columns]
    available = [col for col in cols if col in df.columns]
    out = df[available].copy()
    numeric_cols = ["target_return", *feature_columns]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["target_return", *feature_columns])


def _feature_complete_cases(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    cols = ["country", *feature_columns]
    out = df[cols].copy()
    for col in feature_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=feature_columns)


def _historical_anchor(
    train_pool: pd.DataFrame,
    country: str,
    cfg: dict[str, Any],
) -> tuple[float, float, int]:
    pooled_y = pd.to_numeric(train_pool["target_return"], errors="coerce").dropna()
    pooled_mean = float(pooled_y.mean()) if len(pooled_y) else 0.0
    pooled_std = float(pooled_y.std(ddof=1)) if len(pooled_y) > 1 else float("nan")

    country_y = pd.to_numeric(
        train_pool.loc[train_pool["country"] == country, "target_return"],
        errors="coerce",
    ).dropna()
    if len(country_y) >= cfg["min_mean_train"]:
        n_country = float(len(country_y))
        weight = n_country / (n_country + cfg["country_mean_shrink_k"])
        country_mean = float(country_y.mean())
        country_std = float(country_y.std(ddof=1)) if len(country_y) > 1 else pooled_std
        anchor = weight * country_mean + (1.0 - weight) * pooled_mean
        std = country_std if np.isfinite(country_std) and country_std > 0 else pooled_std
        return float(anchor), float(std), int(len(country_y))

    return pooled_mean, pooled_std, int(len(pooled_y))


def _combine_and_restrict(
    model_pred: float,
    anchor: float,
    historical_std: float,
    cfg: dict[str, Any],
) -> float:
    if not np.isfinite(model_pred):
        model_pred = anchor

    shrinkage = min(max(cfg["model_shrinkage"], 0.0), 1.0)
    forecast = anchor + shrinkage * (model_pred - anchor)

    bound_sigma = cfg["forecast_bound_sigma"]
    if bound_sigma is not None and np.isfinite(historical_std) and historical_std > 0:
        lower = anchor - bound_sigma * historical_std
        upper = anchor + bound_sigma * historical_std
        forecast = min(max(forecast, lower), upper)
    return float(forecast)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
