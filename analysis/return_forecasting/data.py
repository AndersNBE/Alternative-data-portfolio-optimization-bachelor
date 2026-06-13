from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "gnc",
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


def file_fingerprint(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": h.hexdigest(),
    }


def load_country_gnc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"country", "date", "GNC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    out["month_str"] = out["date"].dt.to_period("M").astype(str)
    out["gnc_same_month"] = pd.to_numeric(out["GNC"], errors="coerce")
    for col in ["n_ports", "mean_NC", "mean_observations_per_port"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[
        [
            "country",
            "date",
            "month_str",
            "gnc_same_month",
            "n_ports",
            "mean_NC",
            "mean_observations_per_port",
        ]
    ].dropna(subset=["country", "date", "gnc_same_month"])


def load_market_monthly_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"country", "index_name", "symbol", "date", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out = out.dropna(subset=["country", "date", "adj_close"])
    out = out[out["adj_close"] > 0].copy()
    out["month"] = out["date"].dt.to_period("M")

    out = out.sort_values(["country", "symbol", "date"])
    monthly_close = (
        out.groupby(["country", "index_name", "symbol", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly_close["date"] = monthly_close["month"].dt.to_timestamp()
    monthly_close = monthly_close.sort_values(["country", "symbol", "date"])
    monthly_close["adj_close_prev"] = monthly_close.groupby(["country", "symbol"])["adj_close"].shift(1)
    monthly_close["market_return"] = np.log(monthly_close["adj_close"] / monthly_close["adj_close_prev"])
    monthly_close["month_str"] = monthly_close["date"].dt.to_period("M").astype(str)
    return monthly_close[
        [
            "country",
            "index_name",
            "symbol",
            "date",
            "month_str",
            "adj_close",
            "market_return",
        ]
    ].dropna(subset=["market_return"])


def add_usd_market_returns(market_returns: pd.DataFrame, fx_rates_path: Path) -> pd.DataFrame:
    fx = pd.read_csv(fx_rates_path)
    required = {"country", "date", "currency", "usd_per_local_currency"}
    missing = required - set(fx.columns)
    if missing:
        raise ValueError(f"{fx_rates_path} is missing required columns: {sorted(missing)}")

    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx["usd_per_local_currency"] = pd.to_numeric(fx["usd_per_local_currency"], errors="coerce")
    fx = fx.dropna(subset=["country", "date", "currency", "usd_per_local_currency"])
    fx = fx[fx["usd_per_local_currency"] > 0].copy()
    fx["month"] = fx["date"].dt.to_period("M")
    monthly_fx = (
        fx.sort_values(["country", "date"])
        .groupby(["country", "currency", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly_fx["fx_return_usd"] = monthly_fx.groupby("country")["usd_per_local_currency"].transform(
        lambda values: np.log(values / values.shift(1))
    )
    monthly_fx["date"] = monthly_fx["month"].dt.to_timestamp()

    out = market_returns.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    out = out.merge(
        monthly_fx[["country", "date", "currency", "usd_per_local_currency", "fx_return_usd"]],
        on=["country", "date"],
        how="left",
        validate="one_to_one",
    )
    out["market_return_local"] = out["market_return"]
    out["market_return_usd"] = out["market_return_local"] + out["fx_return_usd"]
    return out


def build_forecast_panel(
    *,
    country_gnc: pd.DataFrame,
    market_returns: pd.DataFrame,
    forecast_horizon_months: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if forecast_horizon_months < 1:
        raise ValueError("forecast_horizon_months must be >= 1")

    market = market_returns.sort_values(["country", "date"]).copy()
    market["target_return"] = market.groupby("country")["market_return"].shift(
        -forecast_horizon_months + 1
    )

    # Rows are dated by the target return month. All features are shifted at
    # least one month back, so a row for month t only uses information known by
    # the end of month t-1.
    panel = market[
        ["country", "index_name", "symbol", "date", "month_str", "target_return"]
    ].merge(
        country_gnc,
        on=["country", "date", "month_str"],
        how="left",
    )
    panel = panel.sort_values(["country", "date"]).reset_index(drop=True)

    group = panel.groupby("country", group_keys=False)
    panel["gnc"] = group["gnc_same_month"].shift(1)
    for lag in range(1, 7):
        panel[f"gnc_lag_{lag}"] = group["gnc_same_month"].shift(lag)
        panel[f"return_lag_{lag}"] = group["target_return"].shift(lag)

    panel["gnc_ma_3"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(3).mean())
    panel["gnc_ma_6"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(6).mean())
    panel["gnc_delta_1"] = panel["gnc_lag_1"] - panel["gnc_lag_2"]
    panel["gnc_delta_3"] = panel["gnc_lag_1"] - panel["gnc_lag_4"]
    panel["rolling_vol_3"] = group["target_return"].transform(lambda s: s.shift(1).rolling(3).std())
    panel["rolling_vol_6"] = group["target_return"].transform(lambda s: s.shift(1).rolling(6).std())
    for col in ["n_ports", "mean_NC", "mean_observations_per_port"]:
        panel[col] = group[col].shift(1)

    panel["target_month_str"] = panel["month_str"]
    panel = panel.dropna(subset=["target_return"]).reset_index(drop=True)

    market_countries = set(market_returns["country"].dropna().unique())
    gnc_countries = set(country_gnc["country"].dropna().unique())
    panel_countries = set(panel["country"].dropna().unique())
    diagnostics = {
        "forecast_horizon_months": forecast_horizon_months,
        "market_countries": sorted(market_countries),
        "gnc_countries": sorted(gnc_countries),
        "panel_countries": sorted(panel_countries),
        "countries_missing_market_index": sorted(gnc_countries - market_countries),
        "countries_missing_gnc": sorted(market_countries - gnc_countries),
        "rows_before_feature_drop": int(len(panel)),
    }
    return panel, diagnostics


def feature_ready_panel(panel: pd.DataFrame, min_non_null_features: int = 3) -> pd.DataFrame:
    out = panel.copy()
    feature_non_null = out[FEATURE_COLUMNS].notna().sum(axis=1)
    out = out[feature_non_null >= min_non_null_features].copy()
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)
