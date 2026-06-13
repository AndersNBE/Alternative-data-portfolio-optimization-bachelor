"""
Builds a monthly container index from infer.py's predictions.csv.

Pipeline:

  1. Parse daily date from timestamp column.
  2. Sum pred_container_pixels across all patches for the same (port, date)
     -> daily NC_{port, date}
  3. Aggregate daily NC to one monthly port-level signal using the median
     across available observations in the month
     -> monthly NC_{port, month}
  4. Compute monthly GNC (Growth in Number of Containers):
       GNC_{port, m} = [log(NC_m) - log(NC_prev)] / month_gap
  5. Map each port to its country.
  6. Average monthly GNC across all ports that belong to the same country
     on the same month -> country_gnc.csv.

Outputs
-------
  port_timeseries.csv   : monthly NC and GNC per (port, month)
  country_gnc.csv       : monthly mean GNC and port coverage per country

Usage
-----
  python -m analysis.build_container_index \
    --predictions predictions.csv \
    --out data/outputs/container_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.port_country_map import port_to_country


# -- Step 1 & 2: load predictions and aggregate pixels per (port, date) --------

def load_daily_nc(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)

    # timestamp is like "20170104T065252Z" -- keep only the date part
    df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d").dt.date

    if "pred_container_pixels" in df.columns:
        pixel_col = "pred_container_pixels"
    else:
        raise ValueError("predictions.csv must contain 'pred_container_pixels'.")

    # Sum across patches for the same port on the same date.
    nc = (
        df.groupby(["port_id", "date"])[pixel_col]
        .sum()
        .reset_index()
        .rename(columns={pixel_col: "NC"})
    )
    nc["date"] = pd.to_datetime(nc["date"])
    return nc.sort_values(["port_id", "date"]).reset_index(drop=True)


# -- Step 3: aggregate daily NC to monthly port levels -------------------------

def build_monthly_port_levels(daily_nc: pd.DataFrame) -> pd.DataFrame:
    df = daily_nc.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(["port_id", "month"])
        .agg(
            n_observations=("date", "count"),
            NC_mean=("NC", "mean"),
            NC_median=("NC", "median"),
            NC_std=("NC", "std"),
            NC_min=("NC", "min"),
            NC_max=("NC", "max"),
        )
        .reset_index()
    )
    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly["month_str"] = monthly["month"].astype(str)
    monthly["NC_std"] = monthly["NC_std"].fillna(0.0)
    # Use monthly median as the robust monthly level used for GNC.
    monthly["NC"] = monthly["NC_median"]
    return monthly.sort_values(["port_id", "date"]).reset_index(drop=True)


# -- Step 4: compute monthly GNC per port --------------------------------------

def compute_gnc_for_port(port_df: pd.DataFrame) -> pd.DataFrame:
    port_df = port_df.sort_values("date").copy()

    port_df["month_index"] = port_df["date"].dt.year * 12 + port_df["date"].dt.month
    port_df["NC_prev"] = port_df["NC"].shift(1)
    port_df["month_index_prev"] = port_df["month_index"].shift(1)
    port_df["month_gap"] = port_df["month_index"] - port_df["month_index_prev"]

    # Clip to 1 so log is always defined (0 container pixels -> treat as 1).
    log_nc = np.log(port_df["NC"].clip(lower=1))
    log_nc_prev = np.log(port_df["NC_prev"].clip(lower=1))

    port_df["GNC"] = (log_nc - log_nc_prev) / port_df["month_gap"]

    return (
        port_df
        .dropna(subset=["GNC"])
        .drop(columns=["month", "month_index", "NC_prev", "month_index_prev", "month_gap"])
        .reset_index(drop=True)
    )


def build_port_timeseries(daily_nc: pd.DataFrame) -> pd.DataFrame:
    monthly_levels = build_monthly_port_levels(daily_nc)
    parts = []
    for port_id, group in monthly_levels.groupby("port_id"):
        result = compute_gnc_for_port(group)
        result["port_id"] = port_id
        parts.append(result)
    if not parts:
        return pd.DataFrame(
            columns=[
                "port_id",
                "date",
                "month_str",
                "n_observations",
                "NC",
                "NC_mean",
                "NC_median",
                "NC_std",
                "NC_min",
                "NC_max",
                "GNC",
            ]
        )
    return pd.concat(parts, ignore_index=True)


# -- Step 5 & 6: map to country and average across ports -----------------------

def build_country_gnc(port_ts: pd.DataFrame) -> pd.DataFrame:
    df = port_ts.copy()
    df["country"] = df["port_id"].map(port_to_country)

    unknown = df[df["country"].isna()]["port_id"].unique()
    if len(unknown) > 0:
        print(f"  Warning: {len(unknown)} port(s) have no country mapping: {sorted(unknown)}")

    df = df.dropna(subset=["country"])

    country_gnc = (
        df.groupby(["country", "date", "month_str"])
        .agg(
            GNC=("GNC", "mean"),
            n_ports=("port_id", "nunique"),
            mean_NC=("NC", "mean"),
            mean_observations_per_port=("n_observations", "mean"),
        )
        .reset_index()
    )
    return country_gnc.sort_values(["country", "date"]).reset_index(drop=True)


# -- Main ----------------------------------------------------------------------

def run(predictions_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1+2: Loading predictions and summing pixels per (port, date)...")
    daily_nc = load_daily_nc(predictions_csv)
    print(f"  {len(daily_nc):,} daily (port, date) observations across {daily_nc['port_id'].nunique()} ports")

    print("Step 3+4: Aggregating to monthly port index and computing monthly GNC...")
    port_ts = build_port_timeseries(daily_nc)
    port_ts_path = out_dir / "port_timeseries.csv"
    port_ts.to_csv(port_ts_path, index=False)
    print(f"  Saved {len(port_ts):,} monthly port rows -> {port_ts_path}")

    print("Step 5+6: Mapping to countries and averaging monthly GNC...")
    country_gnc = build_country_gnc(port_ts)
    country_gnc_path = out_dir / "country_gnc.csv"
    country_gnc.to_csv(country_gnc_path, index=False)
    print(
        f"  Saved {len(country_gnc):,} monthly country rows across "
        f"{country_gnc['country'].nunique()} countries -> {country_gnc_path}"
    )

    print("\nDone. Output files:")
    print(f"  {port_ts_path}")
    print(f"  {country_gnc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build container index from predictions.csv.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv from infer.py.")
    parser.add_argument("--out", required=True, help="Output directory for index CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        predictions_csv=Path(args.predictions),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
