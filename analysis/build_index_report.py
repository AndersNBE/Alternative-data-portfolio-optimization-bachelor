"""
Builds a visual report from the monthly container index outputs.

Reads port_timeseries.csv and country_gnc.csv (produced by build_container_index.py)
and saves plots + a per-port summary table to the output directory.

Plots produced
--------------
  port_nc/<port>.png        : monthly NC over time per port
  country_gnc/<country>.png : monthly GNC signal per country
  coverage_heatmap.png      : monthly observation coverage per port
  port_summary.csv          : per-port monthly summary stats

Usage
-----
  python -m analysis.build_index_report \
    --port-timeseries data/outputs/container_index/port_timeseries.csv \
    --country-gnc     data/outputs/container_index/country_gnc.csv \
    --out             data/outputs/container_index/report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")


# -- Per-port NC timeseries ----------------------------------------------------

def plot_port_nc(port_ts: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "port_nc"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for port_id, group in port_ts.groupby("port_id"):
        group = group.sort_values("date")
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        axes[0].plot(group["date"], group["NC"], marker="o", markersize=3, linewidth=1, color="steelblue")
        axes[0].set_title(f"{port_id} -- monthly container index (median NC)")
        axes[0].set_ylabel("Monthly NC")

        axes[1].bar(group["date"], group["n_observations"], width=20, color="steelblue", alpha=0.5)
        axes[1].set_ylabel("Obs/mo")
        axes[1].set_xlabel("Month")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()
        fig.savefig(plot_dir / f"{port_id}.png", dpi=120)
        plt.close(fig)

    print(f"  Saved {port_ts['port_id'].nunique()} port NC plots -> {plot_dir}")


# -- Per-country GNC timeseries ------------------------------------------------

def plot_country_gnc(country_gnc: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "country_gnc"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for country, group in country_gnc.groupby("country"):
        group = group.sort_values("date").copy()
        # Country-level file stores mean observations per contributing port.
        # Multiply by active ports to recover total observation count in the month.
        group["total_observations"] = (
            group["mean_observations_per_port"] * group["n_ports"]
        ).round()
        for window in (2, 3, 4):
            group[f"GNC_roll_{window}m"] = group["GNC"].rolling(window=window, min_periods=1).mean()

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )

        axes[0].plot(
            group["date"],
            group["GNC"],
            marker="o",
            markersize=2.5,
            linewidth=0.9,
            color="lightsteelblue",
            alpha=0.8,
            label="Raw monthly GNC",
        )
        axes[0].plot(
            group["date"],
            group["GNC_roll_2m"],
            linewidth=1.4,
            color="tab:orange",
            label="2-month rolling mean",
        )
        axes[0].plot(
            group["date"],
            group["GNC_roll_3m"],
            linewidth=1.6,
            color="tab:green",
            label="3-month rolling mean",
        )
        axes[0].plot(
            group["date"],
            group["GNC_roll_4m"],
            linewidth=1.8,
            color="tab:red",
            label="4-month rolling mean",
        )
        axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes[0].set_title(f"{country} -- monthly GNC (container growth signal)")
        axes[0].set_ylabel("Monthly GNC")
        axes[0].legend(loc="upper left", fontsize=8)

        axes[1].bar(group["date"], group["total_observations"], width=20, color="steelblue", alpha=0.5)
        axes[1].set_ylabel("Obs/mo")

        axes[2].bar(group["date"], group["n_ports"], width=20, color="slategray", alpha=0.6)
        axes[2].set_ylabel("Ports")
        axes[2].set_xlabel("Month")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[2].xaxis.set_major_locator(mdates.YearLocator())

        plt.tight_layout()
        safe_name = country.replace("/", "_").replace(" ", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=120)
        plt.close(fig)

    print(f"  Saved {country_gnc['country'].nunique()} country GNC plots -> {plot_dir}")


# -- Coverage heatmap ----------------------------------------------------------

def plot_coverage_heatmap(port_ts: pd.DataFrame, out_dir: Path) -> None:
    df = port_ts.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    pivot = df.pivot_table(
        index="port_id",
        columns="year_month",
        values="n_observations",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) // 3), max(8, len(pivot) // 2)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Blues",
        linewidths=0,
        cbar_kws={"label": "Observations within month"},
        xticklabels=max(1, len(pivot.columns) // 24),
    )
    ax.set_title("Port monthly coverage")
    ax.set_xlabel("Month")
    ax.set_ylabel("Port")
    plt.tight_layout()
    path = out_dir / "coverage_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved coverage heatmap -> {path}")


# -- Per-port summary table ----------------------------------------------------

def build_port_summary(port_ts: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    from analysis.port_country_map import port_to_country

    summary = (
        port_ts.groupby("port_id")
        .agg(
            country=("port_id", lambda s: port_to_country(s.iloc[0]) or "-"),
            n_months=("date", "count"),
            date_first=("date", "min"),
            date_last=("date", "max"),
            mean_NC=("NC", "mean"),
            median_NC=("NC", "median"),
            mean_GNC=("GNC", "mean"),
            std_GNC=("GNC", "std"),
            mean_obs_per_month=("n_observations", "mean"),
        )
        .reset_index()
        .sort_values("port_id")
    )
    summary["date_first"] = summary["date_first"].dt.strftime("%Y-%m-%d")
    summary["date_last"] = summary["date_last"].dt.strftime("%Y-%m-%d")
    summary["mean_NC"] = summary["mean_NC"].round(0).astype(int)
    summary["median_NC"] = summary["median_NC"].round(0).astype(int)
    summary["mean_GNC"] = summary["mean_GNC"].round(4)
    summary["std_GNC"] = summary["std_GNC"].round(4)
    summary["mean_obs_per_month"] = summary["mean_obs_per_month"].round(2)

    path = out_dir / "port_summary.csv"
    summary.to_csv(path, index=False)
    print(f"  Saved port summary -> {path}")
    return summary


# -- Main ----------------------------------------------------------------------

def run(port_timeseries_csv: Path, country_gnc_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading index data...")
    port_ts = pd.read_csv(port_timeseries_csv, parse_dates=["date"])
    country_gnc = pd.read_csv(country_gnc_csv, parse_dates=["date"])
    print(f"  {len(port_ts):,} port-month rows, {port_ts['port_id'].nunique()} ports")
    print(f"  {len(country_gnc):,} country-month rows, {country_gnc['country'].nunique()} countries")

    print("Building port summary table...")
    summary = build_port_summary(port_ts, out_dir)
    print(summary[["port_id", "country", "n_months", "date_first", "date_last", "mean_NC"]].to_string(index=False))

    print("\nPlotting per-port monthly NC timeseries...")
    plot_port_nc(port_ts, out_dir)

    print("Plotting per-country monthly GNC timeseries...")
    plot_country_gnc(country_gnc, out_dir)

    print("Plotting coverage heatmap...")
    plot_coverage_heatmap(port_ts, out_dir)

    print(f"\nAll outputs in: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual report from container index.")
    parser.add_argument("--port-timeseries", required=True)
    parser.add_argument("--country-gnc", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        port_timeseries_csv=Path(args.port_timeseries),
        country_gnc_csv=Path(args.country_gnc),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
