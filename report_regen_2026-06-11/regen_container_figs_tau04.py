from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent
FINAL_RUN_COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
FINAL_RUN_ROOT = "final_runs/tau04_hk_28623539"
FINAL_RUN_ROI10 = FINAL_RUN_ROOT  # tau04-runnet har ingen runs/roi10-undermappe; ROI10 er selve kørslen
MAD_SUITE = f"{FINAL_RUN_ROOT}/return_forecasting/usd19_no_sri_lanka_28623539"
SOURCE_SHORT = "Source: locked bd5d48e final_runs/tau04_hk_28623539 (tau=0.4, ROI10)."
OUT_DIR = AUDIT / "container_index_tau04_bundle"

# Run-mappingen (Hong Kong separat) — afviger bevidst fra repoets analysis/port_country_map.py,
# som stadig har hong_kong -> Kina. tau04_hk-kørslen brugte HK som selvstændigt land.
PORT_COUNTRY = {
    "abu_dhabi": "UAE", "algeciras": "Spanien", "antwerpbrugges": "Belgien",
    "balboa": "Panama", "bremen": "Tyskland", "busan": "Sydkorea",
    "cai_mep": "Vietnam", "colombo": "Sri Lanka", "colon": "Panama",
    "da_lian": "Kina", "dongguan": "Kina", "guangxi_beibu": "Kina",
    "guangzhou": "Kina", "hai_phong": "Vietnam", "hamborg": "Tyskland",
    "ho_chi_minh_city": "Vietnam", "hong_kong": "Hong Kong", "houston": "USA",
    "jawaharal_nehru": "Indien", "jebel_ali": "UAE", "kaohsiung": "Taiwan",
    "laem_chabang": "Thailand", "lianyungang": "Kina", "long_beach": "USA",
    "los_angeles": "USA", "manila": "Filippinerne", "mundra": "Indien",
    "new_york_new_jersey": "USA", "ningbozhoushan": "Kina", "piraeus": "Grækenland",
    "port_klang": "Malaysia", "qing_dao": "Kina", "rizhao": "Kina",
    "rotterdam": "Holland", "santos": "Brasilien", "savannah": "USA",
    "shanghai": "Kina", "shenzhen": "Kina", "singapore": "Singapore",
    "suzhou": "Kina", "tanger_med": "Marokko", "tanjung_pelepas": "Malaysia",
    "tanjung_perak": "Malaysia", "tanjung_priok": "Indonesia", "tianjin": "Kina",
    "tokyo": "Japan", "valencia": "Spanien", "xiamen": "Kina",
    "yantai": "Kina", "yingkou": "Kina",
}


def port_to_country(port_slug: str):
    return PORT_COUNTRY.get(port_slug)


BG = "#fafaf7"
PANEL = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
BLUE = "#2b9fd6"
BLUE_DARK = "#0b4775"
BLUE_LIGHT = "#9bd2ee"
GREEN = "#00a878"
GREEN_DARK = "#006d58"
GOLD = "#edae13"
GOLD_DARK = "#a86700"
PINK = "#d889b8"
PINK_DARK = "#9b3e5d"
NEUTRAL = "#66717c"

OBS_CMAP = LinearSegmentedColormap.from_list("obs_blue", ["#f2f7fb", "#9bd2ee", "#2b9fd6", "#0b4775"])
GNC_CMAP = LinearSegmentedColormap.from_list("gnc_teal", ["#eff7f5", "#84d2c0", "#00a878", "#006d58"])

PORT_LABELS = {
    "abu_dhabi": "Abu Dhabi",
    "algeciras": "Algeciras",
    "antwerpbrugges": "Antwerp-Bruges",
    "balboa": "Balboa",
    "bremen": "Bremen",
    "busan": "Busan",
    "cai_mep": "Cai Mep",
    "colombo": "Colombo",
    "colon": "Colon",
    "da_lian": "Dalian",
    "dongguan": "Dongguan",
    "guangxi_beibu": "Guangxi Beibu",
    "guangzhou": "Guangzhou",
    "hai_phong": "Hai Phong",
    "hamborg": "Hamburg",
    "ho_chi_minh_city": "Ho Chi Minh City",
    "hong_kong": "Hong Kong",
    "houston": "Houston",
    "jawaharal_nehru": "Jawaharlal Nehru",
    "jebel_ali": "Jebel Ali",
    "kaohsiung": "Kaohsiung",
    "laem_chabang": "Laem Chabang",
    "lianyungang": "Lianyungang",
    "long_beach": "Long Beach",
    "manila": "Manila",
    "mundra": "Mundra",
    "new_york_new_jersey": "New York/New Jersey",
    "ningbozhoushan": "Ningbo-Zhoushan",
    "piraeus": "Piraeus",
    "port_klang": "Port Klang",
    "qing_dao": "Qingdao",
    "rizhao": "Rizhao",
    "rotterdam": "Rotterdam",
    "santos": "Santos",
    "savannah": "Savannah",
    "shanghai": "Shanghai",
    "shenzhen": "Shenzhen",
    "singapore": "Singapore",
    "suzhou": "Suzhou",
    "tanger_med": "Tanger Med",
    "tanjung_pelepas": "Tanjung Pelepas",
    "tanjung_perak": "Tanjung Perak",
    "tanjung_priok": "Tanjung Priok",
    "tianjin": "Tianjin",
    "tokyo": "Tokyo",
    "valencia": "Valencia",
    "xiamen": "Xiamen",
    "yantai": "Yantai",
    "yingkou": "Yingkou",
}

COUNTRY_LABELS = {
    "Belgien": "Belgium",
    "Brasilien": "Brazil",
    "Filippinerne": "Philippines",
    "Holland": "Netherlands",
    "Indien": "India",
    "Kina": "China",
    "Sydkorea": "South Korea",
    "Tyskland": "Germany",
    "Spanien": "Spain",
}

SELECTED_COUNTRIES = ["Kina", "USA", "Indien", "Malaysia", "Vietnam", "Tyskland", "UAE", "Spanien"]
MAD_COPY_FILES = [
    "gnc_mad_top15_sharpe.png",
    "gnc_mad_best_sharpe_by_forecast_method.png",
    "gnc_mad_method_performance_dashboard.png",
]

METHOD_META = {
    "elastic_net_panel": ("Elastic net panel", BLUE),
    "random_forest_panel": ("Random forest panel", GREEN),
    "ols_predictive": ("OLS predictive", GOLD),
    "distributed_lag": ("Distributed lag", PINK),
}
METHOD_ORDER = ["elastic_net_panel", "random_forest_panel", "ols_predictive", "distributed_lag"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "legend.frameon": False,
        }
    )


def port_label(port_id: str) -> str:
    return PORT_LABELS.get(port_id, port_id.replace("_", " ").title())


def country_label(country: str) -> str:
    return COUNTRY_LABELS.get(country, country)


def add_header(fig: plt.Figure, title: str, subtitle: str, x: float = 0.06, y: float = 0.965) -> None:
    fig.text(x, y, title, ha="left", va="top", fontsize=17.5, weight="bold", color=INK)
    if subtitle:
        fig.text(x, y - 0.038, subtitle, ha="left", va="top", fontsize=10.2, color=MUTED)


def add_source(fig: plt.Figure, text: str, x: float = 0.06, y: float = 0.035) -> None:
    fig.text(x, y, text, ha="left", va="bottom", fontsize=8.6, color=MUTED)


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.82)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.5, color=AXIS, pad=5)


def format_year_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))


def save(fig: plt.Figure, name: str, pad: float = 0.12) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(path)


def git_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(REPO / relative_path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Kanonisk kontraktfil: port-dato-niveau med NC (daglig pixelsum) og
    # observation_count (antal patch-billeder den dato).
    daily = git_csv(f"{FINAL_RUN_ROOT}/daily_container_index/port_timeseries.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns={"NC": "NC_daily", "observation_count": "patch_images"})
    daily["NC_daily"] = pd.to_numeric(daily["NC_daily"], errors="coerce").fillna(0)
    daily["patch_images"] = pd.to_numeric(daily["patch_images"], errors="coerce").fillna(0)
    daily = daily.sort_values(["port_id", "date"])
    pred = daily.copy()
    daily["pixels_per_patch_image_daily"] = daily["NC_daily"] / daily["patch_images"].clip(lower=1)
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    port = (
        daily.groupby(["port_id", "month"], as_index=False)
        .agg(
            n_observations=("date", "nunique"),
            NC_mean=("NC_daily", "mean"),
            NC_median=("NC_daily", "median"),
            NC_std=("NC_daily", "std"),
            NC_min=("NC_daily", "min"),
            NC_max=("NC_daily", "max"),
            patch_images=("patch_images", "sum"),
            pixels_per_patch_image=("pixels_per_patch_image_daily", "median"),
        )
        .rename(columns={"month": "date"})
        .sort_values(["port_id", "date"])
    )
    port["month_str"] = port["date"].dt.strftime("%Y-%m")
    port["NC_std"] = port["NC_std"].fillna(0)
    port["NC"] = port["NC_median"]
    port = compute_monthly_growth(port)
    port = port[port["GNC"].notna()].copy()
    port["country"] = port["port_id"].map(port_to_country)
    port["port_label"] = port["port_id"].map(port_label)
    port["country_label"] = port["country"].map(country_label)

    # Månedligt landepanel bygges fra port-panelet (gennemsnit af aktive havnes
    # GNC per land-måned), som beskrevet i rapportteksten. tau04-runnet har ikke
    # nogen månedlig landefil; daily_container_index/country_gnc.csv er på datoniveau.
    country = (
        port.groupby(["country", "date"], as_index=False)
        .agg(
            GNC=("GNC", "mean"),
            n_ports=("port_id", "nunique"),
            mean_observations_per_port=("n_observations", "mean"),
            total_observation_dates=("n_observations", "sum"),
        )
    )
    country = country[country["country"].notna()].copy()
    country["country_label"] = country["country"].map(country_label)
    return port, country, pred, daily


def compute_monthly_growth(port: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for port_id, group in port.sort_values("date").groupby("port_id"):
        g = group.copy()
        g["month_index"] = g["date"].dt.year * 12 + g["date"].dt.month
        g["NC_prev"] = g["NC"].shift(1)
        g["patch_norm_level_prev"] = g["pixels_per_patch_image"].shift(1)
        g["month_index_prev"] = g["month_index"].shift(1)
        g["month_gap"] = g["month_index"] - g["month_index_prev"]
        raw_current = np.log(g["NC"].clip(lower=1))
        raw_previous = np.log(g["NC_prev"].clip(lower=1))
        current = np.log(g["pixels_per_patch_image"].clip(lower=1))
        previous = np.log(g["patch_norm_level_prev"].clip(lower=1))
        g["GNC"] = (raw_current - raw_previous) / g["month_gap"]
        g["patch_norm_gnc"] = (current - previous) / g["month_gap"]
        parts.append(g.drop(columns=["NC_prev", "patch_norm_level_prev", "month_index_prev", "month_gap"]))
    return pd.concat(parts, ignore_index=True)


def coverage_matrix(port: pd.DataFrame, value: str) -> pd.DataFrame:
    df = port.copy()
    df["month"] = df["date"].dt.to_period("M")
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    order = (
        df[["port_id", "country", "country_label", "port_label"]]
        .drop_duplicates()
        .assign(country_sort=lambda x: x["country_label"].fillna(""))
        .sort_values(["country_sort", "port_label", "port_id"])
    )
    pivot = df.pivot_table(index="port_id", columns="month", values=value, aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=order["port_id"], columns=months, fill_value=0)
    pivot.index = [port_label(p) for p in pivot.index]
    return pivot


def plot_coverage_heatmap(port: pd.DataFrame) -> None:
    pivot = coverage_matrix(port, "n_observations")
    values = pivot.to_numpy(dtype=float)
    fig = plt.figure(figsize=(13.2, 8.9), constrained_layout=False)
    ax = fig.add_axes([0.18, 0.14, 0.70, 0.72])
    cax = fig.add_axes([0.90, 0.24, 0.018, 0.50])
    add_header(
        fig,
        "Monthly observation coverage by port",
        f"{pivot.shape[0]} ports; cells count usable observation dates in each port-month.",
    )
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=OBS_CMAP, vmin=0, vmax=np.nanmax(values))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6.8)
    years = [i for i, p in enumerate(pivot.columns) if p.month == 1]
    ax.set_xticks(years)
    ax.set_xticklabels([str(pivot.columns[i].year) for i in years], fontsize=8.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Port")
    ax.tick_params(axis="x", rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Observation dates", color=MUTED)
    add_source(fig, f"{SOURCE_SHORT} Valid monthly GNC rows span 2017-02 to 2026-04.")
    save(fig, "container_index_coverage_heatmap.png")


def plot_patch_image_coverage_heatmap(port: pd.DataFrame) -> None:
    pivot = coverage_matrix(port, "patch_images")
    values = pivot.to_numpy(dtype=float)
    fig = plt.figure(figsize=(13.2, 8.9), constrained_layout=False)
    ax = fig.add_axes([0.18, 0.14, 0.70, 0.72])
    cax = fig.add_axes([0.90, 0.24, 0.018, 0.50])
    add_header(
        fig,
        "Monthly patch-image coverage by port",
        f"{pivot.shape[0]} ports; cells count contributing patch images in each valid port-month.",
    )
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=OBS_CMAP, vmin=0, vmax=np.nanmax(values))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6.8)
    years = [i for i, p in enumerate(pivot.columns) if p.month == 1]
    ax.set_xticks(years)
    ax.set_xticklabels([str(pivot.columns[i].year) for i in years], fontsize=8.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Port")
    ax.tick_params(axis="x", rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Patch images", color=MUTED)
    add_source(fig, f"{SOURCE_SHORT} Patch counts are reconstructed from observation-level port_timeseries.csv.")
    save(fig, "container_index_patch_image_coverage_heatmap.png")


def plot_shanghai_nc(port: pd.DataFrame) -> None:
    g = port[port["port_id"] == "shanghai"].sort_values("date").copy()
    g["NC_ma3"] = g["NC"].rolling(3, min_periods=2).mean()
    fig = plt.figure(figsize=(10.8, 5.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.1], left=0.09, right=0.96, top=0.80, bottom=0.16, hspace=0.13)
    ax = fig.add_subplot(gs[0])
    ax_obs = fig.add_subplot(gs[1], sharex=ax)
    add_header(fig, "Shanghai monthly container activity", "Monthly median NC with usable observation dates below.")
    style_axis(ax, "y")
    style_axis(ax_obs, "y")
    ax.plot(g["date"], g["NC"], color=BLUE_LIGHT, linewidth=1.0, marker="o", markersize=2.8, label="Monthly NC")
    ax.plot(g["date"], g["NC_ma3"], color=BLUE_DARK, linewidth=2.0, label="3-month mean")
    ax.set_ylabel("Monthly NC")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="upper left", ncols=2, fontsize=8.6)
    ax_obs.bar(g["date"], g["n_observations"], width=22, color=BLUE, alpha=0.62)
    ax_obs.set_ylabel("Obs.")
    ax_obs.set_xlabel("Month")
    ax_obs.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
    format_year_axis(ax_obs)
    plt.setp(ax.get_xticklabels(), visible=False)
    add_source(fig, f"{SOURCE_SHORT} NC is the monthly median of daily predicted container pixels for Shanghai.")
    save(fig, "container_index_port_shanghai_nc.png")


def plot_shanghai_total_vs_patch(port: pd.DataFrame) -> None:
    g = port[port["port_id"] == "shanghai"].sort_values("date").copy()
    g["NC_ma3"] = g["NC"].rolling(3, min_periods=2).mean()
    g["per_patch_ma3"] = g["pixels_per_patch_image"].rolling(3, min_periods=2).mean()
    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 2.0, 1.0], left=0.09, right=0.96, top=0.82, bottom=0.14, hspace=0.13)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    add_header(fig, "Shanghai raw and patch-normalized activity", "Top: monthly NC; middle: predicted pixels per patch image; bottom: patch-image coverage.")
    for ax in axes:
        style_axis(ax, "y")
    axes[0].plot(g["date"], g["NC"], color=BLUE_LIGHT, linewidth=1.0, marker="o", markersize=2.4, label="Monthly NC")
    axes[0].plot(g["date"], g["NC_ma3"], color=BLUE_DARK, linewidth=1.8, label="3-month mean")
    axes[0].set_ylabel("NC")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].legend(loc="upper left", ncols=2, fontsize=8.4)
    axes[1].plot(g["date"], g["pixels_per_patch_image"], color=GREEN, linewidth=1.0, marker="o", markersize=2.4, alpha=0.75)
    axes[1].plot(g["date"], g["per_patch_ma3"], color=GREEN_DARK, linewidth=1.8)
    axes[1].set_ylabel("Pixels / patch")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[2].bar(g["date"], g["patch_images"], width=22, color=NEUTRAL, alpha=0.62)
    axes[2].set_ylabel("Patch imgs")
    axes[2].set_xlabel("Month")
    axes[2].yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
    format_year_axis(axes[2])
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    add_source(fig, f"{SOURCE_SHORT} Patch-normalized activity is reconstructed from observation-level patch counts.")
    save(fig, "container_index_port_shanghai_total_vs_patch_normalized.png")


def plot_china_country_gnc(country: pd.DataFrame) -> None:
    g = country[country["country"] == "Kina"].sort_values("date").copy()
    g["plot_observations"] = g.get("total_observation_dates", g["mean_observations_per_port"] * g["n_ports"])
    for window in (2, 3, 4):
        g[f"GNC_roll_{window}m"] = g["GNC"].rolling(window, min_periods=1).mean()
    fig = plt.figure(figsize=(10.8, 6.9), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.1, 1.1], left=0.09, right=0.96, top=0.82, bottom=0.14, hspace=0.12)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    add_header(fig, "China monthly GNC signal", "Raw country GNC with 2-, 3- and 4-month moving averages plus coverage context.")
    for ax in axes:
        style_axis(ax, "y")
    axes[0].plot(g["date"], g["GNC"], color=BLUE_LIGHT, linewidth=1.0, marker="o", markersize=2.3, alpha=0.78, label="Monthly GNC")
    axes[0].plot(g["date"], g["GNC_roll_2m"], color=GOLD, linewidth=1.5, label="2m mean")
    axes[0].plot(g["date"], g["GNC_roll_3m"], color=GREEN, linewidth=1.8, label="3m mean")
    axes[0].plot(g["date"], g["GNC_roll_4m"], color=PINK_DARK, linewidth=1.7, label="4m mean")
    axes[0].axhline(0, color=AXIS, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("GNC")
    axes[0].legend(loc="upper left", ncols=4, fontsize=8.0)
    axes[1].bar(g["date"], g["plot_observations"], width=22, color=BLUE, alpha=0.58)
    axes[1].set_ylabel("Obs.")
    axes[2].bar(g["date"], g["n_ports"], width=22, color=NEUTRAL, alpha=0.62)
    axes[2].set_ylabel("Ports")
    axes[2].set_xlabel("Month")
    axes[2].yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
    format_year_axis(axes[2])
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    add_source(fig, f"{SOURCE_SHORT} Country GNC is the mean of active ports' monthly GNC per country-month.")
    save(fig, "container_index_country_kina_gnc.png")


def plot_country_panel(country: pd.DataFrame) -> None:
    panel = country[country["country"].isin(SELECTED_COUNTRIES)].sort_values("date").copy()
    panel["GNC_ma3"] = panel.groupby("country")["GNC"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    fig, axes = plt.subplots(4, 2, figsize=(11.6, 9.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.86, bottom=0.12, hspace=0.34, wspace=0.16)
    add_header(fig, "Country-level container activity growth signals", "Selected country GNC series; each panel uses its own y-axis scale.")
    for ax, country_name in zip(axes.flat, SELECTED_COUNTRIES):
        g = panel[panel["country"] == country_name]
        style_axis(ax, "y")
        ax.plot(g["date"], g["GNC"], color=BLUE_LIGHT, linewidth=0.9, marker="o", markersize=1.8, alpha=0.74)
        ax.plot(g["date"], g["GNC_ma3"], color=GOLD_DARK, linewidth=1.55)
        ax.axhline(0, color=AXIS, linewidth=0.7, linestyle=":")
        med_ports = g["n_ports"].median()
        ax.set_title(f"{country_label(country_name)} (median active ports: {med_ports:.0f})", loc="left", fontsize=9.0, weight="bold")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        format_year_axis(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel("Month")
    add_source(fig, f"{SOURCE_SHORT} Blue line: monthly GNC. Gold line: 3-month moving average for visual inspection only.")
    save(fig, "container_index_country_gnc_panel_selected.png", pad=0.10)


def plot_china_port_small_multiples(port: pd.DataFrame, country: pd.DataFrame) -> None:
    china_ports = sorted(port.loc[port["country"] == "Kina", "port_id"].dropna().unique(), key=port_label)
    china = country[country["country"] == "Kina"].sort_values("date").copy()
    china["GNC_ma3"] = china["GNC"].rolling(3, min_periods=1).mean()
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 9.3), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.11, hspace=0.40, wspace=0.18)
    add_header(fig, "Chinese port-level GNC signals", "Each panel shows one port; black line is the country-level 3-month China GNC mean.")
    for ax in axes.flat[len(china_ports):]:
        ax.set_visible(False)
    for ax, port_id in zip(axes.flat, china_ports):
        g = port[port["port_id"] == port_id].sort_values("date")
        style_axis(ax, "y")
        ax.plot(g["date"], g["GNC"], color=GREEN, linewidth=0.85, marker="o", markersize=1.6, alpha=0.72)
        ax.plot(china["date"], china["GNC_ma3"], color=INK, linewidth=1.05, alpha=0.80)
        ax.axhline(0, color=AXIS, linewidth=0.65, linestyle=":")
        ax.set_title(port_label(port_id), loc="left", fontsize=8.7, weight="bold")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        format_year_axis(ax)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.tick_params(axis="y", labelsize=7.2)
    add_source(fig, f"{SOURCE_SHORT} Port rows are reconstructed from observation-level port_timeseries.csv; panels use local y-axis scales.")
    save(fig, "container_index_china_port_gnc_small_multiples.png", pad=0.10)


def patch_normalized_country(port: pd.DataFrame, country_name: str) -> pd.DataFrame:
    subset = port[(port["country"] == country_name) & port["patch_norm_gnc"].notna()].copy()
    grouped = (
        subset.groupby("date", as_index=False)
        .agg(
            patch_norm_gnc=("patch_norm_gnc", "mean"),
            mean_patch_images_per_port=("patch_images", "mean"),
            n_ports_patch_norm=("port_id", "nunique"),
        )
        .sort_values("date")
    )
    grouped["patch_norm_gnc_ma3"] = grouped["patch_norm_gnc"].rolling(3, min_periods=1).mean()
    return grouped


def plot_china_raw_vs_patch_norm(port: pd.DataFrame, country: pd.DataFrame) -> None:
    raw = country[country["country"] == "Kina"].sort_values("date").copy()
    raw["GNC_ma3"] = raw["GNC"].rolling(3, min_periods=1).mean()
    patch_country = patch_normalized_country(port, "Kina")
    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 1.0], left=0.09, right=0.96, top=0.82, bottom=0.14, hspace=0.13)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    add_header(fig, "China raw versus patch-normalized GNC", "Patch-normalized GNC recomputes growth from predicted pixels per patch image.")
    for ax in axes:
        style_axis(ax, "y")
    axes[0].plot(raw["date"], raw["GNC"], color=BLUE_LIGHT, linewidth=0.95, marker="o", markersize=2.0, alpha=0.75)
    axes[0].plot(raw["date"], raw["GNC_ma3"], color=BLUE_DARK, linewidth=1.8)
    axes[0].axhline(0, color=AXIS, linewidth=0.75, linestyle=":")
    axes[0].set_ylabel("Raw GNC")
    axes[1].plot(patch_country["date"], patch_country["patch_norm_gnc"], color=GREEN, linewidth=0.95, marker="o", markersize=2.0, alpha=0.75)
    axes[1].plot(patch_country["date"], patch_country["patch_norm_gnc_ma3"], color=GREEN_DARK, linewidth=1.8)
    axes[1].axhline(0, color=AXIS, linewidth=0.75, linestyle=":")
    axes[1].set_ylabel("Patch-norm. GNC")
    axes[2].bar(patch_country["date"], patch_country["mean_patch_images_per_port"], width=22, color=NEUTRAL, alpha=0.62)
    axes[2].set_ylabel("Patch imgs")
    axes[2].set_xlabel("Month")
    format_year_axis(axes[2])
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    add_source(fig, f"{SOURCE_SHORT} Patch-normalized GNC is reconstructed from observation-level patch counts.")
    save(fig, "container_index_country_kina_raw_vs_patch_normalized_gnc.png")


def plot_patch_count_vs_abs_gnc(port: pd.DataFrame) -> None:
    df = port[(port["patch_images"] > 0) & port["GNC"].notna()].copy()
    df["abs_gnc"] = df["GNC"].abs()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["patch_images", "abs_gnc"])
    df["log_patch"] = np.log10(df["patch_images"])
    df["bin"] = pd.qcut(df["log_patch"], q=14, duplicates="drop")
    summary = (
        df.groupby("bin", observed=True)
        .agg(
            x=("patch_images", "median"),
            median_abs_gnc=("abs_gnc", "median"),
            q75_abs_gnc=("abs_gnc", lambda s: s.quantile(0.75)),
            n=("abs_gnc", "size"),
        )
        .reset_index(drop=True)
    )
    rho = df["patch_images"].rank().corr(df["abs_gnc"].rank())
    from scipy.stats import spearmanr

    rho_p = float(spearmanr(df["patch_images"], df["abs_gnc"]).pvalue)
    p_text = "p < 0.001" if rho_p < 0.001 else f"p = {rho_p:.3f}"
    y_cap = 4.2
    clipped = df[df["abs_gnc"] > y_cap].copy()
    visible = df[df["abs_gnc"] <= y_cap].copy()
    fig = plt.figure(figsize=(10.8, 6.4), constrained_layout=False)
    ax = fig.add_axes([0.09, 0.16, 0.78, 0.68])
    add_header(fig, "Patch-image coverage and raw GNC volatility", "Each point is one port-month; lines summarize binned median and 75th percentile |GNC|.")
    style_axis(ax, "both")
    ax.scatter(visible["patch_images"], visible["abs_gnc"], s=13, color=BLUE, alpha=0.22, edgecolors="none", label="Port-month")
    if not clipped.empty:
        ax.scatter(
            clipped["patch_images"],
            np.full(len(clipped), y_cap),
            s=24,
            marker="^",
            color=PINK_DARK,
            alpha=0.78,
            edgecolors=INK,
            linewidth=0.25,
            label=f">{y_cap:.1f} |GNC| cap",
            zorder=4,
        )
    ax.plot(summary["x"], summary["median_abs_gnc"], color=BLUE_DARK, linewidth=2.1, marker="o", markersize=4.5, label="Median |GNC|")
    ax.plot(summary["x"], summary["q75_abs_gnc"], color=GOLD_DARK, linewidth=2.0, marker="o", markersize=4.5, label="75th percentile |GNC|")
    ax.set_xscale("log")
    ax.set_xlabel("Patch images per port-month (log scale)")
    ax.set_ylabel("Raw monthly |GNC|")
    ax.set_ylim(0, y_cap * 1.05)
    ax.legend(loc="upper right", fontsize=8.6)
    ax.text(
        0.02,
        0.94,
        f"Spearman rho = {rho:.2f} ({p_text})\nN = {len(df):,}; capped points = {len(clipped)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=BG, edgecolor=GRID, linewidth=0.8),
    )
    add_source(fig, f"{SOURCE_SHORT} Each point is one valid port-month with reconstructed patch-image counts.")
    save(fig, "container_index_patch_count_vs_abs_gnc.png")


def plot_hai_phong_sensitivity(port: pd.DataFrame) -> None:
    g = port[port["port_id"] == "hai_phong"].sort_values("date").copy()
    g["GNC_ma3"] = g["GNC"].rolling(3, min_periods=1).mean()
    g["patch_norm_gnc_ma3"] = g["patch_norm_gnc"].rolling(3, min_periods=1).mean()
    g["divergence"] = (g["GNC"] - g["patch_norm_gnc"]).abs()
    low_cut = g["patch_images"].quantile(0.30)
    div_cut = g["divergence"].quantile(0.72)
    g["flag"] = (g["patch_images"] <= low_cut) & (g["divergence"] >= div_cut)
    if g["flag"].sum() < 3:
        idx = g[g["patch_images"] <= low_cut].nlargest(4, "divergence").index
        g.loc[idx, "flag"] = True

    fig = plt.figure(figsize=(11.0, 7.5), constrained_layout=False)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.1, 2.1, 1.0, 1.0], left=0.09, right=0.96, top=0.84, bottom=0.13, hspace=0.13)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    add_header(fig, "Hai Phong low-coverage GNC sensitivity", "Flagged months combine low patch-image coverage with high raw-vs-normalized divergence.")
    for ax in axes:
        style_axis(ax, "y")
    flagged = g[g["flag"]]
    axes[0].plot(g["date"], g["GNC"], color=BLUE_LIGHT, linewidth=0.95, marker="o", markersize=2.0, alpha=0.80, label="Raw GNC")
    axes[0].plot(g["date"], g["GNC_ma3"], color=BLUE_DARK, linewidth=1.7, label="3m mean")
    axes[0].scatter(flagged["date"], flagged["GNC"], s=34, color=PINK_DARK, edgecolor=INK, linewidth=0.35, zorder=4)
    axes[0].axhline(0, color=AXIS, linewidth=0.75, linestyle=":")
    axes[0].set_ylabel("Raw GNC")
    axes[1].plot(g["date"], g["patch_norm_gnc"], color=GREEN, linewidth=0.95, marker="o", markersize=2.0, alpha=0.80, label="Patch-normalized GNC")
    axes[1].plot(g["date"], g["patch_norm_gnc_ma3"], color=GREEN_DARK, linewidth=1.7, label="3m mean")
    axes[1].scatter(flagged["date"], flagged["patch_norm_gnc"], s=34, color=PINK_DARK, edgecolor=INK, linewidth=0.35, zorder=4)
    axes[1].axhline(0, color=AXIS, linewidth=0.75, linestyle=":")
    axes[1].set_ylabel("Patch-norm. GNC")
    axes[2].bar(g["date"], g["patch_images"], width=22, color=NEUTRAL, alpha=0.60)
    axes[2].scatter(flagged["date"], flagged["patch_images"], s=28, color=PINK_DARK, zorder=4)
    axes[2].set_ylabel("Patch imgs")
    axes[3].bar(g["date"], g["n_observations"], width=22, color=NEUTRAL, alpha=0.48)
    axes[3].scatter(flagged["date"], flagged["n_observations"], s=28, color=PINK_DARK, zorder=4)
    axes[3].set_ylabel("Obs.")
    axes[3].set_xlabel("Month")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in axes[2:]:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
    format_year_axis(axes[3])
    flag_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=PINK_DARK,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=6.0,
        label="Flagged low-coverage divergence month",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles + [flag_handle], loc="upper left", ncols=3, fontsize=8.1)
    axes[1].legend(loc="upper left", ncols=2, fontsize=8.3)
    add_source(fig, f"{SOURCE_SHORT} Pink markers flag low-coverage divergence months for Hai Phong.")
    save(fig, "container_index_bad_example_hai_phong_patch_sensitivity.png")


def copy_mad_treasury_plots() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in MAD_COPY_FILES:
        src = TREASURY_BUNDLE / name
        dst = OUT_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        print(dst)


def plot_mad_sharpe_by_method_horizon() -> None:
    df = git_csv(f"{MAD_SUITE}/mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv")
    df = df[df["model_id"].isin(METHOD_ORDER)].copy()
    metric = "annualized_sharpe_treasury_excess"
    idx = df.groupby(["model_id", "forecast_horizon_months"])[metric].idxmax()
    best = df.loc[idx].copy().sort_values(["model_id", "forecast_horizon_months"])
    equal_weight = float(df.sort_values(metric, ascending=False).iloc[0]["equal_weight_annualized_sharpe_treasury_excess"])

    fig = plt.figure(figsize=(9.4, 5.8), constrained_layout=False)
    ax = fig.add_axes([0.11, 0.16, 0.78, 0.66])
    add_header(
        fig,
        "Sharpe by method and forecast horizon",
        "Each point is the best signal/lookback configuration for that method and horizon.",
        x=0.08,
    )
    style_axis(ax, "y")
    for method in METHOD_ORDER:
        label, color = METHOD_META[method]
        g = best[best["model_id"] == method].sort_values("forecast_horizon_months")
        ax.plot(
            g["forecast_horizon_months"],
            g[metric],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.2,
            label=label,
        )

    ax.axhline(equal_weight, color=NEUTRAL, linewidth=1.1, linestyle=(0, (3, 3)))
    ax.text(1.02, equal_weight + 0.015, "equal weight", ha="left", va="bottom", fontsize=8.4, color=MUTED)
    ax.set_xlim(0.9, 5.48)
    ax.set_ylim(0.35, 1.06)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Forecast horizon h, months")
    ax.set_ylabel("Treasury-adjusted Sharpe")
    ax.legend(loc="lower right", ncols=2, fontsize=8.4, handlelength=2.2)

    endpoints = best[best["forecast_horizon_months"] == 5].copy()
    endpoints["offset"] = endpoints["model_id"].map(
        {
            "elastic_net_panel": 0.018,
            "random_forest_panel": -0.048,
            "ols_predictive": -0.026,
            "distributed_lag": -0.004,
        }
    )
    for row in endpoints.itertuples(index=False):
        label, color = METHOD_META[row.model_id]
        ax.text(
            row.forecast_horizon_months + 0.035,
            getattr(row, metric) + row.offset,
            f"{getattr(row, metric):.3f}",
            ha="left",
            va="center",
            fontsize=8.2,
            weight="bold",
            color=color,
            bbox=dict(facecolor=BG, edgecolor="none", pad=0.4, alpha=0.82),
        )

    add_source(fig, "Source: bd5d48e usd19_no_sri_lanka_28623539 mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv; GNC-informed models.")
    save(fig, "gnc_mad_sharpe_by_method_and_horizon.png", pad=0.10)


def write_manifest() -> None:
    source_root = f"regenerated from locked bd5d48e:{FINAL_RUN_ROOT}"
    port_source = f"{source_root}/daily_container_index/port_timeseries.csv"
    country_source = f"{source_root}/daily_container_index/country_gnc.csv"
    mad_source = f"{source_root}/return_forecasting/usd19_no_sri_lanka_28623539/mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv"
    rows = [
        ("4.19", "container_index_coverage_heatmap.png", port_source),
        ("4.20", "container_index_patch_image_coverage_heatmap.png", port_source),
        ("4.21", "container_index_port_shanghai_nc.png", port_source),
        ("4.22", "container_index_port_shanghai_total_vs_patch_normalized.png", port_source),
        ("4.23", "container_index_country_kina_gnc.png", country_source),
        ("4.24", "container_index_country_gnc_panel_selected.png", country_source),
        ("4.25", "container_index_china_port_gnc_small_multiples.png", f"{port_source} and {country_source}"),
        ("4.26", "container_index_country_kina_raw_vs_patch_normalized_gnc.png", f"{port_source} and {country_source}"),
        ("4.27", "container_index_patch_count_vs_abs_gnc.png", port_source),
        ("4.28", "container_index_bad_example_hai_phong_patch_sensitivity.png", port_source),
        ("4.29", "gnc_mad_top15_sharpe.png", mad_source),
        ("4.30", "gnc_mad_best_sharpe_by_forecast_method.png", mad_source),
        ("4.31", "gnc_mad_method_performance_dashboard.png", mad_source),
        ("4.32", "gnc_mad_sharpe_by_method_and_horizon.png", mad_source),
    ]
    manifest = pd.DataFrame(rows, columns=["figure", "filename", "source"])
    manifest.to_csv(OUT_DIR / "MANIFEST.csv", index=False)


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    port, country, _pred, _patch = load_data()
    plot_coverage_heatmap(port)
    plot_patch_image_coverage_heatmap(port)
    plot_shanghai_nc(port)
    plot_shanghai_total_vs_patch(port)
    plot_china_country_gnc(country)
    plot_country_panel(country)
    plot_china_port_small_multiples(port, country)
    plot_china_raw_vs_patch_norm(port, country)
    plot_patch_count_vs_abs_gnc(port)
    plot_hai_phong_sensitivity(port)
    # copy_mad_treasury_plots() udeladt: top15/best-by-method/dashboard regenereres
    # separat fra den nye MAD-CSV (Phase B) i stedet for at kopiere gamle PNG'er.
    plot_mad_sharpe_by_method_horizon()
    write_manifest()


if __name__ == "__main__":
    main()
