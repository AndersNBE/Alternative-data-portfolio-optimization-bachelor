"""Phase B1: MAD-oversigtsfigurer fra bd5d48e USD19-suiten (tau=0.4).

Genererer: top15, best-by-method, dashboard, heatmap, distribution.
Kilde: bd5d48e:final_runs/tau04_hk_28623539/return_forecasting/
       usd19_no_sri_lanka_28623539/mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
MAD_CSV = (
    "final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/"
    "mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv"
)
OUT_DIR = AUDIT / "mad_tau04_bundle"
SOURCE = "Source: bd5d48e usd19_no_sri_lanka_28623539 portfolio_metrics_treasury_adjusted.csv (tau=0.4, ROI10)."

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
NEUTRAL = "#66717c"
METHOD_META = {
    "elastic_net_panel": ("Elastic net panel", "#2b9fd6"),
    "random_forest_panel": ("Random forest panel", "#00a878"),
    "ols_predictive": ("OLS predictive", "#edae13"),
    "distributed_lag": ("Distributed lag", "#d889b8"),
}
GNC_MODELS = list(METHOD_META)
VARIANT_SHORT = {"raw": "raw", "direction": "dir", "direction_risk_scaled": "DRS"}
METRIC = "annualized_sharpe_treasury_excess"
EW_METRIC = "equal_weight_annualized_sharpe_treasury_excess"


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.5, color=AXIS, pad=5)


def header(fig: plt.Figure, title: str, subtitle: str, x: float = 0.08) -> None:
    fig.text(x, 0.955, title, ha="left", va="top", fontsize=15, weight="bold", color=INK)
    fig.text(x, 0.905, subtitle, ha="left", va="top", fontsize=8.8, color=MUTED)


def source(fig: plt.Figure, x: float = 0.08) -> None:
    fig.text(x, 0.022, SOURCE, ha="left", va="bottom", fontsize=7.8, color=MUTED)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(path)


def load_grid() -> pd.DataFrame:
    df = pd.read_csv(REPO / MAD_CSV)
    return df[df["model_id"].isin(GNC_MODELS)].copy()


def config_label(row: pd.Series) -> str:
    name, _ = METHOD_META[row["model_id"]]
    return f"{name} · {VARIANT_SHORT[row['signal_variant']]} · L{int(row['lookback_months'])} · h{int(row['forecast_horizon_months'])}"


def plot_top15(grid: pd.DataFrame) -> None:
    top = grid.nlargest(15, METRIC).iloc[::-1]
    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_axes([0.34, 0.07, 0.62, 0.78])
    header(fig, "Top Treasury-adjusted GNC-informed MAD configurations",
           "Bars: the 15 highest Treasury-adjusted Sharpe ratios. Grey diamonds: matched equal-weight benchmark.")
    style_axis(ax, "x")
    y = np.arange(len(top))
    colors = [METHOD_META[m][1] for m in top["model_id"]]
    ax.barh(y, top[METRIC], color=colors, alpha=0.88, height=0.62)
    ax.scatter(top[EW_METRIC], y, marker="D", s=34, color=NEUTRAL, edgecolor="white", linewidth=0.5, zorder=5)
    for yy, (_, row) in zip(y, top.iterrows()):
        ax.text(row[METRIC] + 0.008, yy, f"{row[METRIC]:.3f}", va="center", ha="left", fontsize=8.2, weight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([config_label(r) for _, r in top.iterrows()], fontsize=8.4)
    ax.set_xlim(0, top[METRIC].max() * 1.12)
    ax.set_xlabel("Treasury-adjusted annualized Sharpe")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88) for _, c in METHOD_META.values()]
    handles.append(plt.Line2D([0], [0], marker="D", linestyle="None", color=NEUTRAL, markersize=6, label="Equal weight"))
    ax.legend(handles, [n for n, _ in METHOD_META.values()] + ["Equal weight"],
              loc="upper center", bbox_to_anchor=(0.5, -0.085), ncols=3, fontsize=8.0)
    source(fig)
    save(fig, "gnc_mad_top15_sharpe.png")


def plot_best_by_method(grid: pd.DataFrame) -> None:
    best = grid.loc[grid.groupby("model_id")[METRIC].idxmax()].sort_values(METRIC, ascending=False)
    fig = plt.figure(figsize=(8.8, 5.4))
    ax = fig.add_axes([0.10, 0.13, 0.86, 0.66])
    header(fig, "Best Treasury-adjusted Sharpe ratio by forecast method",
           "Each bar is the best configuration within one forecast-model family. Grey diamonds: matched equal weight.")
    style_axis(ax)
    x = np.arange(len(best))
    colors = [METHOD_META[m][1] for m in best["model_id"]]
    ax.bar(x, best[METRIC], color=colors, alpha=0.88, width=0.58)
    ax.scatter(x, best[EW_METRIC], marker="D", s=44, color=NEUTRAL, edgecolor="white", linewidth=0.6, zorder=5)
    for xx, (_, row) in zip(x, best.iterrows()):
        ax.text(xx, row[METRIC] + 0.012, f"{row[METRIC]:.3f}", ha="center", va="bottom", fontsize=9.0, weight="bold")
        ax.text(xx, row[METRIC] / 2,
                f"h={int(row['forecast_horizon_months'])}\n{VARIANT_SHORT[row['signal_variant']]}\nL={int(row['lookback_months'])}",
                ha="center", va="center", fontsize=8.6, color="white", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_META[m][0] for m in best["model_id"]], fontsize=9.2)
    ax.set_ylim(0, best[METRIC].max() * 1.16)
    ax.set_ylabel("Treasury-adjusted annualized Sharpe")
    source(fig)
    save(fig, "gnc_mad_best_sharpe_by_forecast_method.png")


def plot_dashboard(grid: pd.DataFrame) -> None:
    best = grid.loc[grid.groupby("model_id")[METRIC].idxmax()].sort_values(METRIC, ascending=False)
    panels = [
        (METRIC, "Treasury-adjusted Sharpe", "{:.3f}"),
        ("compound_excess_return_treasury", "Compound Treasury-excess return", "{:.3f}"),
        ("max_drawdown_excess_treasury", "Treasury-excess max drawdown", "{:.3f}"),
        ("mean_turnover", "Mean turnover", "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.10, hspace=0.55, wspace=0.28)
    header(fig, "Best method configurations across portfolio metrics",
           "Each panel uses the best-Sharpe GNC-informed MAD configuration per forecast method.")
    x = np.arange(len(best))
    colors = [METHOD_META[m][1] for m in best["model_id"]]
    short = {"elastic_net_panel": "Elastic net", "random_forest_panel": "Random forest",
             "ols_predictive": "OLS", "distributed_lag": "Dist. lag"}
    for ax, (col, title, fmt) in zip(axes.flat, panels):
        style_axis(ax)
        vals = best[col].to_numpy()
        ax.bar(x, vals, color=colors, alpha=0.88, width=0.6)
        for xx, v in zip(x, vals):
            off = (abs(vals).max() * 0.05) * (1 if v >= 0 else -1)
            ax.text(xx, v + off, fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.2, weight="bold")
        ax.set_title(title, fontsize=9.6, loc="left", color=INK, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in best["model_id"]], fontsize=8.2)
        if (vals < 0).any():
            ax.set_ylim(vals.min() * 1.3, 0.02)
        else:
            ax.set_ylim(0, vals.max() * 1.30)
    source(fig)
    save(fig, "gnc_mad_method_performance_dashboard.png")


def plot_heatmap(grid: pd.DataFrame) -> None:
    variants = ["raw", "direction", "direction_risk_scaled"]
    rows = []
    labels = []
    for model in GNC_MODELS:
        for variant in variants:
            sub = grid[(grid["model_id"] == model) & (grid["signal_variant"] == variant)]
            vals = [sub[sub["forecast_horizon_months"] == h][METRIC].max() for h in range(1, 6)]
            rows.append(vals)
            labels.append(f"{METHOD_META[model][0]} · {VARIANT_SHORT[variant]}")
    matrix = np.array(rows)
    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_axes([0.30, 0.09, 0.58, 0.74])
    header(fig, "Sharpe heatmap across model, signal variant and horizon",
           "Each cell is the best Treasury-adjusted Sharpe over MAD lookbacks for that combination.", x=0.06)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"h={h}" for h in range(1, 6)], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.8,
                    color="white" if v > matrix.min() + 0.62 * (matrix.max() - matrix.min()) else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Treasury-adjusted Sharpe", fontsize=8.6)
    source(fig, x=0.06)
    save(fig, "gnc_mad_sharpe_heatmap_model_variant_horizon.png")


def plot_distribution(grid: pd.DataFrame) -> None:
    best = grid.loc[grid.groupby("model_id")[METRIC].idxmax()].sort_values(METRIC, ascending=False)
    order = best["model_id"].tolist()
    fig = plt.figure(figsize=(8.8, 5.2))
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.62])
    header(fig, "Distribution of Sharpe ratios by forecast method",
           "Point colour encodes MAD lookback L; black stars mark the best h/L choice within each method.", x=0.09)
    style_axis(ax)
    rng = np.random.default_rng(20260611)
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=5, vmax=18)
    for x, model in enumerate(order):
        part = grid[grid["model_id"] == model]
        q1, q3 = part[METRIC].quantile([0.25, 0.75])
        med = part[METRIC].median()
        lo, hi = part[METRIC].quantile([0.05, 0.95])
        _, color = METHOD_META[model]
        ax.add_patch(plt.Rectangle((x - 0.22, q1), 0.44, q3 - q1, facecolor=color,
                                   edgecolor=color, alpha=0.14, linewidth=0.9))
        ax.plot([x - 0.22, x + 0.22], [med, med], color=color, linewidth=1.25)
        ax.plot([x, x], [lo, hi], color=color, linewidth=0.9, alpha=0.9)
        for v in (lo, hi):
            ax.plot([x - 0.12, x + 0.12], [v, v], color=color, linewidth=0.9)
        jitter = np.clip(rng.normal(0, 0.055, len(part)), -0.14, 0.14)
        ax.scatter(x + jitter, part[METRIC], c=part["lookback_months"], cmap=cmap, norm=norm,
                   s=23, alpha=0.92, edgecolor="white", linewidth=0.22, zorder=3)
        row = best[best["model_id"] == model].iloc[0]
        ax.scatter(x, row[METRIC], marker="*", s=120, color=INK, edgecolor="white", linewidth=0.45, zorder=7)
        ax.annotate(f"{row[METRIC]:.3f}", xy=(x, row[METRIC]), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.4, weight="bold", color=INK,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=0.4, alpha=0.88), zorder=8)
        ax.annotate(f"h{int(row.forecast_horizon_months)} L{int(row.lookback_months)}", xy=(x, row[METRIC]),
                    xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.8, color=MUTED, zorder=8)
    ew = float(best.iloc[0][EW_METRIC])
    ax.axhline(ew, color=AXIS, linestyle=(0, (3, 3)), linewidth=0.9)
    ax.text(len(order) - 1.18, ew + 0.012, "equal weight (matched)", fontsize=7.8, color=MUTED)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([METHOD_META[m][0] for m in order], fontsize=9.2)
    ax.set_ylabel("Treasury-adjusted annualized Sharpe")
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("MAD lookback L, months", fontsize=8.4)
    source(fig, x=0.09)
    save(fig, "gnc_mad_sharpe_distribution_by_forecast_method.png")


def main() -> None:
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG, "legend.frameon": False,
    })
    grid = load_grid()
    print(f"GNC-grid: {len(grid)} konfigurationer")
    plot_top15(grid)
    plot_best_by_method(grid)
    plot_dashboard(grid)
    plot_heatmap(grid)
    plot_distribution(grid)


if __name__ == "__main__":
    main()
