from __future__ import annotations

import io
import shutil
import subprocess
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
OUT_DIRS = [AUDIT / "mad_tau04_bundle"]

REPO = AUDIT.parent / "Bachelor-portfolio"
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
SUITE = "final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539"
BEST = dict(model_id="distributed_lag", forecast_horizon_months=1, signal_variant="raw", lookback_months=5)

BG = "#fafaf7"
PANEL = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
BLUE = "#2b9fd6"
BLUE_DARK = "#0b4775"
BLUE_LIGHT = "#9bd2ee"
GOLD = "#edae13"
GOLD_DARK = "#a86700"
PINK = "#d889b8"
PINK_DARK = "#9b3e5d"
NEUTRAL = "#66717c"

METHOD_LABEL = "Distributed lag, h=1, raw signal, L=5"
SUB_SCOPE = "ROI tau=0.4 buffer=10; monthly horizon-normalized Treasury-excess log returns."
FONT = ["DejaVu Sans"]

COUNTRY_MAP = {
    "Belgien": "Belgium",
    "Brasilien": "Brazil",
    "Filippinerne": "Philippines",
    "Gr\\u00e6kenland".encode().decode("unicode_escape"): "Greece",
    "Holland": "Netherlands",
    "Indien": "India",
    "Kina": "China",
    "Sydkorea": "South Korea",
    "Tyskland": "Germany",
    "Spanien": "Spain",
}


def git_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO / rel)


def git_lfs_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO / rel)


def filter_best(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for key, value in BEST.items():
        mask &= df[key] == value
    return df[mask].copy()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    returns = filter_best(git_csv(f"{SUITE}/mad_portfolio_cleaned/portfolio_returns.csv"))
    weights = filter_best(git_lfs_csv(f"{SUITE}/mad_portfolio_cleaned/weights.csv"))
    weights = weights[["date", "country", "weight", "signal"]].copy()
    pivot = weights.pivot_table(index="country", columns="date", values="weight", aggfunc="sum").fillna(0.0)
    trades = pivot.diff(axis=1).iloc[:, 1:].stack().reset_index()
    trades.columns = ["country", "date", "trade"]
    headline = git_csv(f"{SUITE}/mad_portfolio_cleaned/top20_treasury_adjusted.csv").iloc[0]
    rf = git_csv("final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/inputs/treasury_1mo_monthly_rf_used.csv")

    returns["date"] = pd.to_datetime(returns["date"])
    returns["period"] = pd.PeriodIndex(returns["date"], freq="M")
    weights["date"] = pd.to_datetime(weights["date"])
    trades["date"] = pd.to_datetime(trades["date"])

    rf_map = dict(zip(pd.PeriodIndex(rf["month"], freq="M"), rf["treasury_1mo_monthly_log_return"]))

    def period_rf(row: pd.Series) -> float:
        months = pd.period_range(row["period"], periods=int(row["forecast_horizon_months"]), freq="M")
        values = [rf_map.get(m, np.nan) for m in months]
        return float(np.sum(values)) if not any(pd.isna(values)) else np.nan

    returns = returns.sort_values("date").copy()
    horizon = int(returns["forecast_horizon_months"].iloc[0])
    returns["rf_period"] = returns.apply(period_rf, axis=1)
    returns["portfolio_monthly_treasury_excess"] = (
        returns["portfolio_return"] - returns["rf_period"]
    ) / horizon
    returns["equal_weight_monthly_treasury_excess"] = (
        returns["equal_weight_return"] - returns["rf_period"]
    ) / horizon
    returns["relative_monthly_excess"] = returns["excess_return"] / horizon
    returns["mad_wealth"] = np.exp(returns["portfolio_monthly_treasury_excess"].cumsum())
    returns["equal_weight_wealth"] = np.exp(returns["equal_weight_monthly_treasury_excess"].cumsum())
    returns["mad_drawdown"] = returns["mad_wealth"] / returns["mad_wealth"].cummax() - 1.0
    returns["equal_weight_drawdown"] = (
        returns["equal_weight_wealth"] / returns["equal_weight_wealth"].cummax() - 1.0
    )
    returns["relative_rolling_6"] = returns["relative_monthly_excess"].rolling(6, min_periods=2).mean()

    weights["country_label"] = weights["country"].map(COUNTRY_MAP).fillna(weights["country"])
    trades["country_label"] = trades["country"].map(COUNTRY_MAP).fillna(trades["country"])
    return returns, weights, trades, headline


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "legend.frameon": False,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.5, color=AXIS, pad=5)


def add_header(fig: plt.Figure, title: str, subtitle: str, y: float = 0.955, x: float = 0.07) -> None:
    fig.text(x, y, title, ha="left", va="top", fontsize=18, weight="bold", color=INK)
    fig.text(x, y - 0.035, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)


def add_source(fig: plt.Figure, text: str, x: float = 0.07, y: float = 0.045) -> None:
    fig.text(x, y, text, ha="left", va="bottom", fontsize=8.8, color=MUTED)


def format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))


def save_all(fig: plt.Figure, name: str, pad_inches: float = 0.24) -> None:
    primary = OUT_DIRS[0] / name
    fig.savefig(primary, dpi=150, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    for out_dir in OUT_DIRS:
        target = out_dir / name
        if target != primary:
            shutil.copy2(primary, target)


def plot_wealth(returns: pd.DataFrame, headline: pd.Series) -> None:
    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.2, 1.18], left=0.08, right=0.94, top=0.82, bottom=0.15, hspace=0.16
    )
    ax = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax)
    add_header(fig, "Best GNC-MAD wealth path after Treasury adjustment", f"{METHOD_LABEL}. {SUB_SCOPE}")
    style_axis(ax, "y")
    style_axis(ax_dd, "y")

    ax.plot(returns["date"], returns["mad_wealth"], color=BLUE, linewidth=2.6, label="GNC-informed MAD")
    ax.plot(
        returns["date"],
        returns["equal_weight_wealth"],
        color=NEUTRAL,
        linewidth=2.2,
        linestyle="--",
        label="Equal weight",
    )
    ax.fill_between(
        returns["date"],
        returns["mad_wealth"],
        returns["equal_weight_wealth"],
        where=returns["mad_wealth"] >= returns["equal_weight_wealth"],
        color=BLUE_LIGHT,
        alpha=0.20,
        interpolate=True,
    )
    ax.set_ylabel("Treasury-excess wealth")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.03), handlelength=2.8)

    last = returns.iloc[-1]
    ax.scatter([last["date"]], [last["mad_wealth"]], s=42, color=BLUE, zorder=5, edgecolor=INK, linewidth=0.5)
    ax.scatter(
        [last["date"]],
        [last["equal_weight_wealth"]],
        s=42,
        color=NEUTRAL,
        zorder=5,
        edgecolor=INK,
        linewidth=0.5,
    )
    ax.annotate(
        f"MAD {last['mad_wealth']:.2f}x",
        xy=(last["date"], last["mad_wealth"]),
        xytext=(12, 7),
        textcoords="offset points",
        color=BLUE_DARK,
        fontsize=10,
        weight="bold",
    )
    ax.annotate(
        f"EW {last['equal_weight_wealth']:.2f}x",
        xy=(last["date"], last["equal_weight_wealth"]),
        xytext=(12, -14),
        textcoords="offset points",
        color=NEUTRAL,
        fontsize=10,
        weight="bold",
    )
    callout = (
        f"Sharpe {headline.annualized_sharpe_treasury_excess:.3f} vs "
        f"{headline.equal_weight_annualized_sharpe_treasury_excess:.3f}\n"
        f"Max DD {100 * headline.max_drawdown_excess_treasury:.1f}% vs "
        f"{100 * headline.equal_weight_max_drawdown_excess_treasury:.1f}%"
    )
    ax.text(
        0.02,
        0.08,
        callout,
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=GRID, linewidth=0.8),
    )

    ax_dd.fill_between(returns["date"], returns["mad_drawdown"], 0, color=BLUE, alpha=0.22)
    ax_dd.fill_between(returns["date"], returns["equal_weight_drawdown"], 0, color=NEUTRAL, alpha=0.18)
    ax_dd.plot(returns["date"], returns["mad_drawdown"], color=BLUE, linewidth=1.7)
    ax_dd.plot(returns["date"], returns["equal_weight_drawdown"], color=NEUTRAL, linewidth=1.5, linestyle="--")
    ax_dd.axhline(0, color=AXIS, linewidth=0.9)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    format_date_axis(ax_dd)
    plt.setp(ax.get_xticklabels(), visible=False)
    add_source(
        fig,
        "Source: bd5d48e usd19_no_sri_lanka_28623539 (tau=0.4, ROI10); Treasury adjustment sums 1M Treasury log returns over each forecast horizon.",
    )
    save_all(fig, "gnc_mad_best_wealth_vs_equal_weight.png")


def plot_monthly_excess(returns: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.8, 6.25), constrained_layout=False)
    ax = fig.add_axes([0.08, 0.25, 0.86, 0.55])
    add_header(
        fig,
        "Month-by-month excess return against equal weight",
        f"{METHOD_LABEL}. Bars show MAD Treasury-excess return minus equal-weight Treasury-excess return.",
    )
    style_axis(ax, "y")
    positive = returns["relative_monthly_excess"] >= 0
    ax.bar(
        returns.loc[positive, "date"],
        returns.loc[positive, "relative_monthly_excess"],
        width=23,
        color=BLUE,
        alpha=0.82,
        edgecolor=BLUE_DARK,
        linewidth=0.35,
        label="MAD outperforms",
    )
    ax.bar(
        returns.loc[~positive, "date"],
        returns.loc[~positive, "relative_monthly_excess"],
        width=23,
        color=PINK,
        alpha=0.78,
        edgecolor=PINK_DARK,
        linewidth=0.35,
        label="MAD underperforms",
    )
    ax.plot(returns["date"], returns["relative_rolling_6"], color=INK, linewidth=2.0, label="6-month mean")
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.axhline(
        returns["relative_monthly_excess"].mean(),
        color=GOLD_DARK,
        linestyle=":",
        linewidth=1.8,
        label="Mean",
    )
    ax.set_ylabel("Monthly difference")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    lower = min(-0.016, returns["relative_monthly_excess"].min() * 1.28)
    upper = max(0.036, returns["relative_monthly_excess"].max() * 1.18)
    ax.set_ylim(lower, upper)
    format_date_axis(ax)
    ax.set_xlim(returns["date"].min() - pd.Timedelta(days=25), returns["date"].max() + pd.Timedelta(days=20))

    best = returns.loc[returns["relative_monthly_excess"].idxmax()]
    worst = returns.loc[returns["relative_monthly_excess"].idxmin()]
    ax.scatter([best["date"]], [best["relative_monthly_excess"]], color=INK, s=24, zorder=5)
    ax.annotate(
        f"{best['date'].strftime('%Y-%m')} {100 * best['relative_monthly_excess']:+.1f} pp",
        xy=(best["date"], best["relative_monthly_excess"]),
        xytext=(10, -3),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        va="top",
        ha="left",
    )

    summary = (
        f"Hit rate {100 * (returns['relative_monthly_excess'] > 0).mean():.0f}%\n"
        f"Mean {100 * returns['relative_monthly_excess'].mean():+.2f} pp/mo\n"
        f"Worst {worst['date'].strftime('%Y-%m')} {100 * worst['relative_monthly_excess']:.1f} pp"
    )
    fig.text(
        0.085,
        0.135,
        summary,
        fontsize=9.2,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=GRID, linewidth=0.8),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04), ncols=4, fontsize=9.2, handlelength=1.8, columnspacing=1.3)
    add_source(
        fig,
        "Source: monthly horizon-normalized log-return difference; positive values mean the MAD portfolio beats equal weight.",
    )
    save_all(fig, "gnc_mad_best_monthly_excess_return.png")


def weight_summary(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = weights.pivot_table(index="country_label", columns="date", values="weight", aggfunc="sum").fillna(0.0)
    summary = (
        weights.groupby("country_label")
        .agg(
            average_weight=("weight", "mean"),
            max_weight=("weight", "max"),
            active_months=("weight", lambda s: int((s > 1e-9).sum())),
            cap_hits=("weight", lambda s: int((s >= 0.349999).sum())),
            mean_signal=("signal", "mean"),
        )
        .sort_values("average_weight", ascending=False)
    )
    return pivot.loc[summary.index], summary


def plot_weight_heatmap(returns: pd.DataFrame, weights: pd.DataFrame) -> None:
    pivot, summary = weight_summary(weights)
    cap_pivot = pivot >= 0.349999
    months = pivot.columns
    cmap = LinearSegmentedColormap.from_list("weight_blue", [BG, "#d7eef8", BLUE_LIGHT, BLUE, BLUE_DARK])

    fig = plt.figure(figsize=(10.8, 7.8), constrained_layout=False)
    ax = fig.add_axes([0.13, 0.20, 0.76, 0.61])
    cax = fig.add_axes([0.92, 0.29, 0.018, 0.38])
    add_header(
        fig,
        "Country weights in the best Treasury-adjusted MAD portfolio",
        f"{METHOD_LABEL}. Rows are ordered by average portfolio weight; circles mark months at the 35% weight cap.",
    )
    style_axis(ax, None)
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=0.35, interpolation="nearest")
    ax.set_xticks(np.arange(len(months)), minor=True)
    ax.set_yticks(np.arange(len(pivot.index)), minor=True)
    ax.grid(which="minor", color="#eef2f5", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    xticks = np.arange(0, len(months), 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([pd.Timestamp(months[i]).strftime("%Y-%m") for i in xticks], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ys, xs = np.where(cap_pivot.values)
    ax.scatter(xs, ys, s=25, facecolor=GOLD, edgecolor=INK, linewidth=0.45, alpha=0.95)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Portfolio weight", color=MUTED, fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    fig.text(
        0.13,
        0.078,
        f"Cap hits: {int(summary.cap_hits.sum())} country-months; mean active assets: {returns.n_assets.mean():.0f}.",
        fontsize=8.8,
        color=MUTED,
    )
    cap_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=GOLD,
        markeredgecolor=INK,
        markersize=6.2,
        label="35% cap hit",
    )
    ax.legend(handles=[cap_handle], loc="upper right", bbox_to_anchor=(1.0, 1.075), fontsize=9)
    add_source(fig, "Source: saved monthly country weights for the best ROI10 GNC-informed MAD configuration.")
    save_all(fig, "gnc_mad_best_weight_heatmap.png")


def plot_trade_heatmap(returns: pd.DataFrame, trades: pd.DataFrame) -> None:
    pivot = trades.pivot_table(index="country_label", columns="date", values="trade", aggfunc="sum").fillna(0.0)
    order = pivot.abs().mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    turn = returns.dropna(subset=["turnover"]).set_index("date")["turnover"]
    months = pivot.columns
    turn = turn.reindex(months)
    vmax = max(0.35, float(np.nanmax(np.abs(pivot.values))))
    cmap = LinearSegmentedColormap.from_list("sell_buy", [PINK_DARK, PINK, BG, BLUE_LIGHT, BLUE_DARK])

    fig = plt.figure(figsize=(10.8, 7.35), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 5.0], left=0.13, right=0.895, top=0.875, bottom=0.15, hspace=0.07
    )
    ax_top = fig.add_subplot(gs[0])
    ax_hm = fig.add_subplot(gs[1], sharex=ax_top)
    cax = fig.add_axes([0.925, 0.245, 0.018, 0.44])
    add_header(
        fig,
        "Trading intensity behind the best MAD allocation",
        f"{METHOD_LABEL}. Heatmap cells are monthly changes in country weights; top bars show total one-way turnover.",
    )
    style_axis(ax_top, "y")
    x = np.arange(len(months))
    bar_colors = np.where(turn.values >= turn.mean(), GOLD_DARK, GOLD)
    ax_top.bar(x, turn.values, width=0.86, color=bar_colors, edgecolor=GOLD_DARK, linewidth=0.35, alpha=0.82)
    ax_top.axhline(turn.mean(), color=INK, linestyle=":", linewidth=1.5)
    ax_top.set_xlim(-0.5, len(months) - 0.5)
    ax_top.set_ylim(0, max(1.0, float(turn.max()) * 1.12))
    ax_top.set_ylabel("Turnover")
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax_top.tick_params(labelbottom=False)
    max_turn_idx = int(np.nanargmax(turn.values))
    ax_top.annotate(
        f"max {100 * turn.max():.0f}%",
        xy=(max_turn_idx, turn.max()),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8.4,
        color=INK,
    )

    style_axis(ax_hm, None)
    image = ax_hm.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_hm.set_xticks(np.arange(len(months)), minor=True)
    ax_hm.set_yticks(np.arange(len(order)), minor=True)
    ax_hm.grid(which="minor", color="#eef2f5", linewidth=0.45)
    ax_hm.tick_params(which="minor", bottom=False, left=False)
    xticks = np.arange(0, len(months), 6)
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels([pd.Timestamp(months[i]).strftime("%Y-%m") for i in xticks], rotation=35, ha="right", fontsize=8.5)
    ax_hm.set_yticks(np.arange(len(order)))
    ax_hm.set_yticklabels(order, fontsize=9)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_label("Change in weight", color=MUTED, fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:+.0f}%"))
    fig.text(0.92, 0.70, "buy", fontsize=8.5, color=BLUE_DARK, ha="left")
    fig.text(0.92, 0.19, "sell", fontsize=8.5, color=PINK_DARK, ha="left")
    add_source(
        fig,
        "Source: first differences of saved monthly weights. Positive cells increase exposure; negative cells reduce exposure.",
    )
    save_all(fig, "gnc_mad_best_trade_heatmap.png")


def plot_average_weights(returns: pd.DataFrame, weights: pd.DataFrame) -> None:
    _, summary = weight_summary(weights)
    avg = summary.sort_values("average_weight", ascending=True).copy()

    fig = plt.figure(figsize=(13.6, 7.1), constrained_layout=False)
    ax = fig.add_axes([0.105, 0.15, 0.885, 0.665])
    add_header(
        fig,
        "Average country allocation of the best MAD configuration",
        f"{METHOD_LABEL}. Bars show average weight; dots show maximum weight.",
    )
    style_axis(ax, "x")
    y = np.arange(len(avg))
    threshold = avg["average_weight"].quantile(0.75)
    colors = [BLUE if value >= threshold else BLUE_LIGHT for value in avg["average_weight"]]
    ax.barh(y, avg["average_weight"], color=colors, edgecolor=BLUE_DARK, linewidth=0.4, alpha=0.9, label="Average weight")
    ax.scatter(avg["max_weight"], y, s=34, color=GOLD, edgecolor=GOLD_DARK, linewidth=0.6, zorder=4, label="Maximum weight")
    equal_weight_level = 1.0 / returns["n_assets"].mean()
    ax.axvline(equal_weight_level, color=NEUTRAL, linestyle="--", linewidth=1.7, label="Mean equal-weight level")
    ax.axvline(0.35, color=INK, linestyle=":", linewidth=1.5, label="35% cap")
    ax.set_yticks(y)
    ax.set_yticklabels(avg.index, fontsize=9.3)
    ax.set_xlabel("Portfolio weight")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.set_xlim(0, 0.378)
    ax.margins(x=0)

    top_cutoff = avg["average_weight"].nlargest(5).min()
    for yi, (_, row) in enumerate(avg.iterrows()):
        if row["average_weight"] >= top_cutoff or row["cap_hits"] > 3:
            ax.text(row["average_weight"] + 0.006, yi, f"{100 * row['average_weight']:.1f}%", va="center", ha="left", fontsize=8.2, color=INK)
        if row["cap_hits"] > 0:
            ax.text(0.346, yi, f"{int(row['cap_hits'])} cap", va="center", ha="right", fontsize=7.8, color=MUTED)

    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.03), ncols=2, fontsize=8.7, handlelength=1.8, columnspacing=1.2)
    ax.text(
        0.0,
        -0.13,
        f"Average active assets: {returns.n_assets.mean():.0f}; mean turnover: {returns.turnover.mean():.2f}; cap-hit labels count country-months at the 35% bound.",
        transform=ax.transAxes,
        fontsize=8.8,
        color=MUTED,
    )
    add_source(fig, "Source: saved country weights for the best ROI10 Treasury-adjusted Sharpe configuration.")
    save_all(fig, "gnc_mad_best_average_weights.png", pad_inches=0.07)


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    returns, weights, trades, headline = load_data()
    plot_wealth(returns, headline)
    plot_monthly_excess(returns)
    plot_weight_heatmap(returns, weights)
    plot_trade_heatmap(returns, trades)
    plot_average_weights(returns, weights)
    for name in [
        "gnc_mad_best_wealth_vs_equal_weight.png",
        "gnc_mad_best_monthly_excess_return.png",
        "gnc_mad_best_weight_heatmap.png",
        "gnc_mad_best_trade_heatmap.png",
        "gnc_mad_best_average_weights.png",
    ]:
        print(OUT_DIRS[0] / name)


if __name__ == "__main__":
    main()
