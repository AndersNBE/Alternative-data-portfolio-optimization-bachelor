"""Wealth-kurver for alle 52 pocket-konfigurationer paa samme graf.

Viser at den hoejafkast/lav-vol-lomme vi udpegede ikke bare har gode
slut-tal, men at alle 52 kurver FOELGES TAET AD gennem hele backtesten —
dvs. lommen er en sammenhaengende, robust region og ikke 52 tilfaeldige spikes.

Treasury-excess wealth, samme horisont-normalisering som
gnc_mad_best_wealth_vs_equal_weight.png (sammenlignelig).
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "Bachelor-portfolio"
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
SUITE = "final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539"
OUT = AUDIT / "mad_tau04_bundle" / "gnc_mad_pocket_wealth_curves.png"

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
H_COLORS = {3: "#edae13", 4: "#d889b8", 5: "#2b9fd6"}
H_LABELS = {3: "h=3", 4: "h=4", 5: "h=5"}
X = "monthly_vol_treasury_excess"
Y = "mean_monthly_excess_return_treasury"
S = "annualized_sharpe_treasury_excess"
WIN_X = (0.030, 0.035)
WIN_Y = (0.006, 0.0085)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 9.0,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


def git_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO / rel)


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


# --- identificer de 52 pocket-configs ---
metrics = git_csv(f"{SUITE}/mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv")
gnc = metrics[metrics["model_id"].isin(["ols_predictive", "elastic_net_panel", "distributed_lag", "random_forest_panel"])]
pocket = gnc[gnc[X].between(*WIN_X) & gnc[Y].between(*WIN_Y)].copy()
best = pocket.loc[pocket[S].idxmax()]
KEYS = ["model_id", "forecast_horizon_months", "signal_variant", "lookback_months"]
pocket_keys = set(map(tuple, pocket[KEYS].values))

# --- afkast-serier + risk-free ---
returns = git_csv(f"{SUITE}/mad_portfolio_cleaned/portfolio_returns.csv")
returns["date"] = pd.to_datetime(returns["date"])
returns["period"] = returns["date"].dt.to_period("M")
rf = git_csv(f"{SUITE}/inputs/treasury_1mo_monthly_rf_used.csv")
rf_map = dict(zip(pd.PeriodIndex(rf["month"], freq="M"), rf["treasury_1mo_monthly_log_return"]))


def horizon_rf(period, h):
    months = pd.period_range(period, periods=int(h), freq="M")
    vals = [rf_map.get(m, np.nan) for m in months]
    return float(np.sum(vals)) if not any(pd.isna(vals)) else np.nan


def wealth_curve(df_cfg, col):
    h = int(df_cfg["forecast_horizon_months"].iloc[0])
    d = df_cfg.sort_values("date").copy()
    rf_period = d["period"].apply(lambda p: horizon_rf(p, h))
    monthly_excess = (d[col] - rf_period) / h
    return d["date"].values, np.exp(monthly_excess.cumsum().values)


fig = plt.figure(figsize=(11.6, 6.6), dpi=150, facecolor=BG)
ax = fig.add_axes([0.075, 0.115, 0.905, 0.70])
fig.text(0.04, 0.965, "All pocket configurations track together", ha="left", va="top",
         fontsize=15.5, weight="bold", color=INK)
fig.text(0.04, 0.908,
         "Treasury-excess wealth for each of the 52 high-return, low-volatility configurations (h=4–5, L=6–9). "
         "The tight bundle shows the pocket is a coherent region,\nnot a set of unrelated lucky runs. Same horizon-normalised "
         "convention as the best-configuration wealth plot.",
         ha="left", va="top", fontsize=8.4, color=MUTED)
style_axes(ax)

final_wealths = []
curves_by_h = {3: [], 4: [], 5: []}
ew_h5 = None
for key, df_cfg in returns.groupby(KEYS):
    if tuple(key) not in pocket_keys:
        continue
    h = int(key[1])
    dates, w = wealth_curve(df_cfg, "portfolio_return")
    ax.plot(dates, w, color=H_COLORS[h], linewidth=0.8, alpha=0.38, zorder=2)
    final_wealths.append(w[-1])
    curves_by_h[h].append(pd.Series(w, index=pd.to_datetime(dates)))
    if h == 5 and ew_h5 is None:
        _, ew_h5 = wealth_curve(df_cfg, "equal_weight_return")
        ew_h5_dates = dates

# median-kurve paa tvaers af alle pocket-configs (alignet paa dato)
all_curves = [s for lst in curves_by_h.values() for s in lst]
aligned = pd.concat(all_curves, axis=1)
median_curve = aligned.median(axis=1)
ax.plot(median_curve.index, median_curve.values, color="#111820", linewidth=2.6, zorder=5,
        label="Pocket median")

# equal weight (h=5) reference
ax.plot(ew_h5_dates, ew_h5, color=AXIS, linewidth=2.0, linestyle=(0, (4, 3)), zorder=4,
        label="Equal weight (h=5)")

# fremhaev den enkelte bedste config
best_cfg = returns
for k, v in zip(KEYS, best[KEYS].values):
    best_cfg = best_cfg[best_cfg[k] == v]
bd, bw = wealth_curve(best_cfg, "portfolio_return")
ax.plot(bd, bw, color="#b3001b", linewidth=1.8, zorder=6, label="Best Sharpe (one of the 52)")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Treasury-excess wealth (start = 1.0)", fontsize=9.4)
ax.set_xlabel("")
lo, hi = min(final_wealths), max(final_wealths)
ax.text(0.015, 0.965,
        f"52 configurations · all h=4–5, L=6–9\n"
        f"Final Treasury-excess wealth: {lo:.2f}× – {hi:.2f}×  (median {np.median(final_wealths):.2f}×)\n"
        f"Equal-weight (h=5) ends at {ew_h5[-1]:.2f}×",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=GRID, linewidth=0.8, alpha=0.95))

handles = [Line2D([0], [0], color=H_COLORS[h], linewidth=2.0, label=f"{H_LABELS[h]} configs ({len(curves_by_h[h])})")
           for h in (5, 4, 3)]
handles += [Line2D([0], [0], color="#111820", linewidth=2.6, label="Pocket median"),
            Line2D([0], [0], color="#b3001b", linewidth=1.8, label="Best Sharpe (one of the 52)"),
            Line2D([0], [0], color=AXIS, linewidth=2.0, linestyle=(0, (4, 3)), label="Equal weight (h=5)")]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.0, ncol=2, columnspacing=1.3)

fig.text(0.04, 0.022,
         "Source: bd5d48e usd19_no_sri_lanka_28623539 portfolio_returns.csv; horizon-normalised Treasury-excess monthly returns, tau=0.4, ROI10.",
         fontsize=6.9, color=MUTED)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=BG)
plt.close(fig)
print(OUT)
print(f"pocket configs plottet: {len(final_wealths)} | per horisont: "
      f"{{3:{len(curves_by_h[3])}, 4:{len(curves_by_h[4])}, 5:{len(curves_by_h[5])}}}")
print(f"final wealth: {lo:.3f}-{hi:.3f}x (median {np.median(final_wealths):.3f}) | EW h5 {ew_h5[-1]:.3f}")
