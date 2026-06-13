"""Phase B3: Risk-return scatter for den nye USD19-suite (tau=0.4, ROI10).

Design baseret paa dataanalyse af den nye fordeling:
  Hovedpanel: alle 840 GNC-konfigurationer, farve = forecast-metode.
  a) Vinder-armen: h=1, korte lookbacks (distributed lag raw/direction).
  b) Det stille h=5-cluster: alle metoders bedste configs ved lavere vol end EW.
  c) Lang-lookback-forfaldsbaandet: L>=13 ved h=1-2; mild - ingen katastrofepocket.
  d) Horisont-stratificeringen: hele skyen farvet efter h.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
MAD_CSV = ("final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/"
           "mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv")
OUT = AUDIT / "mad_tau04_bundle" / "gnc_mad_risk_return_scatter.png"

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
METHOD_COLORS = {
    "elastic_net_panel": "#2b9fd6",
    "random_forest_panel": "#00a878",
    "ols_predictive": "#edae13",
    "distributed_lag": "#d889b8",
}
METHOD_LABELS = {
    "elastic_net_panel": "Elastic net panel",
    "random_forest_panel": "Random forest panel",
    "ols_predictive": "OLS predictive",
    "distributed_lag": "Distributed lag",
}
SIGNAL_MARKERS = {"raw": "o", "direction": "^", "direction_risk_scaled": "s"}
SIGNAL_LABELS = {"raw": "raw", "direction": "dir", "direction_risk_scaled": "DRS"}
X = "monthly_vol_treasury_excess"
Y = "mean_monthly_excess_return_treasury"
S = "annualized_sharpe_treasury_excess"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv(REPO / MAD_CSV)
grid = df[df["model_id"].isin(METHOD_COLORS)].copy()
best = grid.loc[grid[S].idxmax()]
worst = grid.loc[grid[S].idxmin()]
ew = grid.groupby("forecast_horizon_months").first().reset_index()

REGIONS = {
    "a": {"x": (0.0315, 0.0438), "y": (0.0062, 0.0112)},   # vinder-armen (venstre -0.3pp, bund -0.1pp)
    "b": {"x": (0.0250, 0.0330), "y": (0.0028, 0.0082)},   # h=5-clusteret inkl. L7-topperne (top loeftet + hoejre kant udvidet)
    "c": {"x": (0.0320, 0.0420), "y": (-0.0008, 0.0024)},  # forfald (venstre +0.2pp, hoejre -0.2pp)
}


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


def pct_axes(ax, dec=1):
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=dec))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=dec))


def region_rows(key):
    r = REGIONS[key]
    return grid[grid[X].between(*r["x"]) & grid[Y].between(*r["y"])].copy()


fig = plt.figure(figsize=(8.27, 11.69), dpi=150, facecolor=BG)
gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[0.30, 3.25, 1.70, 1.70],
                      hspace=0.42, wspace=0.27, left=0.105, right=0.955, top=0.970, bottom=0.085)

ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
ax_t.text(0, 0.90, "Risk-return map with variable-specific zoom panels",
          fontsize=15, fontweight="bold", ha="left", va="center")
ax_t.text(0, 0.34, "USD19 suite at tau=0.4. Zoom panels show the high-return frontier, "
                   "the low-volatility h=5 cluster, the long-lookback decay band, and horizon stratification.",
          fontsize=7.8, color=MUTED, ha="left")

# ===== hovedpanel =====
ax = fig.add_subplot(gs[1, :])
style_axes(ax)
for model, part in grid.groupby("model_id", sort=False):
    ax.scatter(part[X], part[Y], s=25, color=METHOD_COLORS[model], alpha=0.65,
               edgecolor="white", linewidth=0.25, label=METHOD_LABELS[model])
ax.scatter(best[X], best[Y], marker="*", s=110, color="#111820", edgecolor="white",
           linewidth=0.45, zorder=6, label="Best Sharpe")
ax.scatter(ew["equal_weight_monthly_vol_treasury_excess"], ew["equal_weight_mean_monthly_excess_return_treasury"],
           marker="D", s=42, color="#5b626b", edgecolor="white", linewidth=0.35, zorder=6, label="Equal weight (per h)")
for _, row in ew.iterrows():
    ax.annotate(f"h{int(row['forecast_horizon_months'])}",
                xy=(row["equal_weight_monthly_vol_treasury_excess"], row["equal_weight_mean_monthly_excess_return_treasury"]),
                xytext=(4, -9), textcoords="offset points", fontsize=6.4, color=MUTED)
ax.annotate("Best", xy=(best[X], best[Y]), xytext=(6, 2), textcoords="offset points", fontsize=7.2)
ax.scatter(worst[X], worst[Y], marker="X", s=60, color="#9b3e5d", edgecolor="white", linewidth=0.4, zorder=6)
ax.annotate(f"Worst Sharpe {worst[S]:.2f}", xy=(worst[X], worst[Y]), xytext=(6, -10),
            textcoords="offset points", fontsize=6.8, color="#9b3e5d")
for key, r in REGIONS.items():
    ax.add_patch(Rectangle((r["x"][0], r["y"][0]), r["x"][1] - r["x"][0], r["y"][1] - r["y"][0],
                           fill=False, linestyle=(0, (3, 3)), linewidth=0.9, edgecolor=INK))
    ax.text(r["x"][0], r["y"][1] + 0.00016, key, fontsize=10, fontweight="bold", ha="left", va="bottom")
ax.set_xlim(0.0240, 0.0450)
ax.set_ylim(-0.0012, 0.0118)
pct_axes(ax)
ax.set_xlabel("Monthly volatility of Treasury-excess returns", fontsize=8.5)
ax.set_ylabel("Mean monthly Treasury-excess return", fontsize=8.5)
handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=METHOD_COLORS[m],
                  markeredgecolor="white", markeredgewidth=0.3, markersize=5.5, label=METHOD_LABELS[m])
           for m in METHOD_COLORS]
handles += [Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="#111820",
                   markeredgecolor="#111820", markersize=7.5, label="Best Sharpe"),
            Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#5b626b",
                   markeredgecolor="#5b626b", markersize=5.0, label="Equal weight (per h)"),
            Line2D([0], [0], marker="X", linestyle="None", markerfacecolor="#9b3e5d",
                   markeredgecolor="#9b3e5d", markersize=6.0, label="Worst Sharpe")]
ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2, columnspacing=1.1,
          handletextpad=0.45, fontsize=6.9)

ax_a = fig.add_subplot(gs[2, 0])
ax_b = fig.add_subplot(gs[2, 1])
ax_c = fig.add_subplot(gs[3, 0])
ax_d = fig.add_subplot(gs[3, 1])
for a in (ax_a, ax_b, ax_c, ax_d):
    style_axes(a)

# ===== a: hoejafkast-frontieren — drawdown-farve, form = variant =====
pa = region_rows("a")
DD = "max_drawdown_excess_treasury"
norm_a = mpl.colors.Normalize(vmin=-0.24, vmax=-0.10)
for sig, marker in SIGNAL_MARKERS.items():
    pts = pa[pa["signal_variant"] == sig]
    ax_a.scatter(pts[X], pts[Y], c=pts[DD], cmap="RdYlGn", norm=norm_a,
                 marker=marker, s=42, edgecolor="#26323c", linewidth=0.4, zorder=3)
ax_a.scatter(best[X], best[Y], marker="*", s=120, color="#111820", edgecolor="white", linewidth=0.5, zorder=6)
ax_a.annotate("Best 0.915", xy=(best[X], best[Y]), xytext=(-46, -4),
              textcoords="offset points", fontsize=6.2, color=INK)
ax_a.set_title("a  High-return frontier: colour = max drawdown", fontsize=8.2)
ax_a.set_xlim(*REGIONS["a"]["x"]); ax_a.set_ylim(*REGIONS["a"]["y"])
pct_axes(ax_a)
cb = fig.colorbar(mpl.cm.ScalarMappable(cmap="RdYlGn", norm=norm_a), ax=ax_a, fraction=0.045, pad=0.02)
cb.set_label("Treasury-excess max drawdown", fontsize=6.0)
cb.ax.tick_params(labelsize=6)
cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
hs = [Line2D([0], [0], marker=m, linestyle="None", markerfacecolor="#9bb7c9", markeredgecolor="#26323c",
             markersize=4.6, label=SIGNAL_LABELS[s2]) for s2, m in SIGNAL_MARKERS.items()]
ax_a.legend(handles=hs, frameon=False, fontsize=6.2, loc="lower right", handletextpad=0.3)

# ===== b: h=5-clusteret (farve = turnover) =====
pb = region_rows("b")
tmean = grid["mean_turnover"].mean()
norm_b = mpl.colors.Normalize(vmin=pb["mean_turnover"].min(), vmax=pb["mean_turnover"].max())
sc = ax_b.scatter(pb[X], pb[Y], c=pb["mean_turnover"], cmap="YlOrBr", norm=norm_b, s=40,
                  edgecolor="#26323c", linewidth=0.4)
ew5 = ew[ew["forecast_horizon_months"] == 5].iloc[0]
ax_b.annotate(f"Equal weight h=5 sits at {100 * ew5['equal_weight_monthly_vol_treasury_excess']:.1f}% vol (off-panel, right)",
              xy=(REGIONS["b"]["x"][1] - 0.0002, REGIONS["b"]["y"][0] + 0.0002), fontsize=6.2, color=MUTED, ha="right")
top4 = pb.nlargest(4, S)
ax_b.scatter(top4.iloc[0][X], top4.iloc[0][Y], marker="*", s=90, color="#111820",
             edgecolor="white", linewidth=0.4, zorder=6)
ax_b.set_title("b  Quiet h=5 cluster", fontsize=8.2)
ax_b.set_xlim(*REGIONS["b"]["x"]); ax_b.set_ylim(*REGIONS["b"]["y"])
pct_axes(ax_b)
cb2 = fig.colorbar(sc, ax=ax_b, fraction=0.045, pad=0.02)
cb2.set_label(f"Mean turnover (grid mean {tmean:.2f})", fontsize=6.0); cb2.ax.tick_params(labelsize=6)

# ===== c: lang-lookback-forfald (farve = L) =====
pc = region_rows("c")
norm_c = mpl.colors.Normalize(vmin=10, vmax=18)
for sig, marker in SIGNAL_MARKERS.items():
    pts = pc[pc["signal_variant"] == sig]
    ax_c.scatter(pts[X], pts[Y], c=pts["lookback_months"], cmap="RdPu", norm=norm_c,
                 marker=marker, s=38, edgecolor="#26323c", linewidth=0.35)
ax_c.scatter(worst[X], worst[Y], marker="X", s=70, color="#111820", edgecolor="white", linewidth=0.45, zorder=6)
ax_c.annotate(f"Worst: EN dir h2 L15\nSharpe {worst[S]:.3f}", xy=(worst[X], worst[Y]),
              xytext=(6, 6), textcoords="offset points", fontsize=6.4, color=INK)
ax_c.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
ax_c.set_title("c  Long-lookback decay band (h=1-2, L>=13)", fontsize=8.2)
ax_c.set_xlim(*REGIONS["c"]["x"]); ax_c.set_ylim(*REGIONS["c"]["y"])
pct_axes(ax_c)
cb3 = fig.colorbar(mpl.cm.ScalarMappable(cmap="RdPu", norm=norm_c), ax=ax_c, fraction=0.045, pad=0.02)
cb3.set_label("L (months)", fontsize=6.4); cb3.ax.tick_params(labelsize=6)

# ===== d: horisont-stratificering =====
norm_d = mpl.colors.Normalize(vmin=1, vmax=5)
sc4 = ax_d.scatter(grid[X], grid[Y], c=grid["forecast_horizon_months"], cmap="viridis", norm=norm_d,
                   s=16, alpha=0.8, edgecolor="white", linewidth=0.15)
for h, gp in grid.groupby("forecast_horizon_months"):
    ax_d.annotate(f"h{int(h)}", xy=(gp[X].median(), gp[Y].max() + 0.0004),
                  fontsize=6.8, fontweight="bold", ha="center",
                  color=mpl.cm.viridis(norm_d(h)))
ax_d.set_title("d  Volatility is set by the forecast horizon", fontsize=8.2)
ax_d.set_xlim(0.0240, 0.0450)
ax_d.set_ylim(-0.0012, 0.0125)
pct_axes(ax_d)
cb4 = fig.colorbar(sc4, ax=ax_d, fraction=0.045, pad=0.02, ticks=[1, 2, 3, 4, 5])
cb4.set_label("Forecast horizon h", fontsize=6.4); cb4.ax.tick_params(labelsize=6)

fig.text(0.105, 0.018, "Source: bd5d48e usd19_no_sri_lanka_28623539 portfolio_metrics_treasury_adjusted.csv; "
                       "GNC-informed configurations (840), tau=0.4, ROI10.", fontsize=6.6, color=MUTED)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=BG)
plt.close(fig)
print(OUT)
print(f"a: {len(region_rows('a'))} configs | b: {len(region_rows('b'))} | c: {len(region_rows('c'))}")
