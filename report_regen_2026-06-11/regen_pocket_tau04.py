"""Anti-overfitting-figuren: vaelg den robuste konfigurationsregion, ikke det enkelte bedste punkt.

Argument (bd5d48e USD19-suiten, 840 GNC-configs):
  - Vinderen (DL h=1 raw L=5, Sharpe 0.915) er isoleret: kun 2/840 over 0.84,
    og naboen L=7 falder til 0.72 -> kniv-aeg / selektionsbias-risiko.
  - Lommen (ret 0.6-0.85%, vol 3.0-3.5%) = 52 configs, ALLE i griddets top 13%,
    median 0.73 = 96. percentil; kontiguoes parameterblok: h=4-5 (+h=3), L=6-9,
    alle 4 metoder + alle 3 varianter; median max DD -13.3% vs -19.8% for samme
    afkast ved hoejere vol.
  -> Opskriften er h og L (lang horisont + kort lookback), ikke et enkelt punkt.
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
REPO = AUDIT.parent / "Bachelor-portfolio"
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
MAD_CSV = ("final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/"
           "mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv")
OUT = AUDIT / "mad_tau04_bundle" / "gnc_mad_high_return_low_vol_pocket.png"

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
GREEN = "#00a878"
GREEN_DARK = "#006d58"
PINK_DARK = "#9b3e5d"
X = "monthly_vol_treasury_excess"
Y = "mean_monthly_excess_return_treasury"
S = "annualized_sharpe_treasury_excess"
DD = "max_drawdown_excess_treasury"
WIN_X = (0.030, 0.035)
WIN_Y = (0.006, 0.0085)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


df = pd.read_csv(REPO / MAD_CSV)
gnc = df[df["model_id"].isin(["ols_predictive", "elastic_net_panel", "distributed_lag", "random_forest_panel"])].copy()
w = gnc[gnc[X].between(*WIN_X) & gnc[Y].between(*WIN_Y)].copy()
rest = gnc.drop(w.index)
winner = gnc.loc[gnc[S].idxmax()]
same_ret_high_vol = gnc[(gnc[X] > WIN_X[1]) & gnc[Y].between(*WIN_Y)]
top_share = 100 * (gnc[S] >= w[S].min()).mean()
med_pct = 100 * (gnc[S] <= w[S].median()).mean()
n_above_084 = int((gnc[S] > 0.84).sum())
dl_neighbors = gnc[(gnc["model_id"] == "distributed_lag") & (gnc["forecast_horizon_months"] == 1)
                   & (gnc["signal_variant"] == "raw")].set_index("lookback_months")[S]

fig = plt.figure(figsize=(11.8, 6.6), dpi=150, facecolor=BG)
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.55, 1.0], height_ratios=[1.18, 1.0],
                      left=0.07, right=0.975, top=0.815, bottom=0.150, hspace=0.66, wspace=0.26)

fig.text(0.04, 0.965, "A robust configuration region, not a single winner", ha="left", va="top",
         fontsize=15.0, weight="bold", color=INK)
fig.text(0.04, 0.905,
         "Selecting the single best-Sharpe configuration risks selection bias: it is an isolated extreme. The dense pocket of "
         "high-return, low-volatility configurations\ninstead defines a reproducible recipe: forecast horizon h=4–5 with MAD "
         "lookback L=6–9, regardless of forecast method and signal variant.",
         ha="left", va="top", fontsize=8.2, color=MUTED)

# ===== venstre: kontekstkort — outlier vs. klump =====
ax = fig.add_subplot(gs[:, 0])
style_axes(ax)
ax.scatter(rest[X], rest[Y], s=14, color="#b9c3cb", alpha=0.55, edgecolor="none", label="Other configurations")
ax.scatter(w[X], w[Y], s=30, color=GREEN, alpha=0.95, edgecolor="#1c4d40", linewidth=0.4,
           zorder=4, label="The pocket (52 configs)")
ax.add_patch(Rectangle((WIN_X[0], WIN_Y[0]), WIN_X[1] - WIN_X[0], WIN_Y[1] - WIN_Y[0],
                       fill=False, linestyle=(0, (3, 3)), linewidth=1.1, edgecolor=GREEN_DARK, zorder=5))
ax.scatter(winner[X], winner[Y], marker="*", s=150, color="#111820", edgecolor="white",
           linewidth=0.5, zorder=6, label="Single best Sharpe")
ax.annotate(
    f"Single best: DL, h=1, raw, L=5 (Sharpe 0.915)\n"
    f"Isolated: only {n_above_084} of 840 exceed 0.84, and the\n"
    f"L=7 neighbour drops to {dl_neighbors.get(7, float('nan')):.2f}",
    xy=(winner[X], winner[Y]), xytext=(-218, -42), textcoords="offset points",
    fontsize=7.2, color=INK,
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
ax.annotate(
    f"The pocket: every config in the grid's top {top_share:.0f}%\n(median Sharpe 0.73 = {med_pct:.0f}th percentile)",
    xy=(WIN_X[0] + 0.0007, WIN_Y[0] + 0.0002), xytext=(-66, -66), textcoords="offset points",
    fontsize=7.2, color=GREEN_DARK,
    arrowprops=dict(arrowstyle="-", color=GREEN_DARK, linewidth=0.7))
ax.set_xlim(0.0240, 0.0450)
ax.set_ylim(-0.0012, 0.0118)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax.set_xlabel("Monthly volatility of Treasury-excess returns", fontsize=9)
ax.set_ylabel("Mean monthly Treasury-excess return", fontsize=9)
ax.legend(frameon=False, loc="lower right", fontsize=7.2, handletextpad=0.4)
ax.set_title("Isolated winner vs. dense pocket", fontsize=9.2, pad=7)

# ===== hoejre top: h x L plateau-heatmap med opskrift-blok =====
ax1 = fig.add_subplot(gs[0, 1])
piv = gnc.groupby(["forecast_horizon_months", "lookback_months"])[S].mean().unstack()
im = ax1.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0.0, vmax=0.85)
ax1.set_yticks(range(5))
ax1.set_yticklabels([f"h={h}" for h in piv.index], fontsize=7.6)
ax1.set_xticks(range(0, 14, 2))
ax1.set_xticklabels([f"L{c}" for c in piv.columns[::2]], fontsize=7.0)
GREEN_BRIGHT = "#00eaa6"
pocket_cells = w.groupby(["forecast_horizon_months", "lookback_months"]).size()
for (h, L) in pocket_cells.index:
    ax1.add_patch(Rectangle((piv.columns.get_loc(L) - 0.5, piv.index.get_loc(h) - 0.5), 1, 1,
                            fill=False, edgecolor=GREEN_BRIGHT, linewidth=2.6, zorder=5))
ax1.scatter(piv.columns.get_loc(5), piv.index.get_loc(1), marker="*", s=70, color="#111820",
            edgecolor="white", linewidth=0.4, zorder=6)
ax1.annotate(f"single best lives here —\nbut its cell averages only {piv.loc[1, 5]:.2f}",
             xy=(piv.columns.get_loc(5), piv.index.get_loc(1)), xytext=(26, 2),
             textcoords="offset points", fontsize=6.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="none", alpha=0.88),
             arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
ax1.annotate("the recipe: a stable plateau",
             xy=(piv.columns.get_loc(9) + 0.4, piv.index.get_loc(5)), xytext=(30, 4),
             textcoords="offset points", fontsize=6.8, color=GREEN_DARK, weight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="none", alpha=0.88),
             arrowprops=dict(arrowstyle="-", color=GREEN_DARK, linewidth=0.6))
ax1.set_title("Mean Sharpe by horizon and lookback", fontsize=8.8, pad=6)
cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02)
cb.set_label("Mean Treasury Sharpe\n(across methods & variants)", fontsize=6.2)
cb.ax.tick_params(labelsize=6.2)

# ===== hoejre bund: gruppe-performance / sikkerhed =====
ax2 = fig.add_subplot(gs[1, 1])
style_axes(ax2)
ax2.grid(True, axis="x", color=GRID, linewidth=0.55)
ax2.grid(False, axis="y")
rows = [
    ("The pocket", w[DD].median(), GREEN),
    ("Same return,\nvol > 3.5% (87% h=1)", same_ret_high_vol[DD].median(), PINK_DARK),
    ("Full grid", gnc[DD].median(), "#8a949d"),
]
ypos = np.arange(len(rows))
ax2.barh(ypos, [abs(v) for _, v, _ in rows], color=[c for _, _, c in rows], height=0.56, alpha=0.92)
for yy, (_, v, _) in zip(ypos, rows):
    ax2.text(abs(v) + 0.004, yy, f"{100 * v:.1f}%", va="center", fontsize=7.8, weight="bold", color=INK)
ax2.set_yticks(ypos)
ax2.set_yticklabels([r[0] for r in rows], fontsize=7.4)
ax2.set_xlim(0, 0.26)
ax2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax2.invert_yaxis()
ax2.set_title("Median Treasury-excess max drawdown", fontsize=8.8, pad=6)

fig.text(0.5, 0.062,
         "The investable recipe:   forecast horizon h = 4–5   ·   MAD lookback L = 6–9   ·   any of the four forecast methods   ·   any signal variant",
         ha="center", va="center", fontsize=9.2, weight="bold", color=GREEN_DARK,
         bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfbf5", edgecolor=GREEN_DARK, linewidth=1.3))
fig.text(0.04, 0.016,
         "Source: bd5d48e usd19_no_sri_lanka_28623539 portfolio_metrics_treasury_adjusted.csv; GNC-informed configurations (840), tau=0.4, ROI10. "
         "Pocket window: return 0.6–0.85%, volatility 3.0–3.5%.",
         fontsize=6.8, color=MUTED)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=BG)
plt.close(fig)
print(OUT)
print(f"pocket {len(w)} | top-share {top_share:.1f}% | median pct {med_pct:.1f} | >0.84: {n_above_084}")
