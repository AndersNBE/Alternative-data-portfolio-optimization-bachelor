"""Phase D: Appendiks-siderne med alle 23 landes GNC-paneler (bd5d48e, tau=0.4, ROI10)."""
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

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "Bachelor-portfolio"
COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
OUT_DIR = AUDIT / "panels_tau04_bundle"
SOURCE = "Source: Bachelor-portfolio@bd5d48e final_runs/tau04_hk_28623539 (tau=0.4, ROI10)."

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
BLUE = "#2b9fd6"
GOLD = "#edae13"

PORT_COUNTRY = {
    "abu_dhabi": "UAE", "algeciras": "Spanien", "antwerpbrugges": "Belgien",
    "balboa": "Panama", "bremen": "Tyskland", "busan": "Sydkorea",
    "cai_mep": "Vietnam", "colombo": "Sri Lanka", "colon": "Panama",
    "da_lian": "Kina", "dongguan": "Kina", "guangxi_beibu": "Kina",
    "guangzhou": "Kina", "hai_phong": "Vietnam", "hamborg": "Tyskland",
    "ho_chi_minh_city": "Vietnam", "hong_kong": "Hong Kong", "houston": "USA",
    "jawaharal_nehru": "Indien", "jebel_ali": "UAE", "kaohsiung": "Taiwan",
    "laem_chabang": "Thailand", "lianyungang": "Kina", "long_beach": "USA",
    "manila": "Filippinerne", "mundra": "Indien", "new_york_new_jersey": "USA",
    "ningbozhoushan": "Kina", "piraeus": "Grækenland", "port_klang": "Malaysia",
    "qing_dao": "Kina", "rizhao": "Kina", "rotterdam": "Holland",
    "santos": "Brasilien", "savannah": "USA", "shanghai": "Kina",
    "shenzhen": "Kina", "singapore": "Singapore", "suzhou": "Kina",
    "tanger_med": "Marokko", "tanjung_pelepas": "Malaysia", "tanjung_perak": "Malaysia",
    "tanjung_priok": "Indonesia", "tianjin": "Kina", "tokyo": "Japan",
    "valencia": "Spanien", "xiamen": "Kina", "yantai": "Kina", "yingkou": "Kina",
}
COUNTRY_LABELS = {"Belgien": "Belgium", "Brasilien": "Brazil", "Filippinerne": "Philippines",
                  "Holland": "Netherlands", "Indien": "India", "Kina": "China",
                  "Sydkorea": "South Korea", "Tyskland": "Germany", "Spanien": "Spain",
                  "Grækenland": "Greece", "Marokko": "Morocco"}


def main() -> None:
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG,
    })
    pt = pd.read_csv(REPO / "final_runs/tau04_hk_28623539/daily_container_index/port_timeseries.csv", parse_dates=["date"])
    pt["month"] = pt["date"].dt.to_period("M").dt.to_timestamp()
    port = pt.groupby(["port_id", "month"], as_index=False).agg(NC=("NC", "median"))
    port = port.sort_values(["port_id", "month"])
    port["mi"] = port["month"].dt.year * 12 + port["month"].dt.month
    port["pNC"] = port.groupby("port_id")["NC"].shift(1)
    port["pmi"] = port.groupby("port_id")["mi"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        port["GNC"] = np.log(port["NC"] / port["pNC"]) / (port["mi"] - port["pmi"])
    valid = port[port["GNC"].notna()].replace([np.inf, -np.inf], np.nan).dropna(subset=["GNC"])
    valid = valid.copy()
    valid["country"] = valid["port_id"].map(PORT_COUNTRY)
    cm = valid.groupby(["country", "month"], as_index=False).agg(GNC=("GNC", "mean"), n_ports=("port_id", "nunique"))

    countries = sorted(cm["country"].unique())
    pages = [countries[:12], countries[12:]]
    for page_no, page_countries in enumerate(pages, start=1):
        rows = int(np.ceil(len(page_countries) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(13.4, 2.85 * rows + 1.2), squeeze=False)
        fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.06, hspace=0.62, wspace=0.22)
        fig.text(0.04, 0.975, f"Country-level GNC signals — appendix page {page_no}",
                 ha="left", va="top", fontsize=14.5, weight="bold", color=INK)
        fig.text(0.04, 0.945, "Blue: monthly GNC (mean of active ports). Gold: 3-month moving average. Median active ports shown per panel.",
                 ha="left", va="top", fontsize=8.6, color=MUTED)
        for ax in axes.flat[len(page_countries):]:
            ax.axis("off")
        for ax, country in zip(axes.flat, page_countries):
            g = cm[cm["country"] == country].sort_values("month")
            ax.set_facecolor(BG)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.plot(g["month"], g["GNC"], color=BLUE, linewidth=0.85, marker="o", markersize=1.6, alpha=0.85)
            ax.plot(g["month"], g["GNC"].rolling(3, min_periods=2).mean(), color=GOLD, linewidth=1.5)
            ax.axhline(0, color=AXIS, linewidth=0.7, alpha=0.6)
            ax.set_title(f"{COUNTRY_LABELS.get(country, country)}  (median ports: {int(g['n_ports'].median())})",
                         fontsize=9.4, loc="left", color=INK, weight="bold")
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7.2)
        fig.text(0.04, 0.012, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
        path = OUT_DIR / f"container_index_country_gnc_appendix_page_{page_no}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
