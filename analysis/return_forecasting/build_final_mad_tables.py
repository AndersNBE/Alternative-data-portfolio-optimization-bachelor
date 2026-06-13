from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd


SOURCE_COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
SUITE_REL = "final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539"
MAD_REL = f"{SUITE_REL}/mad_portfolio_cleaned"
GNC_MODELS = [
    "distributed_lag",
    "elastic_net_panel",
    "ols_predictive",
    "random_forest_panel",
]
BASELINE_MODELS = ["historical_mean", "always_positive"]

MODEL_LABELS = {
    "always_positive": "Always positive",
    "distributed_lag": "Distributed lag",
    "elastic_net_panel": "Elastic net panel",
    "historical_mean": "Historical mean",
    "ols_predictive": "OLS predictive",
    "random_forest_panel": "Random forest panel",
}

SIGNAL_LABELS = {
    "raw": "Raw",
    "direction": "Direction",
    "direction_risk_scaled": "DRS",
}


def artifact_blob(repo_root: Path, commit: str, rel_path: str) -> bytes:
    local_path = repo_root / rel_path
    if local_path.exists():
        return local_path.read_bytes()
    raise FileNotFoundError(
        f"Missing frozen artifact {rel_path}. Expected it as a local file materialized from {commit}."
    )


def is_lfs_pointer(raw: bytes) -> bool:
    return raw.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def artifact_csv(repo_root: Path, commit: str, rel_path: str) -> pd.DataFrame:
    raw = artifact_blob(repo_root, commit, rel_path)
    if is_lfs_pointer(raw):
        raise RuntimeError(
            f"{rel_path} is a Git LFS pointer. Materialize LFS or pass a concrete --weights-csv path."
        )
    return pd.read_csv(io.BytesIO(raw))


def read_csv_source(repo_root: Path, commit: str, rel_path: str, local_path: Path | None = None) -> pd.DataFrame:
    if local_path is not None:
        return pd.read_csv(local_path)
    return artifact_csv(repo_root, commit, rel_path)


def label_model(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value).replace("_", " ").title())


def label_signal(value: str) -> str:
    return SIGNAL_LABELS.get(str(value), str(value).replace("_", " ").title())


def escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def save_latex_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(frame.columns)
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(escape_latex(fmt(row[col])) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def gnc_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["model_id"].isin(GNC_MODELS)].copy()


def top10_table(metrics: pd.DataFrame) -> pd.DataFrame:
    table = (
        gnc_metrics(metrics)
        .sort_values("annualized_sharpe_treasury_excess", ascending=False)
        .head(10)
        .copy()
    )
    return pd.DataFrame(
        {
            "Forecast method": table["model_id"].map(label_model),
            "h": table["forecast_horizon_months"].astype(int),
            "Signal": table["signal_variant"].map(label_signal),
            "L": table["lookback_months"].astype(int),
            "Treasury Sharpe": table["annualized_sharpe_treasury_excess"],
            "EW Treasury Sharpe": table["equal_weight_annualized_sharpe_treasury_excess"],
            "Raw compound": table["compound_return_original"],
            "Raw max DD": table["max_drawdown_original"],
            "Turnover": table["mean_turnover"],
        }
    )


def best_by_method_table(metrics: pd.DataFrame) -> pd.DataFrame:
    best = (
        gnc_metrics(metrics)
        .sort_values("annualized_sharpe_treasury_excess", ascending=False)
        .groupby("model_id", as_index=False)
        .head(1)
        .sort_values("annualized_sharpe_treasury_excess", ascending=False)
        .copy()
    )
    return pd.DataFrame(
        {
            "Forecast method": best["model_id"].map(label_model),
            "h": best["forecast_horizon_months"].astype(int),
            "Signal": best["signal_variant"].map(label_signal),
            "L": best["lookback_months"].astype(int),
            "Treasury Sharpe": best["annualized_sharpe_treasury_excess"],
            "EW Treasury Sharpe": best["equal_weight_annualized_sharpe_treasury_excess"],
            "Lift": best["annualized_sharpe_treasury_excess"]
            - best["equal_weight_annualized_sharpe_treasury_excess"],
            "Raw compound": best["compound_return_original"],
            "Raw max DD": best["max_drawdown_original"],
            "Turnover": best["mean_turnover"],
        }
    )


def method_horizon_table(metrics: pd.DataFrame) -> pd.DataFrame:
    best = (
        gnc_metrics(metrics)
        .sort_values("annualized_sharpe_treasury_excess", ascending=False)
        .groupby(["model_id", "forecast_horizon_months"], as_index=False)
        .head(1)
        .copy()
    )
    pivot = best.pivot_table(
        index="model_id",
        columns="forecast_horizon_months",
        values="annualized_sharpe_treasury_excess",
        aggfunc="first",
    )
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    pivot = pivot.reset_index()
    pivot["model_id"] = pivot["model_id"].map(label_model)
    pivot.columns = ["Forecast method"] + [f"h={int(col)}" for col in pivot.columns[1:]]
    return pivot


def baseline_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    top_gnc = gnc_metrics(metrics).sort_values("annualized_sharpe_treasury_excess", ascending=False).iloc[0]
    hist = metrics[metrics["model_id"] == "historical_mean"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]
    always = metrics[metrics["model_id"] == "always_positive"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]
    rows = [
        {
            "Portfolio": "GNC-informed MAD",
            "Forecast source": label_model(top_gnc["model_id"]),
            "h": int(top_gnc["forecast_horizon_months"]),
            "Signal": label_signal(top_gnc["signal_variant"]),
            "L": int(top_gnc["lookback_months"]),
            "Treasury Sharpe": top_gnc["annualized_sharpe_treasury_excess"],
            "Raw compound": top_gnc["compound_return_original"],
            "Raw max DD": top_gnc["max_drawdown_original"],
            "Turnover": top_gnc["mean_turnover"],
        },
        {
            "Portfolio": "Historical-mean MAD",
            "Forecast source": "No GNC",
            "h": int(hist["forecast_horizon_months"]),
            "Signal": label_signal(hist["signal_variant"]),
            "L": int(hist["lookback_months"]),
            "Treasury Sharpe": hist["annualized_sharpe_treasury_excess"],
            "Raw compound": hist["compound_return_original"],
            "Raw max DD": hist["max_drawdown_original"],
            "Turnover": hist["mean_turnover"],
        },
        {
            "Portfolio": "Always-positive MAD",
            "Forecast source": "No GNC",
            "h": int(always["forecast_horizon_months"]),
            "Signal": label_signal(always["signal_variant"]),
            "L": int(always["lookback_months"]),
            "Treasury Sharpe": always["annualized_sharpe_treasury_excess"],
            "Raw compound": always["compound_return_original"],
            "Raw max DD": always["max_drawdown_original"],
            "Turnover": always["mean_turnover"],
        },
        {
            "Portfolio": "Equal weight",
            "Forecast source": "Matched to winner",
            "h": int(top_gnc["forecast_horizon_months"]),
            "Signal": label_signal(top_gnc["signal_variant"]),
            "L": int(top_gnc["lookback_months"]),
            "Treasury Sharpe": top_gnc["equal_weight_annualized_sharpe_treasury_excess"],
            "Raw compound": top_gnc["equal_weight_compound_return_original"],
            "Raw max DD": top_gnc["equal_weight_max_drawdown_original"],
            "Turnover": "",
        },
    ]
    return pd.DataFrame(rows)


def weight_summary_table(metrics: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    best = gnc_metrics(metrics).sort_values("annualized_sharpe_treasury_excess", ascending=False).iloc[0]
    keys = {
        "model_id": best["model_id"],
        "forecast_horizon_months": int(best["forecast_horizon_months"]),
        "signal_variant": best["signal_variant"],
        "lookback_months": int(best["lookback_months"]),
    }
    work = weights.copy()
    for col in ["forecast_horizon_months", "lookback_months"]:
        work[col] = work[col].astype(int)
    mask = pd.Series(True, index=work.index)
    for col, value in keys.items():
        mask &= work[col] == value
    best_weights = work[mask].copy()
    if best_weights.empty:
        raise RuntimeError(f"No weights matched the best final configuration: {keys}")
    required = {"country", "weight", "signal"}
    missing = required.difference(best_weights.columns)
    if missing:
        raise RuntimeError(f"weights.csv is missing required columns: {sorted(missing)}")
    summary = (
        best_weights.groupby("country")
        .agg(
            average_weight=("weight", "mean"),
            max_weight=("weight", "max"),
            active_months=("weight", lambda s: int((s > 1e-9).sum())),
            average_signal=("signal", "mean"),
        )
        .sort_values("average_weight", ascending=False)
        .head(10)
        .reset_index()
    )
    return summary.rename(
        columns={
            "country": "Country",
            "average_weight": "Average weight",
            "max_weight": "Max weight",
            "active_months": "Active months",
            "average_signal": "Average signal",
        }
    )


def parse_lfs_oid(pointer_text: str) -> str | None:
    match = re.search(r"oid sha256:([0-9a-f]{64})", pointer_text)
    return match.group(1) if match else None


def load_weights(repo_root: Path, source_commit: str, weights_csv: Path | None) -> pd.DataFrame:
    if weights_csv is not None:
        return pd.read_csv(weights_csv)
    raw = artifact_blob(repo_root, source_commit, f"{MAD_REL}/weights.csv")
    if is_lfs_pointer(raw):
        oid = parse_lfs_oid(raw.decode("utf-8", errors="replace"))
        raise RuntimeError(
            "The final weights.csv object is stored through Git LFS and is not materialized in this checkout. "
            f"Materialize LFS object {oid or '<unknown>'} and rerun with --weights-csv /path/to/weights.csv."
        )
    return pd.read_csv(io.BytesIO(raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the locked final MAD .tex tables from bd5d48e/USD19 evidence.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--weights-csv",
        type=Path,
        default=None,
        help="Materialized final weights.csv. Defaults to the frozen local suite when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    metrics = artifact_csv(repo_root, args.source_commit, f"{MAD_REL}/portfolio_metrics_treasury_adjusted.csv")
    weights = load_weights(repo_root, args.source_commit, args.weights_csv)
    outputs = {
        "table_gnc_mad_top10_configurations.tex": top10_table(metrics),
        "table_gnc_mad_best_sharpe_by_forecast_method.tex": best_by_method_table(metrics),
        "table_gnc_mad_sharpe_by_method_and_horizon.tex": method_horizon_table(metrics),
        "table_gnc_mad_weight_summary_top10.tex": weight_summary_table(metrics, weights),
        "table_mad_baseline_comparison_treasury.tex": baseline_comparison_table(metrics),
    }
    for name, frame in outputs.items():
        save_latex_table(frame, args.out_dir / name)
        print(args.out_dir / name)


if __name__ == "__main__":
    main()
