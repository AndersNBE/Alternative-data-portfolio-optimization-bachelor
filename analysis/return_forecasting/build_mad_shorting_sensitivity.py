from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "bachelor-mad-shorting-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from analysis.return_forecasting.run_cleaned_research_suite import portfolio_metrics, transform_signal


GNC_MODELS = [
    "ols_predictive",
    "distributed_lag",
    "elastic_net_panel",
    "random_forest_panel",
]


@dataclass(frozen=True)
class ShortingSetting:
    name: str
    min_weight: float
    max_weight: float
    gross_exposure_limit: float


SETTINGS = [
    ShortingSetting("mild_short_gross_1p20", min_weight=-0.10, max_weight=0.35, gross_exposure_limit=1.20),
    ShortingSetting("controlled_short_gross_1p50", min_weight=-0.20, max_weight=0.35, gross_exposure_limit=1.50),
]


def _format_model_name(model_id: str) -> str:
    return {
        "ols_predictive": "OLS predictive",
        "distributed_lag": "Distributed lag",
        "elastic_net_panel": "Elastic net panel",
        "random_forest_panel": "Random forest panel",
    }.get(model_id, model_id.replace("_", " ").title())


def solve_mad_weights_long_short(
    *,
    scenarios: np.ndarray,
    expected: np.ndarray,
    min_weight: float,
    max_weight: float,
    gross_exposure_limit: float,
) -> tuple[np.ndarray | None, bool]:
    scenarios = np.asarray(scenarios, dtype=float)
    n_scenarios, n_assets = scenarios.shape
    if n_assets == 0 or n_scenarios == 0:
        return None, False

    # Variables: weights n, deviations T, scenario mean 1, absolute weights n.
    weight_start = 0
    dev_start = n_assets
    mean_idx = dev_start + n_scenarios
    abs_start = mean_idx + 1
    n_vars = abs_start + n_assets

    c = np.zeros(n_vars)
    c[dev_start : dev_start + n_scenarios] = 1.0 / n_scenarios

    a_eq = np.zeros((2, n_vars))
    a_eq[0, weight_start:dev_start] = 1.0
    # Bind mu to the mean scenario portfolio return for classical mean-centered MAD.
    a_eq[1, weight_start:dev_start] = scenarios.mean(axis=0)
    a_eq[1, mean_idx] = -1.0
    b_eq = np.array([1.0, 0.0])

    a_ub_rows: list[np.ndarray] = []
    b_ub_rows: list[float] = []

    for s_idx in range(n_scenarios):
        row = np.zeros(n_vars)
        row[weight_start:dev_start] = scenarios[s_idx]
        row[dev_start + s_idx] = -1.0
        row[mean_idx] = -1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

        row = np.zeros(n_vars)
        row[weight_start:dev_start] = -scenarios[s_idx]
        row[dev_start + s_idx] = -1.0
        row[mean_idx] = 1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

    expected = np.asarray(expected, dtype=float)
    if np.isfinite(expected).all() and np.nanstd(expected) > 0:
        row = np.zeros(n_vars)
        row[weight_start:dev_start] = -expected
        a_ub_rows.append(row)
        b_ub_rows.append(-float(np.mean(expected)))

    # z_j >= |x_j|, implemented as x_j - z_j <= 0 and -x_j - z_j <= 0.
    for j in range(n_assets):
        row = np.zeros(n_vars)
        row[weight_start + j] = 1.0
        row[abs_start + j] = -1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

        row = np.zeros(n_vars)
        row[weight_start + j] = -1.0
        row[abs_start + j] = -1.0
        a_ub_rows.append(row)
        b_ub_rows.append(0.0)

    row = np.zeros(n_vars)
    row[abs_start : abs_start + n_assets] = 1.0
    a_ub_rows.append(row)
    b_ub_rows.append(float(gross_exposure_limit))

    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    bounds += [(0.0, None) for _ in range(n_scenarios)]
    bounds += [(None, None)]
    bounds += [(0.0, None) for _ in range(n_assets)]

    result = linprog(
        c,
        A_ub=np.asarray(a_ub_rows, dtype=float),
        b_ub=np.asarray(b_ub_rows, dtype=float),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None, False
    weights = np.asarray(result.x[:n_assets], dtype=float)
    if not np.isfinite(weights).all() or abs(weights.sum() - 1.0) > 1e-5:
        return None, False
    return weights, True


def backtest_mad_long_short(
    *,
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    lookback_months: int,
    signal_variant: str,
    setting: ShortingSetting,
) -> dict[str, pd.DataFrame]:
    dates = sorted(actual_returns.index.intersection(predicted_returns.index))
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous_weights: pd.Series | None = None

    for idx, date in enumerate(dates):
        if idx < lookback_months:
            continue
        history = actual_returns.loc[dates[idx - lookback_months : idx - 1]]
        realised = actual_returns.loc[date]
        signal = transform_signal(signal=predicted_returns.loc[date], history=history, variant=signal_variant)
        countries = sorted(
            set(history.dropna(axis=1, how="any").columns)
            .intersection(signal.dropna().index)
            .intersection(realised.dropna().index)
        )
        if len(countries) < 3:
            continue
        scenarios = history[countries].to_numpy(dtype=float)
        expected = signal[countries].to_numpy(dtype=float)
        weights, optimal = solve_mad_weights_long_short(
            scenarios=scenarios,
            expected=expected,
            min_weight=setting.min_weight,
            max_weight=setting.max_weight,
            gross_exposure_limit=setting.gross_exposure_limit,
        )
        if weights is None:
            weights = np.full(len(countries), 1.0 / len(countries))
            optimal = False

        realised_vec = realised[countries].to_numpy(dtype=float)
        port_return = float(np.dot(weights, realised_vec))
        equal_weight_return = float(np.mean(realised_vec))
        weights_series = pd.Series(weights, index=countries)
        turnover = (
            float(0.5 * np.abs(weights_series.subtract(previous_weights, fill_value=0.0)).sum())
            if previous_weights is not None
            else float("nan")
        )
        previous_weights = weights_series

        short_exposure = float(np.clip(-weights, 0.0, None).sum())
        long_exposure = float(np.clip(weights, 0.0, None).sum())
        gross_exposure = float(np.abs(weights).sum())
        return_rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_return": port_return,
                "equal_weight_return": equal_weight_return,
                "excess_return": port_return - equal_weight_return,
                "turnover": turnover,
                "optimal": bool(optimal),
                "n_assets": len(countries),
                "n_scenarios": scenarios.shape[0],
                "gross_exposure": gross_exposure,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
            }
        )
        for country, weight in zip(countries, weights):
            weight_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "country": country,
                    "weight": float(weight),
                    "signal": float(signal[country]),
                }
            )

    return {"returns": pd.DataFrame(return_rows), "weights": pd.DataFrame(weight_rows)}


def run_shorting_grid(
    *,
    predictions: pd.DataFrame,
    lookbacks: list[int],
    signal_variants: list[str],
    setting: ShortingSetting,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = predictions.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.to_period("M").dt.to_timestamp()
    work["actual_return"] = pd.to_numeric(work["actual_return"], errors="coerce")
    work["predicted_return"] = pd.to_numeric(work["predicted_return"], errors="coerce")
    work = work[work["model_id"].isin(GNC_MODELS)].dropna(subset=["actual_return", "predicted_return"])

    metrics_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for (model_id, horizon), group in work.groupby(["model_id", "forecast_horizon_months"]):
        actual_pivot = group.pivot_table(index="date", columns="country", values="actual_return", aggfunc="mean").sort_index()
        pred_pivot = group.pivot_table(index="date", columns="country", values="predicted_return", aggfunc="mean").sort_index()
        for lookback in lookbacks:
            for variant in signal_variants:
                backtest = backtest_mad_long_short(
                    actual_returns=actual_pivot,
                    predicted_returns=pred_pivot,
                    lookback_months=lookback,
                    signal_variant=variant,
                    setting=setting,
                )
                if backtest["returns"].empty:
                    continue
                returns = backtest["returns"]
                metrics = portfolio_metrics(returns, forecast_horizon_months=int(horizon))
                metrics.update(
                    {
                        "mean_gross_exposure": float(returns["gross_exposure"].mean()),
                        "mean_long_exposure": float(returns["long_exposure"].mean()),
                        "mean_short_exposure": float(returns["short_exposure"].mean()),
                        "max_short_exposure": float(returns["short_exposure"].max()),
                    }
                )
                row = {
                    "shorting_setting": setting.name,
                    "model_id": model_id,
                    "forecast_horizon_months": int(horizon),
                    "signal_variant": variant,
                    "lookback_months": int(lookback),
                    **metrics,
                }
                metrics_rows.append(row)

                p = returns.copy()
                p.insert(0, "shorting_setting", setting.name)
                p.insert(1, "model_id", model_id)
                p.insert(2, "forecast_horizon_months", int(horizon))
                p.insert(3, "signal_variant", variant)
                p.insert(4, "lookback_months", int(lookback))
                portfolio_rows.extend(p.to_dict("records"))

                w = backtest["weights"].copy()
                w.insert(0, "shorting_setting", setting.name)
                w.insert(1, "model_id", model_id)
                w.insert(2, "forecast_horizon_months", int(horizon))
                w.insert(3, "signal_variant", variant)
                w.insert(4, "lookback_months", int(lookback))
                weight_rows.extend(w.to_dict("records"))

    return pd.DataFrame(metrics_rows), pd.DataFrame(portfolio_rows), pd.DataFrame(weight_rows)


def _plot_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = summary["setting_label"]

    axes[0].bar(labels, summary["annualized_sharpe"], color=["#2563eb", "#7c3aed", "#f97316"], alpha=0.88)
    axes[0].set_title("Best annualized Sharpe", fontsize=13)
    axes[0].set_ylabel("Sharpe ratio")

    axes[1].bar(labels, summary["compound_return"], color=["#2563eb", "#7c3aed", "#f97316"], alpha=0.88)
    axes[1].set_title("Best compound return", fontsize=13)
    axes[1].set_ylabel("Compound return")

    axes[2].bar(labels, summary["max_drawdown"], color=["#2563eb", "#7c3aed", "#f97316"], alpha=0.88)
    axes[2].set_title("Best max drawdown", fontsize=13)
    axes[2].set_ylabel("Max drawdown")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("GNC-MAD Long-Only vs Shorting Sensitivity", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_shorting_summary_comparison.png", dpi=220)
    plt.close(fig)


def _plot_wealth(returns: pd.DataFrame, out_dir: Path, filename: str, title: str) -> None:
    data = returns.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    data["mad_wealth"] = np.exp(data["portfolio_return"].cumsum())
    data["equal_weight_wealth"] = np.exp(data["equal_weight_return"].cumsum())
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(data["date"], data["mad_wealth"], label="MAD portfolio", linewidth=2.4, color="#2563eb")
    ax.plot(data["date"], data["equal_weight_wealth"], label="Equal weight", linewidth=2.1, color="#dc2626")
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel("Cumulative wealth")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def _best_row(metrics: pd.DataFrame) -> pd.Series:
    return metrics.sort_values("annualized_sharpe", ascending=False).iloc[0]


def build_report(base_dir: Path, out_dir: Path) -> None:
    predictions = pd.read_csv(base_dir / "all_predictions_eval_window.csv")
    long_only_metrics = pd.read_csv(base_dir / "mad_portfolio_cleaned/report_assets_gnc_forecasts/gnc_mad_all_config_metrics.csv")
    long_only_returns = pd.read_csv(base_dir / "mad_portfolio_cleaned/portfolio_returns.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_short_metrics = []
    all_short_returns = []
    all_short_weights = []
    lookbacks = list(range(5, 19))
    signal_variants = ["raw", "direction", "direction_risk_scaled"]
    for setting in SETTINGS:
        metrics, returns, weights = run_shorting_grid(
            predictions=predictions,
            lookbacks=lookbacks,
            signal_variants=signal_variants,
            setting=setting,
        )
        all_short_metrics.append(metrics)
        all_short_returns.append(returns)
        all_short_weights.append(weights)

    short_metrics = pd.concat(all_short_metrics, ignore_index=True)
    short_returns = pd.concat(all_short_returns, ignore_index=True)
    short_weights = pd.concat(all_short_weights, ignore_index=True)

    short_metrics.to_csv(out_dir / "gnc_mad_shorting_all_config_metrics.csv", index=False)
    short_returns.to_csv(out_dir / "gnc_mad_shorting_portfolio_returns.csv", index=False)
    short_weights.to_csv(out_dir / "gnc_mad_shorting_weights.csv", index=False)

    long_best = _best_row(long_only_metrics)
    rows = []
    rows.append(
        {
            "setting": "long_only",
            "setting_label": "Long-only",
            "model_id": long_best["model_id"],
            "forecast_horizon_months": int(long_best["forecast_horizon_months"]),
            "signal_variant": long_best["signal_variant"],
            "lookback_months": int(long_best["lookback_months"]),
            "annualized_sharpe": float(long_best["annualized_sharpe"]),
            "equal_weight_annualized_sharpe": float(long_best["equal_weight_annualized_sharpe"]),
            "compound_return": float(long_best["compound_return"]),
            "equal_weight_compound_return": float(long_best["equal_weight_compound_return"]),
            "max_drawdown": float(long_best["max_drawdown"]),
            "equal_weight_max_drawdown": float(long_best["equal_weight_max_drawdown"]),
            "mean_turnover": float(long_best["mean_turnover"]),
            "mean_gross_exposure": 1.0,
            "mean_short_exposure": 0.0,
            "max_short_exposure": 0.0,
        }
    )
    for setting in SETTINGS:
        best = _best_row(short_metrics[short_metrics["shorting_setting"] == setting.name])
        rows.append(
            {
                "setting": setting.name,
                "setting_label": setting.name.replace("_", " ").replace("p", "."),
                "model_id": best["model_id"],
                "forecast_horizon_months": int(best["forecast_horizon_months"]),
                "signal_variant": best["signal_variant"],
                "lookback_months": int(best["lookback_months"]),
                "annualized_sharpe": float(best["annualized_sharpe"]),
                "equal_weight_annualized_sharpe": float(best["equal_weight_annualized_sharpe"]),
                "compound_return": float(best["compound_return"]),
                "equal_weight_compound_return": float(best["equal_weight_compound_return"]),
                "max_drawdown": float(best["max_drawdown"]),
                "equal_weight_max_drawdown": float(best["equal_weight_max_drawdown"]),
                "mean_turnover": float(best["mean_turnover"]),
                "mean_gross_exposure": float(best["mean_gross_exposure"]),
                "mean_short_exposure": float(best["mean_short_exposure"]),
                "max_short_exposure": float(best["max_short_exposure"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "gnc_mad_shorting_best_summary.csv", index=False)
    summary.to_latex(
        out_dir / "table_gnc_mad_shorting_best_summary.tex",
        index=False,
        float_format="%.3f",
        escape=False,
    )
    _plot_comparison(summary, out_dir)

    for row in summary.itertuples(index=False):
        if row.setting == "long_only":
            mask = (
                (long_only_returns["model_id"] == row.model_id)
                & (long_only_returns["forecast_horizon_months"] == row.forecast_horizon_months)
                & (long_only_returns["signal_variant"] == row.signal_variant)
                & (long_only_returns["lookback_months"] == row.lookback_months)
            )
            returns = long_only_returns[mask].copy()
        else:
            mask = (
                (short_returns["shorting_setting"] == row.setting)
                & (short_returns["model_id"] == row.model_id)
                & (short_returns["forecast_horizon_months"] == row.forecast_horizon_months)
                & (short_returns["signal_variant"] == row.signal_variant)
                & (short_returns["lookback_months"] == row.lookback_months)
            )
            returns = short_returns[mask].copy()
        returns.to_csv(out_dir / f"{row.setting}_best_returns.csv", index=False)
        _plot_wealth(
            returns,
            out_dir,
            f"{row.setting}_best_wealth_vs_equal_weight.png",
            f"{row.setting_label}: best GNC-MAD vs equal weight",
        )

    top_short = short_metrics.sort_values("annualized_sharpe", ascending=False).head(20)
    top_short.to_csv(out_dir / "gnc_mad_shorting_top20_configurations.csv", index=False)

    manifest = {
        "source_suite": str(base_dir),
        "models": GNC_MODELS,
        "signal_variants": signal_variants,
        "lookbacks": lookbacks,
        "shorting_settings": [setting.__dict__ for setting in SETTINGS],
        "interpretation_note": (
            "Shorting is tested with net exposure sum(weights)=1, per-country lower/upper bounds, "
            "and a gross exposure limit. This avoids the unstable unconstrained long-short case."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled shorting sensitivity for GNC-informed MAD portfolios.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/outputs/return_forecasting/usd19_no_sri_lanka_28623539/"
            "mad_portfolio_cleaned/report_assets_gnc_shorting_sensitivity"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.base_dir, args.out_dir)


if __name__ == "__main__":
    main()
