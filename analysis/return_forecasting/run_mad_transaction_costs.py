from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "bachelor-mad-costs-mpl"
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

DEFAULT_LOOKBACKS = list(range(5, 19))
DEFAULT_SIGNAL_VARIANTS = ["raw", "direction", "direction_risk_scaled"]


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 21,
    }
)


def _format_model_name(model_id: str) -> str:
    return {
        "ols_predictive": "OLS predictive",
        "distributed_lag": "Distributed lag",
        "elastic_net_panel": "Elastic net panel",
        "random_forest_panel": "Random forest panel",
    }.get(model_id, model_id.replace("_", " ").title())


def _save_table_tex(frame: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    path.write_text(frame.to_latex(index=False, escape=False, float_format=float_format), encoding="utf-8")


def solve_mad_weights_with_costs(
    *,
    scenarios: np.ndarray,
    expected: np.ndarray,
    previous_weights: np.ndarray,
    max_weight: float,
    transaction_cost_rate: float,
    forced_sale_cost: float,
) -> tuple[np.ndarray | None, bool]:
    scenarios = np.asarray(scenarios, dtype=float)
    n_scenarios, n_assets = scenarios.shape
    if n_assets == 0 or n_scenarios == 0:
        return None, False

    # Variables: weights n, deviations T, scenario mean 1, buys n, sells n.
    weight_start = 0
    dev_start = n_assets
    mean_idx = dev_start + n_scenarios
    buy_start = mean_idx + 1
    sell_start = buy_start + n_assets
    n_vars = sell_start + n_assets

    c = np.zeros(n_vars)
    c[dev_start : dev_start + n_scenarios] = 1.0 / n_scenarios

    a_eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []

    row = np.zeros(n_vars)
    row[weight_start:dev_start] = 1.0
    a_eq_rows.append(row)
    b_eq.append(1.0)

    row = np.zeros(n_vars)
    # Bind mu to the mean scenario portfolio return for classical mean-centered MAD.
    row[weight_start:dev_start] = scenarios.mean(axis=0)
    row[mean_idx] = -1.0
    a_eq_rows.append(row)
    b_eq.append(0.0)

    for j in range(n_assets):
        row = np.zeros(n_vars)
        row[weight_start + j] = 1.0
        row[buy_start + j] = -1.0
        row[sell_start + j] = 1.0
        a_eq_rows.append(row)
        b_eq.append(float(previous_weights[j]))

    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    for s_idx in range(n_scenarios):
        row = np.zeros(n_vars)
        row[weight_start:dev_start] = scenarios[s_idx]
        row[dev_start + s_idx] = -1.0
        row[mean_idx] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

        row = np.zeros(n_vars)
        row[weight_start:dev_start] = -scenarios[s_idx]
        row[dev_start + s_idx] = -1.0
        row[mean_idx] = 1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

    expected = np.asarray(expected, dtype=float)
    if np.isfinite(expected).all() and np.nanstd(expected) > 0:
        row = np.zeros(n_vars)
        row[weight_start:dev_start] = -expected
        row[buy_start:sell_start] = transaction_cost_rate
        row[sell_start : sell_start + n_assets] = transaction_cost_rate
        # expected @ weights - variable_cost - forced_sale_cost >= mean(expected)
        a_ub_rows.append(row)
        b_ub.append(-float(np.mean(expected)) - float(forced_sale_cost))

    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    bounds += [(0.0, None) for _ in range(n_scenarios)]
    bounds += [(None, None)]
    bounds += [(0.0, None) for _ in range(n_assets)]
    bounds += [(0.0, None) for _ in range(n_assets)]

    result = linprog(
        c,
        A_ub=np.asarray(a_ub_rows, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.asarray(a_eq_rows, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None, False

    weights = np.asarray(result.x[:n_assets], dtype=float)
    if not np.isfinite(weights).all() or abs(weights.sum() - 1.0) > 1e-5:
        return None, False
    return weights, True


def _aligned_trade_amount(new_weights: pd.Series, previous_weights: pd.Series) -> float:
    index = new_weights.index.union(previous_weights.index)
    new = new_weights.reindex(index, fill_value=0.0)
    old = previous_weights.reindex(index, fill_value=0.0)
    return float(np.abs(new - old).sum())


def backtest_mad_with_costs(
    *,
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    lookback_months: int,
    signal_variant: str,
    max_weight: float,
    transaction_cost_rate: float,
) -> dict[str, pd.DataFrame]:
    dates = sorted(actual_returns.index.intersection(predicted_returns.index))
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous_weights = pd.Series(dtype=float)
    previous_equal_weights = pd.Series(dtype=float)

    for idx, date in enumerate(dates):
        if idx < lookback_months:
            continue
        history = actual_returns.loc[dates[idx - lookback_months : idx - 1]]
        realised = actual_returns.loc[date]
        signal = transform_signal(
            signal=predicted_returns.loc[date],
            history=history,
            variant=signal_variant,
        )
        countries = sorted(
            set(history.dropna(axis=1, how="any").columns)
            .intersection(signal.dropna().index)
            .intersection(realised.dropna().index)
        )
        if len(countries) < 3:
            continue

        scenarios = history[countries].to_numpy(dtype=float)
        expected = signal[countries].to_numpy(dtype=float)
        previous_current = previous_weights.reindex(countries, fill_value=0.0)
        forced_sale_cost = transaction_cost_rate * float(previous_weights.drop(index=countries, errors="ignore").abs().sum())

        weights, optimal = solve_mad_weights_with_costs(
            scenarios=scenarios,
            expected=expected,
            previous_weights=previous_current.to_numpy(dtype=float),
            max_weight=max_weight,
            transaction_cost_rate=transaction_cost_rate,
            forced_sale_cost=forced_sale_cost,
        )
        if weights is None:
            weights = np.full(len(countries), 1.0 / len(countries))
            optimal = False

        weights_series = pd.Series(weights, index=countries)
        equal_weights_series = pd.Series(np.full(len(countries), 1.0 / len(countries)), index=countries)

        trade_amount = _aligned_trade_amount(weights_series, previous_weights)
        equal_weight_trade_amount = _aligned_trade_amount(equal_weights_series, previous_equal_weights)
        transaction_cost = transaction_cost_rate * trade_amount
        equal_weight_transaction_cost = transaction_cost_rate * equal_weight_trade_amount
        turnover = 0.5 * trade_amount
        equal_weight_turnover = 0.5 * equal_weight_trade_amount

        realised_vec = realised[countries].to_numpy(dtype=float)
        gross_return = float(np.dot(weights, realised_vec))
        equal_weight_gross_return = float(np.mean(realised_vec))
        net_return = gross_return - transaction_cost
        equal_weight_net_return = equal_weight_gross_return - equal_weight_transaction_cost

        previous_weights = weights_series
        previous_equal_weights = equal_weights_series

        return_rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_return": net_return,
                "gross_portfolio_return": gross_return,
                "transaction_cost": transaction_cost,
                "equal_weight_return": equal_weight_net_return,
                "equal_weight_gross_return": equal_weight_gross_return,
                "equal_weight_transaction_cost": equal_weight_transaction_cost,
                "excess_return": net_return - equal_weight_net_return,
                "turnover": turnover,
                "equal_weight_turnover": equal_weight_turnover,
                "total_trade_amount": trade_amount,
                "equal_weight_trade_amount": equal_weight_trade_amount,
                "optimal": bool(optimal),
                "n_assets": len(countries),
                "n_scenarios": scenarios.shape[0],
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


def run_cost_grid(
    *,
    predictions: pd.DataFrame,
    lookbacks: list[int],
    signal_variants: list[str],
    max_weight: float,
    transaction_cost_rate: float,
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
                backtest = backtest_mad_with_costs(
                    actual_returns=actual_pivot,
                    predicted_returns=pred_pivot,
                    lookback_months=lookback,
                    signal_variant=variant,
                    max_weight=max_weight,
                    transaction_cost_rate=transaction_cost_rate,
                )
                if backtest["returns"].empty:
                    continue
                returns = backtest["returns"]
                metrics = portfolio_metrics(returns, forecast_horizon_months=int(horizon))
                metrics.update(
                    {
                        "transaction_cost_rate": transaction_cost_rate,
                        "mean_transaction_cost": float(returns["transaction_cost"].mean()),
                        "total_transaction_cost": float(returns["transaction_cost"].sum()),
                        "mean_equal_weight_transaction_cost": float(returns["equal_weight_transaction_cost"].mean()),
                        "total_equal_weight_transaction_cost": float(returns["equal_weight_transaction_cost"].sum()),
                        "mean_gross_period_return": float(returns["gross_portfolio_return"].mean()),
                        "mean_net_period_return": float(returns["portfolio_return"].mean()),
                        "mean_equal_weight_turnover": float(returns["equal_weight_turnover"].mean()),
                    }
                )
                row = {
                    "model_id": model_id,
                    "forecast_horizon_months": int(horizon),
                    "signal_variant": variant,
                    "lookback_months": int(lookback),
                    **metrics,
                }
                metrics_rows.append(row)

                p = returns.copy()
                p.insert(0, "model_id", model_id)
                p.insert(1, "forecast_horizon_months", int(horizon))
                p.insert(2, "signal_variant", variant)
                p.insert(3, "lookback_months", int(lookback))
                portfolio_rows.extend(p.to_dict("records"))

                w = backtest["weights"].copy()
                w.insert(0, "model_id", model_id)
                w.insert(1, "forecast_horizon_months", int(horizon))
                w.insert(2, "signal_variant", variant)
                w.insert(3, "lookback_months", int(lookback))
                weight_rows.extend(w.to_dict("records"))

    return pd.DataFrame(metrics_rows), pd.DataFrame(portfolio_rows), pd.DataFrame(weight_rows)


def _horizon_normalized_returns(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    horizon = int(data["forecast_horizon_months"].iloc[0]) if "forecast_horizon_months" in data.columns else 1
    horizon = max(horizon, 1)
    data["portfolio_monthly_return"] = data["portfolio_return"] / horizon
    data["equal_weight_monthly_return"] = data["equal_weight_return"] / horizon
    data["gross_portfolio_monthly_return"] = data["gross_portfolio_return"] / horizon
    data["monthly_excess_return"] = data["excess_return"] / horizon
    data["mad_wealth"] = np.exp(data["portfolio_monthly_return"].cumsum())
    data["gross_mad_wealth"] = np.exp(data["gross_portfolio_monthly_return"].cumsum())
    data["equal_weight_wealth"] = np.exp(data["equal_weight_monthly_return"].cumsum())
    data["mad_drawdown"] = data["mad_wealth"] / data["mad_wealth"].cummax() - 1.0
    data["equal_weight_drawdown"] = data["equal_weight_wealth"] / data["equal_weight_wealth"].cummax() - 1.0
    return data


def _plot_top_cost_configs(metrics: pd.DataFrame, out_dir: Path) -> None:
    top = metrics.sort_values("annualized_sharpe", ascending=False).head(15).copy()
    labels = [
        f"{_format_model_name(r.model_id)}\nh={r.forecast_horizon_months}, {r.signal_variant}, L={r.lookback_months}"
        for r in top.itertuples(index=False)
    ]
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.barh(y, top["annualized_sharpe"], color="#2563eb", alpha=0.88, label="Cost-aware GNC-MAD")
    ax.scatter(top["equal_weight_annualized_sharpe"], y, color="#dc2626", s=70, label="Equal-weight after costs")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Annualized Sharpe ratio after transaction costs")
    ax.set_title("Top Cost-Aware GNC-MAD Configurations", pad=14)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_costs_top15_sharpe.png", dpi=220)
    plt.close(fig)


def _plot_cost_wealth(returns: pd.DataFrame, out_dir: Path) -> None:
    data = _horizon_normalized_returns(returns)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data["date"], data["mad_wealth"], color="#2563eb", linewidth=2.7, label="Cost-aware GNC-MAD")
    ax.plot(data["date"], data["gross_mad_wealth"], color="#2563eb", linestyle="--", linewidth=2, alpha=0.72, label="GNC-MAD before costs")
    ax.plot(data["date"], data["equal_weight_wealth"], color="#dc2626", linewidth=2.4, label="Equal weight after costs")
    ax.set_ylabel("Cumulative wealth, horizon-normalized")
    ax.set_title("Cost-Aware GNC-MAD Portfolio vs Equal Weight", pad=14)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_costs_wealth_vs_equal_weight.png", dpi=220)
    plt.close(fig)


def _plot_transaction_costs_over_time(returns: pd.DataFrame, out_dir: Path) -> None:
    data = returns.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    data["cost_ma_6"] = data["transaction_cost"].rolling(6, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(data["date"], data["transaction_cost"], width=23, color="#f97316", alpha=0.75, label="GNC-MAD transaction cost")
    ax.plot(data["date"], data["cost_ma_6"], color="#111827", linewidth=2.6, label="6-month moving average")
    ax.plot(data["date"], data["equal_weight_transaction_cost"], color="#dc2626", linewidth=2, label="Equal-weight transaction cost")
    ax.set_ylabel("Transaction cost, log-return units")
    ax.set_title("Transaction Costs Over Time", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_costs_transaction_costs_over_time.png", dpi=220)
    plt.close(fig)


def _plot_cost_vs_nocost_by_model(cost_metrics: pd.DataFrame, no_cost_metrics: pd.DataFrame, out_dir: Path) -> None:
    cost_best = (
        cost_metrics.sort_values("annualized_sharpe", ascending=False)
        .groupby("model_id", as_index=False)
        .head(1)
        .sort_values("annualized_sharpe", ascending=False)
    )
    no_cost_best = (
        no_cost_metrics.sort_values("annualized_sharpe", ascending=False)
        .groupby("model_id", as_index=False)
        .head(1)
        .sort_values("annualized_sharpe", ascending=False)
    )
    compare = cost_best[
        [
            "model_id",
            "forecast_horizon_months",
            "signal_variant",
            "lookback_months",
            "annualized_sharpe",
            "compound_return",
            "max_drawdown",
            "mean_turnover",
            "mean_transaction_cost",
            "total_transaction_cost",
        ]
    ].merge(
        no_cost_best[
            [
                "model_id",
                "annualized_sharpe",
                "compound_return",
                "max_drawdown",
                "mean_turnover",
            ]
        ],
        on="model_id",
        suffixes=("_after_costs", "_without_costs"),
    )
    compare["model_label"] = compare["model_id"].map(_format_model_name)
    compare["sharpe_cost_drag"] = compare["annualized_sharpe_after_costs"] - compare["annualized_sharpe_without_costs"]
    compare.to_csv(out_dir / "gnc_mad_cost_vs_nocost_by_model.csv", index=False)

    x = np.arange(len(compare))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.bar(x - width / 2, compare["annualized_sharpe_without_costs"], width=width, color="#94a3b8", label="Without costs")
    ax.bar(x + width / 2, compare["annualized_sharpe_after_costs"], width=width, color="#2563eb", label="After costs")
    ax.set_xticks(x)
    ax.set_xticklabels(compare["model_label"], rotation=15, ha="right")
    ax.set_ylabel("Best annualized Sharpe ratio")
    ax.set_title("Cost Drag by Forecast Method", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_cost_drag_by_forecast_method.png", dpi=220)
    plt.close(fig)

    return compare


def _plot_turnover_vs_cost(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for model_id, group in metrics.groupby("model_id"):
        ax.scatter(
            group["mean_turnover"],
            group["mean_transaction_cost"],
            s=38 + 16 * group["forecast_horizon_months"],
            alpha=0.62,
            label=_format_model_name(model_id),
        )
    ax.set_xlabel("Mean turnover")
    ax.set_ylabel("Mean transaction cost")
    ax.set_title("Turnover and Transaction Cost Across Cost-Aware MAD Configurations", pad=14)
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "gnc_mad_costs_turnover_vs_transaction_cost.png", dpi=220)
    plt.close(fig)


def build_cost_report(base_dir: Path, out_dir: Path, max_weight: float, transaction_cost_rate: float) -> None:
    predictions = pd.read_csv(base_dir / "all_predictions_eval_window.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, returns, weights = run_cost_grid(
        predictions=predictions,
        lookbacks=DEFAULT_LOOKBACKS,
        signal_variants=DEFAULT_SIGNAL_VARIANTS,
        max_weight=max_weight,
        transaction_cost_rate=transaction_cost_rate,
    )

    metrics.to_csv(out_dir / "gnc_mad_costs_all_config_metrics.csv", index=False)
    returns.to_csv(out_dir / "gnc_mad_costs_portfolio_returns.csv", index=False)
    weights.to_csv(out_dir / "gnc_mad_costs_weights.csv", index=False)

    top = metrics.sort_values("annualized_sharpe", ascending=False).reset_index(drop=True)
    top.head(20).to_csv(out_dir / "gnc_mad_costs_top20_configurations.csv", index=False)

    best_by_model = (
        metrics.sort_values("annualized_sharpe", ascending=False)
        .groupby("model_id", as_index=False)
        .head(1)
        .sort_values("annualized_sharpe", ascending=False)
    )
    best_by_model.to_csv(out_dir / "gnc_mad_costs_best_by_model.csv", index=False)

    table = top.head(10)[
        [
            "model_id",
            "forecast_horizon_months",
            "signal_variant",
            "lookback_months",
            "annualized_sharpe",
            "equal_weight_annualized_sharpe",
            "compound_return",
            "equal_weight_compound_return",
            "max_drawdown",
            "mean_turnover",
            "mean_transaction_cost",
        ]
    ].copy()
    table["model_id"] = table["model_id"].map(_format_model_name)
    _save_table_tex(table, out_dir / "table_gnc_mad_costs_top10_configurations.tex")

    best = top.iloc[0]
    best_filter = (
        (returns["model_id"] == best["model_id"])
        & (returns["forecast_horizon_months"] == best["forecast_horizon_months"])
        & (returns["signal_variant"] == best["signal_variant"])
        & (returns["lookback_months"] == best["lookback_months"])
    )
    best_returns = returns[best_filter].copy()
    best_weights = weights[
        (weights["model_id"] == best["model_id"])
        & (weights["forecast_horizon_months"] == best["forecast_horizon_months"])
        & (weights["signal_variant"] == best["signal_variant"])
        & (weights["lookback_months"] == best["lookback_months"])
    ].copy()
    best_returns.to_csv(out_dir / "gnc_mad_costs_best_config_returns.csv", index=False)
    best_weights.to_csv(out_dir / "gnc_mad_costs_best_config_weights.csv", index=False)

    no_cost_metrics_path = base_dir / "mad_portfolio_cleaned/report_assets_gnc_forecasts/gnc_mad_all_config_metrics.csv"
    if no_cost_metrics_path.exists():
        no_cost_metrics = pd.read_csv(no_cost_metrics_path)
        compare = _plot_cost_vs_nocost_by_model(metrics, no_cost_metrics, out_dir)
        _save_table_tex(
            compare[
                [
                    "model_label",
                    "annualized_sharpe_without_costs",
                    "annualized_sharpe_after_costs",
                    "sharpe_cost_drag",
                    "compound_return_after_costs",
                    "mean_transaction_cost",
                    "total_transaction_cost",
                ]
            ],
            out_dir / "table_gnc_mad_cost_drag_by_forecast_method.tex",
        )

    _plot_top_cost_configs(metrics, out_dir)
    _plot_cost_wealth(best_returns, out_dir)
    _plot_transaction_costs_over_time(best_returns, out_dir)
    _plot_turnover_vs_cost(metrics, out_dir)

    manifest = {
        "source_suite": str(base_dir),
        "included_models": GNC_MODELS,
        "transaction_cost_rate": transaction_cost_rate,
        "transaction_cost_source": "Fagprojekt implementation: PPC = 0.0015 proportional transaction cost",
        "transaction_cost_model": (
            "Costs are deducted from the forecast-signal constraint using buy/sell auxiliary variables "
            "and from realized backtest returns using cost_rate * sum(abs(delta_weights))."
        ),
        "max_weight": max_weight,
        "lookbacks": DEFAULT_LOOKBACKS,
        "signal_variants": DEFAULT_SIGNAL_VARIANTS,
        "best_configuration": {
            "model_id": str(best["model_id"]),
            "forecast_horizon_months": int(best["forecast_horizon_months"]),
            "signal_variant": str(best["signal_variant"]),
            "lookback_months": int(best["lookback_months"]),
            "annualized_sharpe_after_costs": float(best["annualized_sharpe"]),
            "equal_weight_annualized_sharpe_after_costs": float(best["equal_weight_annualized_sharpe"]),
            "compound_return_after_costs": float(best["compound_return"]),
            "equal_weight_compound_return_after_costs": float(best["equal_weight_compound_return"]),
            "max_drawdown_after_costs": float(best["max_drawdown"]),
            "mean_turnover": float(best["mean_turnover"]),
            "mean_transaction_cost": float(best["mean_transaction_cost"]),
            "total_transaction_cost": float(best["total_transaction_cost"]),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = [
        "# Cost-Aware GNC-MAD Portfolio Report",
        "",
        "This run applies the transaction cost approach from the previous portfolio optimization project.",
        f"The proportional cost rate is `{transaction_cost_rate}`.",
        "",
        "The optimizer introduces buy/sell variables for changes in portfolio weights and deducts transaction costs from the forecast-signal floor.",
        "Backtest returns are reported net of transaction costs.",
        "",
        "Key files:",
        "- `gnc_mad_costs_all_config_metrics.csv`",
        "- `gnc_mad_costs_portfolio_returns.csv`",
        "- `gnc_mad_costs_weights.csv`",
        "- `table_gnc_mad_costs_top10_configurations.tex`",
        "- `gnc_mad_costs_top15_sharpe.png`",
        "- `gnc_mad_costs_wealth_vs_equal_weight.png`",
        "- `gnc_mad_cost_drag_by_forecast_method.png`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNC-MAD portfolio optimization with transaction costs.")
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
            "mad_portfolio_costs_0015/report_assets"
        ),
    )
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--transaction-cost-rate", type=float, default=0.0015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_cost_report(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        max_weight=args.max_weight,
        transaction_cost_rate=args.transaction_cost_rate,
    )


if __name__ == "__main__":
    main()
