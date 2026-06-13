from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONFIG_COLS = ["model_id", "forecast_horizon_months", "signal_variant", "lookback_months"]


def load_monthly_risk_free(path: Path) -> dict[pd.Period, float]:
    rf = pd.read_csv(path)
    required = {"month", "treasury_1mo_monthly_log_return"}
    missing = required.difference(rf.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    rf["month_period"] = pd.PeriodIndex(rf["month"], freq="M")
    return dict(zip(rf["month_period"], rf["treasury_1mo_monthly_log_return"]))


def period_risk_free(row: pd.Series, rf_map: dict[pd.Period, float]) -> float:
    horizon = int(row["forecast_horizon_months"])
    months = pd.period_range(row["date"], periods=horizon, freq="M")
    values = [rf_map.get(month, np.nan) for month in months]
    if any(pd.isna(values)):
        return float("nan")
    return float(np.sum(values))


def max_drawdown(cumulative_log_return: pd.Series) -> float:
    wealth = np.exp(cumulative_log_return)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else float("nan")


def compute_treasury_adjusted_metrics(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    rf_map: dict[pd.Period, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = metrics.copy()
    returns = returns.copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.to_period("M")
    returns["forecast_horizon_months"] = returns["forecast_horizon_months"].astype(int)
    returns["lookback_months"] = returns["lookback_months"].astype(int)
    metrics["forecast_horizon_months"] = metrics["forecast_horizon_months"].astype(int)
    metrics["lookback_months"] = metrics["lookback_months"].astype(int)
    returns["treasury_rf_period_log_return"] = returns.apply(period_risk_free, axis=1, rf_map=rf_map)

    metric_lookup = metrics.set_index(CONFIG_COLS)
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for keys, group in returns.groupby(CONFIG_COLS, dropna=False):
        group = group.sort_values("date").copy()
        horizon = int(keys[1])
        valid = group.dropna(
            subset=["portfolio_return", "equal_weight_return", "treasury_rf_period_log_return"]
        ).copy()
        quality_rows.append(
            {
                "model_id": keys[0],
                "forecast_horizon_months": horizon,
                "signal_variant": keys[2],
                "lookback_months": int(keys[3]),
                "return_rows": int(len(group)),
                "missing_rf_rows": int(group["treasury_rf_period_log_return"].isna().sum()),
            }
        )
        if valid.empty:
            continue

        port_period_excess_rf = pd.to_numeric(valid["portfolio_return"], errors="coerce") - valid[
            "treasury_rf_period_log_return"
        ]
        ew_period_excess_rf = pd.to_numeric(valid["equal_weight_return"], errors="coerce") - valid[
            "treasury_rf_period_log_return"
        ]
        port_period_vol_rf = float(port_period_excess_rf.std(ddof=0))
        ew_period_vol_rf = float(ew_period_excess_rf.std(ddof=0))
        port_monthly_excess_rf = port_period_excess_rf / horizon
        ew_monthly_excess_rf = ew_period_excess_rf / horizon
        port_monthly_vol_rf = port_period_vol_rf / math.sqrt(horizon) if port_period_vol_rf > 0 else float("nan")
        ew_monthly_vol_rf = ew_period_vol_rf / math.sqrt(horizon) if ew_period_vol_rf > 0 else float("nan")
        port_sharpe_rf = (
            math.sqrt(12.0) * float(port_monthly_excess_rf.mean()) / port_monthly_vol_rf
            if np.isfinite(port_monthly_vol_rf) and port_monthly_vol_rf > 0
            else float("nan")
        )
        ew_sharpe_rf = (
            math.sqrt(12.0) * float(ew_monthly_excess_rf.mean()) / ew_monthly_vol_rf
            if np.isfinite(ew_monthly_vol_rf) and ew_monthly_vol_rf > 0
            else float("nan")
        )
        cumulative_rf = port_monthly_excess_rf.cumsum()
        ew_cumulative_rf = ew_monthly_excess_rf.cumsum()
        old = metric_lookup.loc[keys].to_dict() if keys in metric_lookup.index else {}

        rows.append(
            {
                "model_id": keys[0],
                "forecast_horizon_months": horizon,
                "signal_variant": keys[2],
                "lookback_months": int(keys[3]),
                "n_periods": int(len(group)),
                "n_periods_with_rf": int(len(valid)),
                "date_min": str(valid["date"].min()),
                "date_max": str(valid["date"].max()),
                "annualized_sharpe_original": float(old.get("annualized_sharpe", np.nan)),
                "annualized_sharpe_treasury_excess": float(port_sharpe_rf),
                "equal_weight_annualized_sharpe_original": float(
                    old.get("equal_weight_annualized_sharpe", np.nan)
                ),
                "equal_weight_annualized_sharpe_treasury_excess": float(ew_sharpe_rf),
                "mean_period_return_original": float(old.get("mean_period_return", np.nan)),
                "mean_period_rf_log_return": float(valid["treasury_rf_period_log_return"].mean()),
                "mean_monthly_return_original": float(old.get("mean_monthly_return", np.nan)),
                "mean_monthly_excess_return_treasury": float(port_monthly_excess_rf.mean()),
                "monthly_vol_treasury_excess": float(port_monthly_vol_rf),
                "equal_weight_mean_monthly_excess_return_treasury": float(ew_monthly_excess_rf.mean()),
                "equal_weight_monthly_vol_treasury_excess": float(ew_monthly_vol_rf),
                "compound_return_original": float(old.get("compound_return", np.nan)),
                "compound_excess_return_treasury": float(np.exp(cumulative_rf.iloc[-1]) - 1.0),
                "equal_weight_compound_return_original": float(old.get("equal_weight_compound_return", np.nan)),
                "equal_weight_compound_excess_return_treasury": float(np.exp(ew_cumulative_rf.iloc[-1]) - 1.0),
                "max_drawdown_original": float(old.get("max_drawdown", np.nan)),
                "max_drawdown_excess_treasury": max_drawdown(cumulative_rf),
                "equal_weight_max_drawdown_original": float(old.get("equal_weight_max_drawdown", np.nan)),
                "equal_weight_max_drawdown_excess_treasury": max_drawdown(ew_cumulative_rf),
                "mean_excess_vs_equal_weight": float(old.get("mean_excess_vs_equal_weight", np.nan)),
                "hit_rate_vs_equal_weight": float(old.get("hit_rate_vs_equal_weight", np.nan)),
                "mean_turnover": float(old.get("mean_turnover", np.nan)),
                "optimal_solver_rate": float(old.get("optimal_solver_rate", np.nan)),
                "mean_n_assets": float(old.get("mean_n_assets", np.nan)),
                "mean_n_scenarios": float(old.get("mean_n_scenarios", np.nan)),
            }
        )

    out = pd.DataFrame(rows).sort_values("annualized_sharpe_treasury_excess", ascending=False)
    quality = pd.DataFrame(quality_rows)
    return out, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the final Treasury excess-return adjustment to a MAD suite.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--risk-free-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mad_dir = args.base_dir / "mad_portfolio_cleaned"
    out_dir = args.out_dir or mad_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(mad_dir / "portfolio_metrics.csv")
    returns = pd.read_csv(mad_dir / "portfolio_returns.csv")
    rf_map = load_monthly_risk_free(args.risk_free_csv)
    adjusted, quality = compute_treasury_adjusted_metrics(metrics, returns, rf_map)

    adjusted.to_csv(out_dir / "portfolio_metrics_treasury_adjusted.csv", index=False)
    adjusted.head(20).to_csv(out_dir / "top20_treasury_adjusted.csv", index=False)
    quality.to_csv(out_dir / "treasury_adjustment_quality_checks.csv", index=False)

    best = adjusted.iloc[0].to_dict()
    summary = {
        "source_suite": str(args.base_dir),
        "risk_free_csv": str(args.risk_free_csv),
        "rows": int(len(adjusted)),
        "missing_rf_rows_total": int(quality["missing_rf_rows"].sum()) if not quality.empty else 0,
        "best_configuration": {
            "model_id": best["model_id"],
            "forecast_horizon_months": int(best["forecast_horizon_months"]),
            "signal_variant": best["signal_variant"],
            "lookback_months": int(best["lookback_months"]),
            "annualized_sharpe_treasury_excess": float(best["annualized_sharpe_treasury_excess"]),
            "equal_weight_annualized_sharpe_treasury_excess": float(
                best["equal_weight_annualized_sharpe_treasury_excess"]
            ),
        },
    }
    (out_dir / "treasury_adjusted_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "treasury_adjusted_summary.md").write_text(
        "\n".join(
            [
                "# Treasury-Adjusted MAD Summary",
                "",
                f"- Source suite: `{args.base_dir}`",
                f"- Risk-free CSV: `{args.risk_free_csv}`",
                f"- Adjusted rows: `{len(adjusted)}`",
                f"- Missing risk-free rows: `{summary['missing_rf_rows_total']}`",
                f"- Best configuration: `{best['model_id']}`, h={int(best['forecast_horizon_months'])}, "
                f"{best['signal_variant']}, L={int(best['lookback_months'])}",
                f"- Treasury-adjusted Sharpe: `{best['annualized_sharpe_treasury_excess']:.6f}`",
                f"- Matched equal-weight Treasury-adjusted Sharpe: "
                f"`{best['equal_weight_annualized_sharpe_treasury_excess']:.6f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(out_dir / "portfolio_metrics_treasury_adjusted.csv")


if __name__ == "__main__":
    main()
