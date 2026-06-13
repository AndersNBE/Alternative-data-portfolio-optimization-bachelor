from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_COMMIT = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
FINAL_RUN_ROOT = "final_runs/tau04_hk_28623539"
SUITE_REL = f"{FINAL_RUN_ROOT}/return_forecasting/usd19_no_sri_lanka_28623539"
MAD_REL = f"{SUITE_REL}/mad_portfolio_cleaned"

ESSENTIAL_FILES = [
    "config.json",
    "input_fingerprints.json",
    "combined_metrics.csv",
    "summary.md",
    "inputs/country_fx_rates_usd_with_hong_kong.csv",
    "inputs/country_fx_rates_usd_with_hong_kong.json",
    "inputs/treasury_1mo_monthly_rf_used.csv",
    "mad_portfolio_cleaned/manifest.json",
    "mad_portfolio_cleaned/portfolio_metrics.csv",
    "mad_portfolio_cleaned/portfolio_metrics_treasury_adjusted.csv",
    "mad_portfolio_cleaned/portfolio_returns.csv",
    "mad_portfolio_cleaned/summary.md",
    "mad_portfolio_cleaned/top20_treasury_adjusted.csv",
    "mad_portfolio_cleaned/treasury_adjusted_summary.md",
]

EXPECTED = {
    "best_model_id": "distributed_lag",
    "best_forecast_horizon_months": 1,
    "best_signal_variant": "raw",
    "best_lookback_months": 5,
    "best_treasury_sharpe": 0.915242569039211,
    "matched_equal_weight_treasury_sharpe": 0.5485295839657542,
    "historical_mean_no_gnc_best_treasury_sharpe": 0.8300625413664386,
    "always_positive_best_treasury_sharpe": 0.7689949639514889,
    "raw_compound_return": 1.5674287724736669,
    "equal_weight_raw_compound_return": 0.9106598159591603,
    "raw_max_drawdown": -0.1944156652029075,
    "equal_weight_raw_max_drawdown": -0.2359735943986396,
    "mean_turnover": 0.5257052515076746,
}

GNC_MODELS = {
    "ols_predictive",
    "distributed_lag",
    "elastic_net_panel",
    "random_forest_panel",
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def artifact_csv(repo_root: Path, commit: str, rel_path: str) -> pd.DataFrame:
    raw = artifact_blob(repo_root, commit, rel_path)
    if is_lfs_pointer(raw):
        raise RuntimeError(f"{rel_path} is a Git LFS pointer; materialize LFS before using it as CSV input.")
    return pd.read_csv(io.BytesIO(raw))


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(f"{name}: expected {expected}, observed {observed}")


def validate_locked_suite(repo_root: Path, commit: str, tolerance: float) -> dict[str, Any]:
    metrics = artifact_csv(repo_root, commit, f"{MAD_REL}/portfolio_metrics_treasury_adjusted.csv")
    gnc_metrics = metrics[metrics["model_id"].isin(GNC_MODELS)].copy()
    if len(gnc_metrics) != 840:
        raise AssertionError(f"Expected 840 GNC-informed MAD rows, observed {len(gnc_metrics)}")
    if len(metrics) != 1260:
        raise AssertionError(f"Expected 1260 total MAD metric rows, observed {len(metrics)}")

    best = gnc_metrics.sort_values("annualized_sharpe_treasury_excess", ascending=False).iloc[0]
    hist = metrics[metrics["model_id"] == "historical_mean"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]
    always = metrics[metrics["model_id"] == "always_positive"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]

    checks = {
        "best_treasury_sharpe": best["annualized_sharpe_treasury_excess"],
        "matched_equal_weight_treasury_sharpe": best["equal_weight_annualized_sharpe_treasury_excess"],
        "historical_mean_no_gnc_best_treasury_sharpe": hist["annualized_sharpe_treasury_excess"],
        "always_positive_best_treasury_sharpe": always["annualized_sharpe_treasury_excess"],
        "raw_compound_return": best["compound_return_original"],
        "equal_weight_raw_compound_return": best["equal_weight_compound_return_original"],
        "raw_max_drawdown": best["max_drawdown_original"],
        "equal_weight_raw_max_drawdown": best["equal_weight_max_drawdown_original"],
        "mean_turnover": best["mean_turnover"],
    }
    for name, observed in checks.items():
        assert_close(name, float(observed), float(EXPECTED[name]), tolerance)

    if str(best["model_id"]) != EXPECTED["best_model_id"]:
        raise AssertionError(f"Expected best model {EXPECTED['best_model_id']}, observed {best['model_id']}")
    if int(best["forecast_horizon_months"]) != EXPECTED["best_forecast_horizon_months"]:
        raise AssertionError("Unexpected best forecast horizon")
    if str(best["signal_variant"]) != EXPECTED["best_signal_variant"]:
        raise AssertionError("Unexpected best signal variant")
    if int(best["lookback_months"]) != EXPECTED["best_lookback_months"]:
        raise AssertionError("Unexpected best lookback")

    combined = artifact_csv(repo_root, commit, f"{SUITE_REL}/combined_metrics.csv")
    if len(combined) != 30:
        raise AssertionError(f"Expected 30 combined forecast metric rows, observed {len(combined)}")

    return {
        "source_commit": commit,
        "suite": SUITE_REL,
        "mad_rows_total": int(len(metrics)),
        "mad_rows_gnc_informed": int(len(gnc_metrics)),
        "combined_metrics_rows": int(len(combined)),
        "best_configuration": {
            "model_id": str(best["model_id"]),
            "forecast_horizon_months": int(best["forecast_horizon_months"]),
            "signal_variant": str(best["signal_variant"]),
            "lookback_months": int(best["lookback_months"]),
            "annualized_sharpe_treasury_excess": float(best["annualized_sharpe_treasury_excess"]),
            "equal_weight_annualized_sharpe_treasury_excess": float(
                best["equal_weight_annualized_sharpe_treasury_excess"]
            ),
            "compound_return_original": float(best["compound_return_original"]),
            "max_drawdown_original": float(best["max_drawdown_original"]),
            "mean_turnover": float(best["mean_turnover"]),
        },
        "baseline_checks": {
            "historical_mean_no_gnc_best_treasury_sharpe": float(hist["annualized_sharpe_treasury_excess"]),
            "always_positive_best_treasury_sharpe": float(always["annualized_sharpe_treasury_excess"]),
        },
    }


def materialize_essential_files(repo_root: Path, commit: str, out_dir: Path) -> list[dict[str, Any]]:
    written = []
    for rel in ESSENTIAL_FILES:
        source_rel = f"{SUITE_REL}/{rel}"
        raw = artifact_blob(repo_root, commit, source_rel)
        if is_lfs_pointer(raw):
            raise RuntimeError(f"{source_rel} is an unresolved Git LFS pointer.")
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        written.append(
            {
                "relative_path": rel,
                "source": f"{commit}:{source_rel}",
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and validate the locked bd5d48e USD19/no-Sri-Lanka MAD suite. "
            "This is the final frozen-mode entrypoint; raw refreshes are documented separately."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/outputs/return_forecasting/usd19_no_sri_lanka_28623539_locked"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    validation = validate_locked_suite(repo_root, args.source_commit, args.tolerance)
    written: list[dict[str, Any]] = []
    if not args.validate_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written = materialize_essential_files(repo_root, args.source_commit, args.out_dir)
        manifest = {
            "mode": "frozen_exact",
            "validation": validation,
            "written_files": written,
            "note": "Git LFS-only files such as weights.csv must be materialized separately when weight tables are regenerated.",
        }
        (args.out_dir / "LOCKED_SUITE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"validation": validation, "written_files": len(written)}, indent=2))


if __name__ == "__main__":
    main()
