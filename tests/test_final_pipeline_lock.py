from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
TRACKER = REPO / "pipelines" / "FINAL_PIPELINE_TRACKER.md"
CLAIMS = REPO / "pipelines" / "final_report_claims.csv"
ARTIFACTS = REPO / "pipelines" / "final_artifact_manifest.csv"
CODE = REPO / "pipelines" / "final_code_manifest.csv"
SELECTED_MODEL = REPO / "pipelines" / "final_selected_model_tau04_bd5d48e.json"
CB031AC = "cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34"
ONEB6C343 = "1b6c343467e7ea73e13554e65316a2fb3b642694"
BD5D48E = "bd5d48e991e362f5114797622be8ca4b622ea0f2"
SUPERVISED_SUMMARY = "report_artifacts/supervised_tau04_checks_cb031ac_8a44dd0/summary_metrics.csv"
MAD_METRICS = (
    "final_runs/tau04_hk_28623539/return_forecasting/"
    "usd19_no_sri_lanka_28623539/mad_portfolio_cleaned/"
    "portfolio_metrics_treasury_adjusted.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO / path


def local_csv(rel_path: str) -> pd.DataFrame:
    path = local_path(rel_path)
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        raise AssertionError(f"{rel_path} is an unresolved Git LFS pointer")
    return pd.read_csv(path)


def row_count(ref: str) -> int:
    path = local_path(ref)
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_schemas_and_tracker_command_ids() -> None:
    assert TRACKER.exists()
    tracker_text = TRACKER.read_text(encoding="utf-8")

    expected_claim_header = [
        "claim_id",
        "report_surface",
        "expected_value",
        "source_role",
        "source_path",
        "extraction_rule",
        "command_id",
        "tolerance",
    ]
    expected_artifact_header = [
        "artifact_id",
        "stage",
        "role",
        "path_or_git_ref",
        "sha256_or_row_count",
        "required_for_claims",
    ]
    expected_code_header = ["path", "status", "stage", "reason", "owning_command_id"]

    for path, expected in [
        (CLAIMS, expected_claim_header),
        (ARTIFACTS, expected_artifact_header),
        (CODE, expected_code_header),
    ]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            assert next(reader) == expected

    for row in read_rows(CLAIMS):
        assert row["claim_id"]
        assert row["command_id"] in tracker_text


def test_claim_source_paths_exist() -> None:
    for row in read_rows(CLAIMS):
        ref = row["source_path"]
        assert local_path(ref).exists(), row


def test_active_code_manifest_paths_exist_and_stale_is_not_active() -> None:
    stale_needles = [
        "7fd2530",
        "tau06",
        "treasury_sharpe_2026-06-05",
        "cleaned_return_suite_20260522_v2",
        "gnc_ablation_20260522",
    ]
    allowed_statuses = {"active", "support", "historical", "deletable_candidate"}
    for row in read_rows(CODE):
        assert row["status"] in allowed_statuses
        path = local_path(row["path"])
        assert path.exists(), row
        text = " ".join(row.values())
        if row["status"] == "active":
            assert not any(needle in text for needle in stale_needles), row


def test_active_python_entrypoints_do_not_hardcode_private_paths_or_old_defaults() -> None:
    forbidden = [
        "/Users/",
        "/private/tmp",
        "models/ml/unet/weights/selected_model.json",
        "container_index_cleaned_local/03_bce_dice_group_100ep_cleaned_28413555",
    ]
    for row in read_rows(CODE):
        if row["status"] != "active":
            continue
        path = local_path(row["path"])
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(needle in text for needle in forbidden), row


def test_final_forecast_runners_default_to_locked_tau04_inputs() -> None:
    cleaned_runner = (REPO / "analysis" / "return_forecasting" / "run_cleaned_research_suite.py").read_text(
        encoding="utf-8"
    )
    ablation_runner = (REPO / "analysis" / "return_forecasting" / "run_gnc_ablation.py").read_text(
        encoding="utf-8"
    )
    assert 'DEFAULT_CLEANED_BUNDLE = Path("final_runs/tau04_hk_28623539/daily_container_index")' in cleaned_runner
    assert 'DEFAULT_SELECTED_MODEL_JSON = Path("pipelines/final_selected_model_tau04_bd5d48e.json")' in cleaned_runner
    assert "DEFAULT_SELECTED_MODEL_JSON" in ablation_runner


def test_artifact_manifest_paths_and_row_counts() -> None:
    for row in read_rows(ARTIFACTS):
        ref = row["path_or_git_ref"]
        assert local_path(ref).exists(), row

        match = re.search(r"rows=(\d+)", row["sha256_or_row_count"])
        if match:
            assert row_count(ref) == int(match.group(1)), row


def test_large_required_artifacts_are_materialized_or_lfs_tracked() -> None:
    checkpoint = REPO / "report_regen_2026-06-11" / "best.pt"
    weights = REPO / MAD_METRICS.replace("portfolio_metrics_treasury_adjusted.csv", "weights.csv")
    assert checkpoint.stat().st_size > 100_000_000
    assert weights.stat().st_size > 100_000_000
    assert sha256(checkpoint) == "876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc"
    attributes = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "*.pt filter=lfs" in attributes
    assert "**/weights.csv filter=lfs" in attributes


def test_locked_mad_claims_match_bd5d48e() -> None:
    metrics = local_csv(MAD_METRICS)
    gnc = metrics[
        metrics["model_id"].isin(
            ["distributed_lag", "elastic_net_panel", "ols_predictive", "random_forest_panel"]
        )
    ].copy()
    assert len(metrics) == 1260
    assert len(gnc) == 840

    best = gnc.sort_values("annualized_sharpe_treasury_excess", ascending=False).iloc[0]
    hist = metrics[metrics["model_id"] == "historical_mean"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]
    always = metrics[metrics["model_id"] == "always_positive"].sort_values(
        "annualized_sharpe_treasury_excess", ascending=False
    ).iloc[0]

    assert best["model_id"] == "distributed_lag"
    assert int(best["forecast_horizon_months"]) == 1
    assert best["signal_variant"] == "raw"
    assert int(best["lookback_months"]) == 5
    assert abs(best["annualized_sharpe_treasury_excess"] - 0.915242569039211) < 1e-12
    assert abs(best["equal_weight_annualized_sharpe_treasury_excess"] - 0.5485295839657542) < 1e-12
    assert abs(hist["annualized_sharpe_treasury_excess"] - 0.8300625413664386) < 1e-12
    assert abs(always["annualized_sharpe_treasury_excess"] - 0.7689949639514889) < 1e-12


def test_locked_segmentation_claims_match_1b6c343() -> None:
    summary = local_csv(SUPERVISED_SUMMARY)
    column_for_metric = {
        "dice": "avg_dice",
        "iou": "avg_iou",
        "precision": "avg_precision",
        "recall": "avg_recall",
    }
    claims = {
        row["claim_id"]: row
        for row in read_rows(CLAIMS)
        if row["claim_id"].startswith("seg_") and len(row["claim_id"].split("_")) == 4
    }
    assert len(claims) == 16
    for claim_id, row in claims.items():
        _, run_type, split, metric = claim_id.rsplit("_", 3)
        source_row = summary[(summary["run_type"] == run_type) & (summary["split"] == split)]
        assert len(source_row) == 1, claim_id
        observed = float(source_row.iloc[0][column_for_metric[metric]])
        expected = float(row["expected_value"])
        tolerance = float(row["tolerance"])
        assert abs(observed - expected) <= tolerance, claim_id


def test_final_selected_model_lock_file() -> None:
    data = pd.read_json(SELECTED_MODEL, typ="series")
    assert data["source_commit"] == BD5D48E
    assert data["threshold"] == 0.4
    assert data["checkpoint_sha256"] == "876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc"


def test_final_entrypoints_exist() -> None:
    assert (REPO / "analysis" / "return_forecasting" / "run_final_usd19_mad_suite.py").exists()
    assert (REPO / "analysis" / "return_forecasting" / "apply_final_treasury_adjustment.py").exists()
    assert (REPO / "analysis" / "return_forecasting" / "build_final_mad_tables.py").exists()
    assert (REPO / "report_regen_2026-06-11" / "assemble_final_figures.py").exists()
