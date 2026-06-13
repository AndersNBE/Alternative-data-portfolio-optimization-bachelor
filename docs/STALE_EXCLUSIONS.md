# Stale Exclusions

The clean repository intentionally excludes code and artifacts that were useful historically but must not drive final report claims.

Excluded as final evidence:

- `7fd2530` downstream rows.
- `tau06` / tau `0.6` downstream results.
- `report_artifacts/treasury_sharpe_2026-06-05/`.
- `cleaned_return_suite_20260522_v2`.
- stale Treasury/MAD scripts such as `recompute_treasury_adjusted_sharpe.py` and `build_mad_gnc_report.py`.
- old ROI5 segmentation metrics unless regenerated and pinned through the current supervised QA role.
- old port/country-month counts `4,795`, `4,823`, `2,238`, and `2,325`.

The active boundary is enforced by:

```text
pipelines/final_code_manifest.csv
pipelines/final_artifact_manifest.csv
pipelines/final_report_claims.csv
tests/test_final_pipeline_lock.py
```
