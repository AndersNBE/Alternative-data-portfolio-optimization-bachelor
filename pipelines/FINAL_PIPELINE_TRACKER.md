# Final Pipeline Tracker

Purpose: this is the locked runbook for reproducing the evidence behind the
report. It covers data refresh, frozen exact reproduction, QA, segmentation,
full inference, container/GNC construction, return forecasting, MAD, tables and
figures. The short conflict-resolution rule remains `../SOURCE_OF_TRUTH.md`.

Do not use the local Overleaf checkout as evidence. It can be behind. Do not
edit Overleaf from this pipeline. The report text consumes outputs from this
tracker; it is not the source of truth for the numbers.

This clean repository materializes the frozen evidence locally. Commit IDs are
kept as provenance labels; commands read the checked-out files in this repo.

Supporting manifests:

```text
pipelines/final_report_claims.csv
pipelines/final_artifact_manifest.csv
pipelines/final_code_manifest.csv
```

## Reproduction Boundary

Two modes are valid:

```text
frozen exact mode:
  Reproduce report numbers from pinned commits and locked artifacts. This is
  the only mode that may be used to check final report claims.

refresh mode:
  Re-download or recompute upstream data with the documented commands and then
  compare against the frozen exact outputs. Refresh mode is allowed to differ
  because Sentinel, market, FX and Treasury sources can change.
```

The final source roles are fixed:

| Role | Commit / artifact | Use for |
| --- | --- | --- |
| U-Net model package provenance | `cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34` | checkpoint identity, model package, split files, pair contract, training metadata |
| Report-facing supervised U-Net metrics | `1b6c343467e7ea73e13554e65316a2fb3b642694` | validation/test Dice, IoU, precision, recall, threshold sweeps and ROI10 metrics against masks |
| Full inference and downstream results | `bd5d48e991e362f5114797622be8ca4b622ea0f2` | full corpus inference, operational ROI, NC/GNC, return forecasting, USD/no-Sri-Lanka finance, Treasury-adjusted MAD |

Short rule:

```text
Model package/provenance -> cb031ac
Supervised segmentation metrics with masks -> 1b6c343
Everything from full inference onward -> bd5d48e
```

The three roles share the selected checkpoint:

```text
876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc
```

## Command IDs

Run from workspace root unless a command says otherwise.

### `control-lock`

Validates the tracker, manifests and stale-code boundary.

```bash
python3 -m pytest tests/test_return_forecasting.py tests/test_final_pipeline_lock.py
```

Acceptance:

```text
All manifests have the required schema.
Every command_id in final_report_claims.csv is present in this tracker.
Every active code path in final_code_manifest.csv exists.
No active code-manifest row owns stale tau06/7fd2530/old Treasury evidence.
Final MAD values validate against bd5d48e to declared tolerance.
```

### `raw-refresh`

Refreshes raw image inputs and QA reports. This does not define final report
numbers by itself; it prepares a new candidate corpus that must be compared
against the frozen artifacts.

```bash
python3 pipelines/run_pipeline.py \
  --repo-root . \
  --mode all \
  --credentials-path config/clientID.txt \
  --ports-path data/inputs/Havne_koor.txt \
  --patch-bboxes-path data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt \
  --start-year 2017 \
  --end-year 2026 \
  --cloud-thr-good 0.04 \
  --inventory-out-dir data/outputs/folder_inventory_report_final_refresh
```

Required QA:

```text
The bbox file is the final 49-port LA/LB file.
The usable output is dataset_root/god.
Cloud and cutoff filters have run.
The inventory report exists and covers all expected ports.
Refresh output is not final evidence until compared against frozen claims.
```

### `segmentation-exact`

Locks the selected U-Net package. Exact reproduction uses the cb031ac model
package, not local `models/ml/unet/weights/selected_best.pt`.

Canonical source:

```text
final_runs/full_gode_havnebilleder_tau06_final_2026_06_05/model/
provenance: cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34
```

Locked values:

```text
Selected checkpoint SHA-256: 876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc
Checkpoint epoch: 96
Pair contract: 1,143 images, 1,143 masks, 1,143 matched pairs, 0 errors
Split: 914 train, 114 validation, 115 test
Image normalization: percentile, low 2, high 98
Loss: BCE + Dice
Normalization layer: GroupNorm, 8 groups
LA/LB: represented under long_beach in the final model package
```

The model package is already materialized in this repository. `best.pt` is
Git-LFS tracked and symlinked into the model package from
`report_regen_2026-06-11/best.pt`.

Verify checkpoint hash:

```bash
shasum -a 256 final_runs/full_gode_havnebilleder_tau06_final_2026_06_05/model/best.pt
```

Refresh training is allowed through `pipelines/run_segmentation_pipeline.py`,
but a refresh run is not final until it reproduces the locked package contract.

### `supervised-qa`

Locks report-facing segmentation quality against masks at `tau=0.4`.

Canonical source:

```text
report_artifacts/supervised_tau04_checks_cb031ac_8a44dd0/
provenance: 1b6c343467e7ea73e13554e65316a2fb3b642694
```

Report-facing metrics:

| Run | Split | n | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | val | 114 | 0.670688 | 0.537722 | 0.661313 | 0.758906 |
| raw | test | 115 | 0.632329 | 0.496682 | 0.645789 | 0.729766 |
| ROI10 | val | 114 | 0.670791 | 0.538289 | 0.662263 | 0.756408 |
| ROI10 | test | 115 | 0.635262 | 0.500053 | 0.651698 | 0.729273 |

Canonical files:

```text
summary_metrics.csv
raw/val/threshold_sweep.csv
raw/test/threshold_sweep.csv
roi10/val/inference_summary.json
roi10/test/inference_summary.json
roi10/val/predictions.csv
roi10/test/predictions.csv
```

Rule:

```text
ROI10 is the only report-facing supervised ROI buffer pinned in 1b6c343.
ROI5 is not final unless regenerated and pinned through this same role.
Do not use bd5d48e full-inference ROI summaries as Dice/IoU/precision/recall evidence.
```

### `full-inference`

Locks the selected U-Net applied to the full image corpus.

Canonical source:

```text
final_runs/tau04_hk_28623539/
provenance: bd5d48e991e362f5114797622be8ca4b622ea0f2
```

Locked values:

```text
Operating threshold: tau = 0.4
ROI buffer: 10 px
Full inference cases: 39,530
Ports in full inference: 49
LA/LB in full inference: long_beach only; no separate los_angeles rows
Checkpoint epoch: 96
Source tag: thesis-final-source-2026-06-10
```

Refresh shape:

```bash
PYTHONPATH=. python3 -m models.ml.unet.infer \
  --checkpoint final_runs/full_gode_havnebilleder_tau06_final_2026_06_05/model/best.pt \
  --input-csv data/outputs/final_full_inference_input.csv \
  --out data/outputs/final_full_inference_tau04_roi10_refresh/inference \
  --threshold 0.4 \
  --no-require-mask \
  --apply-roi \
  --no-save-images \
  --batch-size 8 \
  --device auto
```

The refresh command must be preceded by a frozen image input manifest. The
locked report still uses bd5d48e until a refresh output passes all downstream
claim checks.

### `container-gnc`

Builds and validates port and country GNC panels.

Canonical files:

```text
final_runs/tau04_hk_28623539/daily_container_index/port_timeseries.csv
final_runs/tau04_hk_28623539/daily_container_index/country_gnc.csv
```

Locked values:

```text
Port-month observations: 4,797
Country-month observations: 2,323
Countries: 23
Hong Kong: separate country in the report-facing run mapping
China without Hong Kong: 110 country-months
Patch-count vs |GNC| Spearman: about -0.16
```

Refresh shape:

```bash
PYTHONPATH=. python3 -m analysis.build_container_index \
  --predictions data/outputs/final_full_inference_tau04_roi10_refresh/inference/predictions.csv \
  --out data/outputs/final_container_gnc_tau04_roi10_refresh
```

Port-month convention:

```text
Use 4,797 valid monthly port-GNC observations. This is 4,846 monthly port
levels minus 49 first-month rows without a previous month for GNC. Do not use
old 4,795 or 4,823 counts as final.
```

### `forecast-mad`

Locks the final finance and MAD suite. The final exact command materializes and
validates the bd5d48e USD/no-Sri-Lanka suite instead of using old defaults.

Canonical source:

```text
final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/
provenance: bd5d48e991e362f5114797622be8ca4b622ea0f2
```

Final exact command:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.run_final_usd19_mad_suite \
  --out-dir data/outputs/return_forecasting/usd19_no_sri_lanka_28623539_locked
```

Validation-only:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.run_final_usd19_mad_suite --validate-only
```

Full recompute command from the materialized frozen inputs:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.run_cleaned_research_suite \
  --country-gnc final_runs/tau04_hk_28623539/daily_container_index/country_gnc.csv \
  --port-timeseries final_runs/tau04_hk_28623539/daily_container_index/port_timeseries.csv \
  --market-indices final_runs/tau04_hk_28623539/inputs/all_market_indices_with_hong_kong.csv \
  --fx-rates final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/inputs/country_fx_rates_usd_with_hong_kong.csv \
  --selected-model-json pipelines/final_selected_model_tau04_bd5d48e.json \
  --cleaned-model-id 03_bce_dice_group_100ep_20260521_204006 \
  --cleaned-model-threshold 0.4 \
  --exclude-countries "Sri Lanka" \
  --out data/outputs/return_forecasting \
  --suite-id usd19_no_sri_lanka_28623539_recompute \
  --models historical_mean,always_positive,ols_predictive,elastic_net_panel,distributed_lag,random_forest_panel \
  --horizons 1,2,3,4,5 \
  --test-start 2020-01-01 \
  --min-train-months 36 \
  --min-non-null-features 3 \
  --seed 42 \
  --mad-lookbacks 5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
  --mad-max-weight 0.35

PYTHONPATH=. python3 -m analysis.return_forecasting.apply_final_treasury_adjustment \
  --base-dir data/outputs/return_forecasting/usd19_no_sri_lanka_28623539_recompute \
  --risk-free-csv final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/inputs/treasury_1mo_monthly_rf_used.csv
```

Locked finance/MAD facts:

```text
Return currency: USD
Excluded countries: Sri Lanka
Forecast suite: usd19_no_sri_lanka_28623539
Horizons: 1, 2, 3, 4, 5
Models: historical_mean, always_positive, ols_predictive, elastic_net_panel, distributed_lag, random_forest_panel
Test start: 2020-01-01
Seed: 42
Minimum training months: 36
Minimum non-null features: 3
MAD lookbacks: 5..18
MAD max country weight: 0.35
Total Treasury-adjusted MAD rows: 1,260
GNC-informed MAD rows: 840
```

Locked headline:

```text
Best GNC-informed MAD: distributed_lag, h=1, raw, L=5
Treasury-adjusted Sharpe: 0.915243
Matched equal-weight Treasury Sharpe: 0.548530
Historical-mean no-GNC best baseline: 0.830063
Always-positive baseline: 0.768995
Raw compound return: 1.567429
Equal-weight raw compound return: 0.910660
Raw max drawdown: -0.194416
Equal-weight raw max drawdown: -0.235974
Mean turnover: 0.525705
```

Best h=5 GNC-informed cluster anchor:

```text
elastic_net_panel, h=5, direction_risk_scaled, L=7
Treasury-adjusted Sharpe: 0.837631
Matched equal-weight Treasury Sharpe: 0.499941
```

### `ablation`

Locks the with-GNC versus no-GNC ablation for the final USD/no-Sri-Lanka suite.

Canonical local bundle:

```text
data/outputs/return_forecasting/gnc_ablation_tau04_usd19_no_sri_lanka_bd5d48e_20260612_v2/
```

Final command:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.run_gnc_ablation \
  --country-gnc final_runs/tau04_hk_28623539/daily_container_index/country_gnc.csv \
  --market-indices final_runs/tau04_hk_28623539/inputs/all_market_indices_with_hong_kong.csv \
  --fx-rates final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/inputs/country_fx_rates_usd_with_hong_kong.csv \
  --selected-model-json pipelines/final_selected_model_tau04_bd5d48e.json \
  --cleaned-model-id 03_bce_dice_group_100ep_20260521_204006 \
  --cleaned-model-threshold 0.4 \
  --exclude-countries "Sri Lanka" \
  --out data/outputs/return_forecasting \
  --suite-id gnc_ablation_tau04_usd19_no_sri_lanka_bd5d48e_20260612_v2
```

Output contract:

```text
combined_metrics.csv rows: 60
gnc_delta_metrics.csv rows: 30
Feature sets: with_gnc, no_gnc
combined_metrics.csv SHA256: eea339e76482e4842d96eab21affb3efe46f43df7ee90896ddf2452ed855c7cf
gnc_delta_metrics.csv SHA256: 874c8755027b54f58476e54b76225980388bbf9f57fe3ea328a1d4cd151c72aa
config.json SHA256: 7952dbf29ca92fcf98d4b4c7a0f7255398e1af7cd2761165b32c3c0e4a97b9f9
input_fingerprints.json SHA256: f3656810769560c38868b1034ba3e64716d87ef1bac52a15534995f9d2bb902c
```

Interpretation:

```text
rmse_improvement_from_gnc = RMSE(no GNC) - RMSE(with GNC)
direction_delta_from_gnc = DirAcc(with GNC) - DirAcc(no GNC)

Positive values mean GNC improves the metric. For the four active forecast
models, RMSE deltas range from -0.005013 to +0.000039, and directional
accuracy deltas range from -0.177019 to +0.004033. The evidence does not show
a robust positive forecasting contribution from GNC features in this ablation.
```

### `figures-tables`

Regenerates report-facing figures and MAD `.tex` tables from locked evidence.

Figure regeneration commands:

```bash
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_segmentation_figs_tau04.py
FINAL_SEGMENTATION_IMAGE_ROOT=/path/to/frozen/Final_Segmentation_LA_Edit MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_panels_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_test_grid_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_container_figs_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_appendix_gnc_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_mad_figs_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_best_config_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_scatter_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_pocket_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_pocket_wealth_tau04.py
MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp python3 report_regen_2026-06-11/regen_final_model03_training_figs.py
python3 report_regen_2026-06-11/assemble_final_figures.py
```

MAD table generation:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.build_final_mad_tables \
  --out-dir data/outputs/return_forecasting/final_mad_tables_20260613
```

The `weights.csv` file is stored in this repository through Git LFS. If the
full recompute command above has been run, use the recomputed weights instead:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.build_final_mad_tables \
  --out-dir data/outputs/return_forecasting/final_mad_tables_20260613 \
  --weights-csv data/outputs/return_forecasting/usd19_no_sri_lanka_28623539_recompute/mad_portfolio_cleaned/weights.csv
```

The qualitative panel command is the only figure command that requires a local
data root outside git. `FINAL_SEGMENTATION_IMAGE_ROOT` must point to the frozen
image folder containing `images/` and `masks/`; the script fails if the variable
or those subdirectories are missing.

The five generated MAD table fragments are:

```text
table_gnc_mad_top10_configurations.tex
table_gnc_mad_best_sharpe_by_forecast_method.tex
table_gnc_mad_sharpe_by_method_and_horizon.tex
table_gnc_mad_weight_summary_top10.tex
table_mad_baseline_comparison_treasury.tex
```

Active output bundles:

```text
report_regen_2026-06-11/segmentation_tau04_bundle/
report_regen_2026-06-11/container_index_tau04_bundle/
report_regen_2026-06-11/mad_tau04_bundle/
report_regen_2026-06-11/panels_tau04_bundle/
report_regen_2026-06-11/FinalFigures/
```

## Cleanup Policy

Cleanup is two-step:

```text
1. Use final_code_manifest.csv to classify active, support, historical and deletable_candidate code.
2. Delete or archive only after control-lock passes and no active command depends on the candidate.
```

Historical or stale evidence excluded from this clean repository:

```text
7fd2530 downstream rows
full_gode_havnebilleder_tau06_final_2026_06_05/runs/roi10/ as downstream final
tau06 or tau = 0.6 as the final downstream operating point
report_artifacts/treasury_sharpe_2026-06-05/ as the final finance bundle
cleaned_return_suite_20260522_v2 as the final finance suite
GeneratedEvidence/segmentation_cleaned_tau06 ROI metrics as final ROI metrics
models/ml/unet/weights/selected_best.pt as final checkpoint evidence
GNC-MAD Treasury Sharpe 0.999 as the final winner
Historical-mean no-GNC Sharpe 0.975 as the final baseline
Pure no-signal MAD Sharpe 0.906 as the final baseline
Equal-weight Treasury Sharpe 0.473 as the final benchmark
2,238 country-month observations
2,325 country-month observations
4,795 port-month observations
4,823 port-month observations
0.630502 as the final tau=0.4 test Dice
data/outputs/return_forecasting/gnc_ablation_20260522/ as final GNC ablation evidence
data/outputs/return_forecasting/gnc_ablation_tau04_bd5d48e_20260612/ as final GNC ablation evidence
data/outputs/return_forecasting/gnc_ablation_tau04_usd19_no_sri_lanka_bd5d48e_20260612/ as final GNC ablation evidence
```

Scripts with old defaults are not final entrypoints:

```text
analysis/return_forecasting/recompute_treasury_adjusted_sharpe.py
analysis/return_forecasting/build_mad_gnc_report.py
analysis/return_forecasting/report_country_direction.py
```

Transaction-cost and shorting sensitivity scripts are support-only unless run
with an explicit final USD19 `--base-dir`. Their old defaults must not drive
report claims.

## Open Checks

```text
If ROI5 is needed in final text, regenerate and pin an ROI5 supervised check
under the same 1b6c343/cb031ac-equivalent path. Otherwise remove ROI5 from
final claims.

If a future refresh replaces bd5d48e, update SOURCE_OF_TRUTH.md first, then
update all three manifests, then rerun control-lock before touching report text.
```
