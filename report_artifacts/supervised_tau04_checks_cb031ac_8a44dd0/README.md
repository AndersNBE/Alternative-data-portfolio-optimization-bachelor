# Supervised tau=0.4 checks for cb031ac / 8a44dd0

Purpose: local supervised validation/test checks for the final Model 03 checkpoint after the source-of-truth contract commit `8a44dd0`.

Source contract:
- supervised U-Net source: `cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34`
- source contract commit: `8a44dd0`
- checkpoint SHA-256: `876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc`
- checkpoint epoch: 96
- operating threshold: tau = 0.4
- image normalization: percentile, low 2, high 98
- ROI diagnostics: final `Havne_koor.txt` plus `patch_bboxes_final_49ports_lalb_20260527.txt`, buffer 10 px

Headline supervised metrics at tau=0.4:

| run | split | n | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| raw | val | 114 | 0.670688 | 0.537722 | 0.661313 | 0.758906 |
| raw | test | 115 | 0.632329 | 0.496682 | 0.645789 | 0.729766 |
| ROI10 | val | 114 | 0.670791 | 0.538289 | 0.662263 | 0.756408 |
| ROI10 | test | 115 | 0.635262 | 0.500053 | 0.651698 | 0.729273 |

ROI10 deltas at tau=0.4:

| split | cases with ROI | mean ROI coverage | Delta Dice | Delta IoU | Delta Precision | Delta Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 114 | 0.136310 | 0.000102 | 0.000566 | 0.000950 | -0.002498 |
| test | 115 | 0.146788 | 0.002933 | 0.003371 | 0.005910 | -0.000493 |

Use `summary_metrics.csv` for report-facing numbers. Use `raw/*/threshold_sweep.csv` for threshold-sweep evidence. Use `roi10/*/roi_summary.csv` and `roi10/*/predictions.csv` for per-port and per-case ROI diagnostics.

Note: these runs must use the `cb031ac` U-Net code path or an equivalent code path with percentile normalization. Running the active branch loader without percentile normalization gives lower numbers and is not report evidence.
