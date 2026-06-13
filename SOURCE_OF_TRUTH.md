# Final Source-of-Truth Contract

Last updated: 2026-06-12

This file is the short conflict-resolution rule for final thesis numbers. If an
older tracker row, figure caption, note, script comment, local artifact, or
backup conflicts with this file, this file wins.

This clean repository materializes the final evidence locally and excludes the
old historical backup folders.

## Hard Rule

Use three evidence roles, not one blended "final" source:

| Role | Commit / artifact | Use for |
| --- | --- | --- |
| U-Net model package provenance | `cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34` | checkpoint identity, model package, split files, pair contract, training metadata |
| Report-facing supervised U-Net metrics | `1b6c343467e7ea73e13554e65316a2fb3b642694` | validation/test Dice, IoU, precision, recall, threshold sweep and ROI10 metrics against masks |
| Full inference and downstream results | `bd5d48e991e362f5114797622be8ca4b622ea0f2` | full corpus inference, operational ROI, NC/GNC, return forecasting, USD/no-Sri-Lanka finance, Treasury-adjusted MAD |

Short form:

```text
Model package/provenance -> cb031ac
Supervised segmentation metrics with masks -> 1b6c343
Everything from full inference onward -> bd5d48e
```

## 1. U-Net Model Package Provenance

Use `cb031ac` when the claim is about the selected U-Net package itself:

```text
final_runs/full_gode_havnebilleder_tau06_final_2026_06_05/model/
provenance: cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34
```

Locked provenance:

```text
Selected checkpoint SHA-256: 876d7bb8a492c23fa448d187ebb11ab756a0a80864e140b5740795b7f30a64fc
Checkpoint epoch: 96
Pair contract: 1,143 images, 1,143 masks, 1,143 matched pairs, 0 errors
Frozen split: 914 train, 114 validation, 115 test
Image normalization: percentile, low 2, high 98
Loss: BCE + Dice
Normalization layer: GroupNorm, 8 groups
LA/LB: represented under long_beach in the final package
```

Do not use `models/ml/unet/weights/selected_best.pt` or
`models/ml/unet/weights/selected_model.json` as final checkpoint evidence.

## 2. Report-Facing Supervised U-Net Metrics

Use `1b6c343` for any segmentation-quality claim that requires ground-truth
masks:

```text
report_artifacts/supervised_tau04_checks_cb031ac_8a44dd0/
provenance: 1b6c343467e7ea73e13554e65316a2fb3b642694
```

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

Locked report-facing metrics at tau = 0.4:

| Run | Split | n | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | val | 114 | 0.670688 | 0.537722 | 0.661313 | 0.758906 |
| raw | test | 115 | 0.632329 | 0.496682 | 0.645789 | 0.729766 |
| ROI10 | val | 114 | 0.670791 | 0.538289 | 0.662263 | 0.756408 |
| ROI10 | test | 115 | 0.635262 | 0.500053 | 0.651698 | 0.729273 |

ROI10 deltas at tau = 0.4:

```text
Validation: Delta Dice +0.000102, Delta IoU +0.000566, Delta precision +0.000950, Delta recall -0.002498
Test:       Delta Dice +0.002933, Delta IoU +0.003371, Delta precision +0.005910, Delta recall -0.000493
```

ROI5 is not available in the `1b6c343` artifact. Do not use old ROI5 numbers
as final report evidence unless they are explicitly labelled historical.

## 3. Full Inference And Downstream Results

Use `bd5d48e` for everything after the selected U-Net is applied to the full
image corpus:

```text
final_runs/tau04_hk_28623539/
provenance: bd5d48e991e362f5114797622be8ca4b622ea0f2
```

Canonical files and folders:

```text
inference/inference_summary.json
inference/predictions.csv
inference/roi_summary.csv
daily_container_index/port_timeseries.csv
daily_container_index/country_gnc.csv
return_forecasting/usd19_no_sri_lanka_28623539/
```

Locked full-pipeline facts:

```text
Operating threshold: tau = 0.4
ROI buffer: 10 px
Full inference cases: 39,530
Ports in full inference: 49
LA/LB in full inference: long_beach only; no separate los_angeles rows
Checkpoint epoch: 96
Finance suite: usd19_no_sri_lanka_28623539
Source tag: thesis-final-source-2026-06-10
```

Important ROI rule:

`bd5d48e` full-inference files do not contain ground-truth masks. Dice, IoU,
precision and recall fields in full-inference ROI summaries are not valid
segmentation-quality metrics. For ROI segmentation-quality claims, use
`1b6c343`.

## 4. Report-Facing Downstream Values

Current downstream finance headline from `bd5d48e`:

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

Container/GNC report-facing facts:

```text
Port-month observations: 4,797
Country-month observations: 2,323
Countries: 23
Hong Kong: separate country in the report-facing run mapping
China without Hong Kong: 110 country-months
Patch-count vs |GNC| Spearman: about -0.16
```

Resolved port-month convention:

```text
Use 4,797 port-month observations when the report needs the final monthly
port-GNC panel size. This is reconstructed from final_runs/
tau04_hk_28623539/daily_container_index/port_timeseries.csv by
report_regen_2026-06-11/regen_container_figs_tau04.py:
4,846 monthly port levels minus 49 first-month rows without a previous month
for GNC = 4,797 valid port-month rows. The old 4,795 note is unverified and
must not be used as final. Never use the old 4,823 port-month count as final.
```

## 5. Stale Evidence That Must Not Drive Final Claims

Do not use these as final report evidence:

```text
7fd2530 downstream rows
tau06 or tau = 0.6 as the final downstream operating point
GNC-MAD Treasury Sharpe 0.999 as the final winner
Historical-mean no-GNC Sharpe 0.975 as the final baseline
Pure no-signal MAD Sharpe 0.906 as the final baseline
Equal-weight Sharpe 0.473 as the matched final benchmark
2,238 country-month observations
2,325 country-month observations
4,795 port-month observations
4,823 port-month observations
0.630502 as the final tau=0.4 test Dice
old GeneratedEvidence/segmentation_cleaned_tau06 ROI metrics as final ROI evidence
```

## 6. Required Agent Behavior

Before editing thesis text, regenerating figures, or answering final-number
questions:

1. Identify whether the claim is model provenance, supervised segmentation
   metrics, or downstream analysis.
2. Use `cb031ac`, `1b6c343`, or `bd5d48e` according to that category.
3. Do not silently mix values across categories.
4. If a value still exists only in old tracker history, treat it as stale until
   verified against the active artifacts above.
