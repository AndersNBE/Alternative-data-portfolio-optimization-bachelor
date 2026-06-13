# Reproduction Guide

Run all commands from the repository root.

## Control Lock

```bash
make lock
```

This runs:

```bash
PYTHONPATH=. python3 -m pytest tests/test_return_forecasting.py tests/test_final_pipeline_lock.py
PYTHONPATH=. python3 -m analysis.return_forecasting.run_final_usd19_mad_suite --validate-only
```

## MAD Tables

```bash
make tables
```

Outputs:

```text
data/outputs/return_forecasting/final_mad_tables/
```

Generated table fragments:

```text
table_gnc_mad_top10_configurations.tex
table_gnc_mad_best_sharpe_by_forecast_method.tex
table_gnc_mad_sharpe_by_method_and_horizon.tex
table_gnc_mad_weight_summary_top10.tex
table_mad_baseline_comparison_treasury.tex
```

## Figure Bundles

```bash
make figures
```

The panel and test-grid commands that require raw frozen image/mask files are
intentionally not part of `make figures`. Run the complete figure target when
that root is available:

```bash
FINAL_SEGMENTATION_IMAGE_ROOT=/path/to/Final_Segmentation_LA_Edit \
  make figures-with-images
```

## Data Refresh Boundary

`pipelines/run_pipeline.py` can refresh raw Sentinel imagery with credentials and the final 49-port bbox file. Refresh output is not byte-identical report evidence until it passes the same frozen-claim checks and updates the manifests.
