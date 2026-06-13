# Return Forecasting And MAD

This package contains the final report-facing finance pipeline:

- `run_cleaned_research_suite.py`: recomputes USD/no-Sri-Lanka return forecasts and MAD portfolios from frozen GNC, market, and FX inputs.
- `apply_final_treasury_adjustment.py`: recomputes Treasury-excess metrics for a recomputed suite.
- `run_final_usd19_mad_suite.py`: validates and materializes the locked final suite from `final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/`.
- `build_final_mad_tables.py`: generates the five final MAD LaTeX tables.
- `run_gnc_ablation.py`: final with-GNC versus no-GNC ablation runner.

The active final suite is:

```text
final_runs/tau04_hk_28623539/return_forecasting/usd19_no_sri_lanka_28623539/
```

Quick validation:

```bash
PYTHONPATH=. python3 -m analysis.return_forecasting.run_final_usd19_mad_suite --validate-only
```
