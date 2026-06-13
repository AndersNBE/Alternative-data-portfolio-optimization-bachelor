.PHONY: test lock mad tables figures compile

PYTHON ?= python3
PYTHONPATH := .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/test_return_forecasting.py tests/test_final_pipeline_lock.py

lock: test mad

mad:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.return_forecasting.run_final_usd19_mad_suite --validate-only

tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.return_forecasting.build_final_mad_tables --out-dir data/outputs/return_forecasting/final_mad_tables

figures:
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_segmentation_figs_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_container_figs_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_appendix_gnc_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_mad_figs_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_best_config_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_scatter_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_pocket_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_pocket_wealth_tau04.py
	MPLCONFIGDIR=/tmp/mpl-cache-codex XDG_CACHE_HOME=/tmp $(PYTHON) report_regen_2026-06-11/regen_final_model03_training_figs.py
	$(PYTHON) report_regen_2026-06-11/assemble_final_figures.py

compile:
	$(PYTHON) -m py_compile $$(find analysis data models pipelines report_regen_2026-06-11 tests -name '*.py' -not -path '*/__pycache__/*')
