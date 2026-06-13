from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.return_forecasting.data import build_forecast_panel, feature_ready_panel
from analysis.return_forecasting.evaluation import add_relative_metrics, evaluate_predictions
from analysis.return_forecasting.models import (
    distributed_lag,
    elastic_net_panel,
    historical_mean,
    ols_predictive,
    random_forest_panel,
)


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=54, freq="MS")
    rows = []
    for country_idx, country in enumerate(["A", "B", "C"]):
        gnc = np.sin(np.arange(len(dates)) / 4.0 + country_idx) * 0.2
        returns = 0.003 + 0.04 * np.roll(gnc, 1) + np.linspace(-0.01, 0.01, len(dates))
        returns[0] = 0.0
        for i, date in enumerate(dates):
            rows.append(
                {
                    "country": country,
                    "index_name": f"{country} index",
                    "symbol": country,
                    "date": date,
                    "month_str": date.to_period("M").strftime("%Y-%m"),
                    "target_return": float(returns[i]),
                    "gnc_same_month": float(gnc[i]),
                    "n_ports": 1 + country_idx,
                    "mean_NC": 1000 + 10 * i,
                    "mean_observations_per_port": 4,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    group = panel.groupby("country", group_keys=False)
    panel["gnc"] = group["gnc_same_month"].shift(1)
    for lag in range(1, 7):
        panel[f"gnc_lag_{lag}"] = group["gnc_same_month"].shift(lag)
        panel[f"return_lag_{lag}"] = group["target_return"].shift(lag)
    panel["gnc_ma_3"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(3).mean())
    panel["gnc_ma_6"] = group["gnc_same_month"].transform(lambda s: s.shift(1).rolling(6).mean())
    panel["gnc_delta_1"] = panel["gnc_lag_1"] - panel["gnc_lag_2"]
    panel["gnc_delta_3"] = panel["gnc_lag_1"] - panel["gnc_lag_4"]
    panel["rolling_vol_3"] = group["target_return"].transform(lambda s: s.shift(1).rolling(3).std())
    panel["rolling_vol_6"] = group["target_return"].transform(lambda s: s.shift(1).rolling(6).std())
    return panel.dropna(subset=["target_return"]).reset_index(drop=True)


def test_build_forecast_panel_lags_gnc_before_target_month() -> None:
    country_gnc = pd.DataFrame(
        {
            "country": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "month_str": ["2020-01", "2020-02", "2020-03"],
            "gnc_same_month": [1.0, 2.0, 3.0],
            "n_ports": [1, 1, 1],
            "mean_NC": [100.0, 110.0, 120.0],
            "mean_observations_per_port": [2.0, 2.0, 2.0],
        }
    )
    market_returns = pd.DataFrame(
        {
            "country": ["A", "A", "A"],
            "index_name": ["A"] * 3,
            "symbol": ["A"] * 3,
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "month_str": ["2020-01", "2020-02", "2020-03"],
            "adj_close": [100.0, 101.0, 102.0],
            "market_return": [0.01, 0.02, 0.03],
        }
    )

    panel, _ = build_forecast_panel(country_gnc=country_gnc, market_returns=market_returns)
    feb = panel.loc[panel["month_str"] == "2020-02"].iloc[0]

    assert feb["target_return"] == 0.02
    assert feb["gnc"] == 1.0
    assert feb["gnc_lag_1"] == 1.0
    assert feb["return_lag_1"] == 0.01


def test_all_return_forecasting_models_follow_common_interface() -> None:
    panel = feature_ready_panel(_synthetic_panel(), min_non_null_features=3)
    config = {
        "min_train_months": 12,
        "min_train_size": 12,
        "min_country_train": 12,
        "min_pooled_train": 12,
        "min_feature_obs": 8,
        "use_cv": False,
        "n_estimators": 10,
        "max_depth": 3,
        "n_jobs": 1,
        "random_state": 123,
    }

    modules = [
        historical_mean,
        ols_predictive,
        distributed_lag,
        elastic_net_panel,
        random_forest_panel,
    ]
    all_predictions = []
    for module in modules:
        predictions, meta = module.run_model(panel.copy(), config)
        assert set(
            [
                "model_id",
                "country",
                "date",
                "month_str",
                "actual_return",
                "predicted_return",
                "train_size",
            ]
        ).issubset(predictions.columns)
        assert len(predictions) == len(panel)
        assert predictions["predicted_return"].notna().any()
        json.dumps(meta, default=str)
        all_predictions.append(predictions)

    metrics = add_relative_metrics(evaluate_predictions(pd.concat(all_predictions, ignore_index=True)))
    assert set(module.MODEL_ID for module in modules) == set(metrics["model_id"])
    assert metrics["rmse"].notna().all()


def test_mad_solver_binds_center_to_mean_scenario_return(monkeypatch) -> None:
    from analysis.return_forecasting import run_cleaned_research_suite as suite

    scenarios = np.array(
        [
            [0.0, 2.0],
            [2.0, 4.0],
            [4.0, 6.0],
        ],
        dtype=float,
    )
    captured = {}

    def fake_linprog(c, *, A_ub, b_ub, A_eq, b_eq, bounds, method):
        captured["A_eq"] = np.asarray(A_eq, dtype=float)
        captured["b_eq"] = np.asarray(b_eq, dtype=float)
        x = np.zeros(len(c), dtype=float)
        x[:2] = [0.5, 0.5]
        x[-1] = 3.0

        class Result:
            success = True

        result = Result()
        result.x = x
        return result

    monkeypatch.setattr(suite, "linprog", fake_linprog)

    weights, optimal = suite.solve_mad_weights(
        scenarios=scenarios,
        expected=np.array([0.02, 0.05], dtype=float),
        max_weight=1.0,
    )

    assert optimal
    np.testing.assert_allclose(weights, [0.5, 0.5])
    np.testing.assert_allclose(captured["b_eq"], [1.0, 0.0])
    np.testing.assert_allclose(
        captured["A_eq"],
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0, 0.0, -1.0],
        ],
    )
