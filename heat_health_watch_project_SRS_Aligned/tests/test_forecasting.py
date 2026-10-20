import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import forecasting as F


@pytest.fixture(scope="module")
def models(df):
    return F.fit_all(df)


LAST = pd.Timestamp("2024-11-30")


def test_rejects_end_before_start(df):
    with pytest.raises(F.ForecastRequestError, match="earlier"):
        F.forecast_range(df, "2024-12-10", "2024-12-05")


def test_rejects_start_inside_history(df):
    with pytest.raises(F.ForecastRequestError, match="after the last observed"):
        F.forecast_range(df, "2024-11-30", "2024-12-05")


def test_rejects_too_long_range(df):
    with pytest.raises(F.ForecastRequestError, match="too long"):
        F.forecast_range(df, "2024-12-01", "2025-12-01")


def test_rejects_start_too_far_out(df):
    with pytest.raises(F.ForecastRequestError, match="beyond the data"):
        F.forecast_range(df, "2030-01-01", "2030-01-10")


def test_rejects_invalid_date():
    with pytest.raises(F.ForecastRequestError):
        F.validate_request("not-a-date", "2024-12-05", LAST)


def test_insufficient_history_message(df):
    with pytest.raises(F.ForecastRequestError, match="at least"):
        F.forecast_range(df.head(200), "2024-12-01", "2024-12-05")


def test_output_length_and_schema(df, models):
    out = F.forecast_range(df, "2024-12-01", "2024-12-20", models=models)
    assert len(out) == 20 and not out.empty
    for c in ["Date", "Max_Temperature_C", "Relative_Humidity_%", "AQI", "Risk_Score", "Score_Category_5L",
              "Dataset_Rule_Label", "RF_Class", "Advisory", "Reliability", "Method",
              "Max_Temperature_C_status", "Relative_Humidity_%_status", "AQI_status"]:
        assert c in out.columns
    assert out["Date"].is_monotonic_increasing and out["Date"].iloc[0] == pd.Timestamp("2024-12-01")


def test_values_within_valid_ranges(df, models):
    out = F.forecast_range(df, "2025-03-01", "2025-08-27", models=models)
    assert out["Risk_Score"].between(0, 100).all()
    assert out["Relative_Humidity_%"].between(0, 100).all() and out["AQI"].between(0, 500).all()
    assert (out["Risk_Score_low80"] <= out["Risk_Score_high80"]).all()


def test_status_distinguishes_forecast_assumed_and_user_supplied(df, models):
    out = F.forecast_range(df, "2024-12-01", "2025-02-15", models=models)
    assert out.loc[0, "AQI_status"].startswith("Forecast")
    assert out.loc[len(out) - 1, "AQI_status"].startswith("Assumed")
    sup = F.forecast_range(df, "2024-12-01", "2024-12-05", overrides={"AQI": 300, "Relative_Humidity_%": 40}, models=models)
    assert (sup["AQI"] == 300).all() and (sup["AQI_status"] == "User-supplied").all()
    assert (sup["Relative_Humidity_%_status"] == "User-supplied").all()
    assert sup["Max_Temperature_C_status"].str.startswith("Forecast").all()


def test_reliability_labels(df, models):
    out = F.forecast_range(df, "2024-12-01", "2025-01-15", models=models).set_index("Horizon_days")
    assert out.loc[1, "Reliability"].startswith("Short") and out.loc[20, "Reliability"].startswith("Extended")
    assert out.loc[40, "Reliability"].startswith("Illustrative")


def test_override_validation(df, models):
    with pytest.raises(F.ForecastRequestError):
        F.forecast_range(df, "2024-12-01", "2024-12-03", overrides={"AQI": 900}, models=models)
    with pytest.raises(F.ForecastRequestError):
        F.forecast_range(df, "2024-12-01", "2024-12-03", overrides={"Max_Temperature_C": 30}, models=models)


def test_rf_class_marked_unavailable(df, models):
    out = F.forecast_range(df, "2024-12-01", "2024-12-03", models=models)
    assert out["RF_Class"].str.startswith("Unavailable").all()


def test_far_future_converges_to_seasonal_average(df, models):
    a = F.forecast_range(df, "2026-06-01", "2026-06-02", models=models)
    b = F.forecast_range(df, "2025-06-01", "2025-06-02", models=models)
    assert abs(a["Max_Temperature_C"].iloc[0] - b["Max_Temperature_C"].iloc[0]) < 1.0


def test_summer_forecast_warmer_than_winter(df, models):
    jun = F.forecast_range(df, "2025-06-01", "2025-06-02", models=models)["Max_Temperature_C"].mean()
    jan = F.forecast_range(df, "2025-01-10", "2025-01-11", models=models)["Max_Temperature_C"].mean()
    assert jun > jan + 8


def test_evaluation_table_and_baselines(df):
    ev = F.evaluate(df)
    assert set(ev["Variable"]) == set(F.FORECAST_VARS) and (ev["MAE"] > 0).all()
    t = ev[(ev.Variable == "Max_Temperature_C") & (ev.Setting.str.startswith("Long"))].set_index("Method")["MAE"]
    assert t["Model (seasonal+AR1)"] < t["Persistence"]
    assert t["Model (seasonal+AR1)"] < t["Seasonal naive (t-365)"]


def test_forecast_is_deterministic(df, models):
    a = F.forecast_range(df, "2024-12-01", "2024-12-05", models=models)
    b = F.forecast_range(df, "2024-12-01", "2024-12-05", models=models)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_continuation_is_appended_and_labelled(df):
    end = LAST + pd.Timedelta(days=35)
    before = df.copy(deep=True)
    synthetic = F.generate_synthetic_continuation(df, end, seed=7)
    pd.testing.assert_frame_equal(df, before)  # original dataset is not modified
    assert len(synthetic) == 35
    assert synthetic["Date"].iloc[0] == LAST + pd.Timedelta(days=1)
    assert synthetic["Date"].iloc[-1] == end
    assert synthetic["Data_Source"].eq(F.SYNTHETIC_CONTINUATION_LABEL).all()
    assert synthetic["Max_Temperature_C"].between(-5, 55).all()
    assert synthetic["Relative_Humidity_%"].between(0, 100).all()
    assert synthetic["AQI"].between(0, 500).all()


def test_forecast_after_synthetic_continuation_is_explicitly_marked(df):
    synthetic = F.generate_synthetic_continuation(df, LAST + pd.Timedelta(days=40), seed=7)
    history = pd.concat([
        df[[C.DATE_COL, *F.FORECAST_VARS]],
        synthetic[[C.DATE_COL, *F.FORECAST_VARS, "Data_Source"]],
    ], ignore_index=True).sort_values(C.DATE_COL).reset_index(drop=True)
    forecast_start = pd.Timestamp(synthetic["Date"].max()) + pd.Timedelta(days=1)
    models = F.fit_all(history)
    result = F.forecast_range(history, forecast_start, forecast_start + pd.Timedelta(days=4), models=models)
    assert result["History_Basis"].str.contains("Original observations + generated synthetic continuation", regex=False).all()
    assert result["Method"].str.contains("observed + synthetic continuation", regex=False).all()


def test_synthetic_continuation_is_reproducible(df):
    end = LAST + pd.Timedelta(days=10)
    a = F.generate_synthetic_continuation(df, end, seed=99)
    b = F.generate_synthetic_continuation(df, end, seed=99)
    pd.testing.assert_frame_equal(a, b)
