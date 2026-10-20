import pandas as pd
import pytest

from src import hourly_analysis as H


@pytest.fixture()
def day():
    return H.simulate_hourly(28, 41, 35, 250, date="2024-05-20")


def test_24_rows_and_schema(day):
    assert len(day) == 24 and list(day["Hour"]) == list(range(24))
    for c in ["Temperature_C", "Relative_Humidity_%", "AQI", "Risk_Score", "Score_Category_5L", "Advisory", "Data_Source"]:
        assert c in day.columns


def test_marked_as_simulated(day):
    assert (day["Data_Source"] == H.SIM_LABEL).all() and "SIMULATED" in H.SIM_LABEL


def test_temperature_extremes_match_inputs(day):
    assert day["Temperature_C"].max() == 41.0 and day["Temperature_C"].min() == 28.0
    assert day.loc[day["Hour"] == 15, "Temperature_C"].iloc[0] == 41.0


def test_values_within_physical_ranges(day):
    assert day["Relative_Humidity_%"].between(5, 100).all() and day["AQI"].between(0, 500).all()
    assert day["Risk_Score"].between(0, 100).all()


def test_humidity_lowest_when_hottest(day):
    assert day.loc[day["Temperature_C"].idxmax(), "Relative_Humidity_%"] == day["Relative_Humidity_%"].min()


def test_deterministic_for_seed():
    pd.testing.assert_frame_equal(H.simulate_hourly(28, 41, 35, 250), H.simulate_hourly(28, 41, 35, 250))


def test_peak_hour_is_argmax_not_hardcoded(day):
    pk = H.peak_hour(day)
    assert pk["score"] == day["Risk_Score"].max()
    # With almost no diurnal temperature range and heavy pollution, the AQI profile drives the
    # peak, so the highest-risk hour is NOT the hottest hour: the peak comes from the profile.
    flat = H.simulate_hourly(25, 26, 50, 400)
    assert H.peak_hour(flat)["hour"] != int(flat["Temperature_C"].values.argmax())


@pytest.mark.parametrize("args", [(30, 20, 40, 100), (10, 20, 120, 100), (10, 20, 40, 600), (10, 20, None, 100), (-20, 20, 40, 100)])
def test_invalid_inputs_rejected(args):
    with pytest.raises(H.HourlyInputError):
        H.simulate_hourly(*args)


def test_invalid_hours_rejected():
    with pytest.raises(H.HourlyInputError):
        H.simulate_hourly(20, 30, 40, 100, tmin_hour=16, tmax_hour=15)


def test_risk_windows_contiguous_and_threshold():
    h = H.simulate_hourly(30.1, 44.0, 23.2, 332.0)
    wins = H.risk_windows(h, "High")
    flagged = set(h.loc[h["Risk_Score"] >= 60, "Hour"])
    covered = {x for w in wins for x in range(int(w["from"][:2]), int(w["to"][:2]) + 1)}
    assert flagged == covered


def test_no_windows_on_mild_day():
    assert H.risk_windows(H.simulate_hourly(12, 20, 50, 60), "High") == []
    with pytest.raises(H.HourlyInputError):
        H.risk_windows(H.simulate_hourly(12, 20, 50, 60), "Extreme")
