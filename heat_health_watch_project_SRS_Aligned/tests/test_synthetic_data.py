import pandas as pd
import pytest

from src import config as C
from src import data_loader as D
from src import synthetic_data as S


def test_reproducible_with_same_seed():
    pd.testing.assert_frame_equal(S.generate_scenarios(8, 123), S.generate_scenarios(8, 123))


def test_different_seed_gives_different_data():
    assert not S.generate_scenarios(8, 1).equals(S.generate_scenarios(8, 2))


def test_all_five_levels_present_with_requested_count():
    d = S.generate_scenarios(12, 42)
    assert len(d) == 60
    assert d["Score_Category_5L"].value_counts().to_dict() == {lvl: 12 for lvl in C.LEVEL_ORDER}


def test_rows_labelled_synthetic():
    d = S.generate_scenarios(3, 42)
    assert (d["Data_Source"] == S.SYNTHETIC_LABEL).all() and "NOT OBSERVED" in S.SYNTHETIC_LABEL
    assert d["Scenario_ID"].str.startswith("SYN-").all()


def test_validation_passes_and_values_plausible():
    d = S.generate_scenarios(40, 42)
    assert S.validate_synthetic(d) == []
    assert not ((d["Max_Temperature_C"] >= 42) & (d["Relative_Humidity_%"] > 75)).any()


def test_validation_detects_problems():
    d = S.generate_scenarios(3, 42)
    d.loc[0, "AQI"] = 900
    d.loc[1, "Data_Source"] = "real"
    probs = S.validate_synthetic(d)
    assert any("AQI" in p for p in probs) and any("labelled" in p for p in probs)


@pytest.mark.parametrize("n", [0, -1, 201, 2.5, "5"])
def test_invalid_counts_rejected(n):
    with pytest.raises(ValueError):
        S.generate_scenarios(n)


def test_scores_use_the_same_formula():
    from src.risk_scoring import compute_score
    d = S.generate_scenarios(5, 42)
    assert (compute_score(d["Max_Temperature_C"], d["Relative_Humidity_%"], d["AQI"]) == d["Risk_Score"]).all()


def test_generation_never_touches_historical_data(df):
    before = df.copy()
    S.generate_scenarios(5, 42)
    pd.testing.assert_frame_equal(before, df)


def test_csv_export_keeps_label():
    d = S.generate_scenarios(2, 42)
    csv = D.to_csv_bytes(d).decode()
    assert "SYNTHETIC" in csv and csv.count("\n") == 11
