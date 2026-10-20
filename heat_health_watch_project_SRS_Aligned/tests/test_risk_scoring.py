import math

import numpy as np
import pytest

from src import advisories as A
from src import config as C
from src import risk_scoring as R


def test_formula_reproduces_every_stored_score(df):
    s = R.compute_score(df["Max_Temperature_C"], df["Relative_Humidity_%"], df["AQI"])
    assert np.abs(s - df[C.SCORE_COL]).max() < 0.006


def test_original_label_rule_matches_every_row(df):
    s = R.compute_score(df["Max_Temperature_C"], df["Relative_Humidity_%"], df["AQI"])
    pred = np.where(s > C.ORIGINAL_HIGH_THRESHOLD, "High", "Moderate")
    assert (pred == df[C.TARGET_COL].values).all()


@pytest.mark.parametrize("t,h,a", [(9.1, 16.1, 0), (47.9, 99.3, 500), (-5, 0, 0), (55, 100, 500)])
def test_score_bounds_at_extremes(t, h, a):
    assert 0 <= R.calculate_risk_score(t, h, a) <= 100


def test_max_and_min_score_values():
    assert R.calculate_risk_score(47.9, 99.3, 500) == 100.0
    assert R.calculate_risk_score(9.1, 16.1, 0) == 0.0


@pytest.mark.parametrize("score,expected", [
    (0, "Very Low"), (19.99, "Very Low"), (20, "Low"), (39.99, "Low"), (40, "Moderate"),
    (59.99, "Moderate"), (60, "High"), (79.99, "High"), (80, "Severe"), (100, "Severe")])
def test_all_five_categories_and_boundaries(score, expected):
    assert R.categorize_score(score) == expected


def test_vectorised_matches_scalar():
    xs = [0, 19.99, 20, 45, 60, 79.99, 80, 100]
    assert list(R.categorize_series(xs)) == [R.categorize_score(x) for x in xs]


def test_thresholds_are_configurable(monkeypatch):
    monkeypatch.setattr(C, "RISK_LEVELS", [(0.0, "Very Low"), (10.0, "Low"), (20.0, "Moderate"), (30.0, "High"), (40.0, "Severe")])
    assert R.categorize_score(45) == "Severe" and R.categorize_score(5) == "Very Low"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "hot", -50, 80])
def test_invalid_temperature_rejected(bad):
    with pytest.raises(R.InvalidInputError):
        R.calculate_risk_score(bad, 50, 100)


@pytest.mark.parametrize("h,a", [(-1, 100), (101, 100), (50, -1), (50, 501), (None, 100), (50, None)])
def test_invalid_humidity_or_aqi_rejected(h, a):
    with pytest.raises(R.InvalidInputError):
        R.calculate_risk_score(30, h, a)


def test_components_sum_to_score():
    r = R.assess(38, 45, 250)
    assert math.isclose(sum(r["components"].values()), r["score"], abs_tol=0.02)


def test_score_is_deterministic():
    assert R.calculate_risk_score(40, 30, 200) == R.calculate_risk_score(40, 30, 200)


def test_categorize_rejects_bad_scores():
    with pytest.raises(R.InvalidInputError):
        R.categorize_score(101)
    with pytest.raises(R.InvalidInputError):
        R.categorize_score(float("nan"))


@pytest.mark.parametrize("cat", C.LEVEL_ORDER)
def test_advisory_for_every_category(cat):
    a = A.advisory_for_category(cat)
    assert a["summary"] and a["actions"] and "not medical advice" in a["disclaimer"].lower()


def test_advisory_unknown_category():
    with pytest.raises(KeyError):
        A.advisory_for_category("Catastrophic")


def test_advisories_avoid_diagnosis_claims():
    text = " ".join(A.advisory_text(c) for c in C.LEVEL_ORDER).lower()
    assert "diagnos" not in text and "cure" not in text
