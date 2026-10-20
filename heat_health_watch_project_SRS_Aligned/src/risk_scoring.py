"""Heat-Health Risk Score and project-defined five-level categories.

The score is a transparent project heuristic that reproduces the formula that
was used to build the supplied dataset. It is NOT a calibrated probability of
illness and the category thresholds are NOT official medical standards.
"""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np
import pandas as pd

from . import config as C

Number = Union[int, float, np.ndarray, pd.Series]


class InvalidInputError(ValueError):
    """Raised when a scalar input is missing or outside its valid range."""


def validate_inputs(max_temp_c, humidity_pct, aqi) -> None:
    """Validate scalar inputs; raise InvalidInputError with a helpful message."""
    named = {"Max_Temperature_C": max_temp_c, "Relative_Humidity_%": humidity_pct, "AQI": aqi}
    for name, val in named.items():
        if val is None:
            raise InvalidInputError(f"{name} is missing.")
        try:
            v = float(val)
        except (TypeError, ValueError):
            raise InvalidInputError(f"{name} must be a number, got {val!r}.")
        if math.isnan(v) or math.isinf(v):
            raise InvalidInputError(f"{name} must be a finite number.")
        lo, hi = C.VALID_RANGES[name]
        if not lo <= v <= hi:
            raise InvalidInputError(f"{name}={v} is outside the valid range {lo} to {hi}.")


def normalise_temperature(max_temp_c: Number):
    return np.clip((np.asarray(max_temp_c, dtype=float) - C.TEMP_MIN_C) / (C.TEMP_MAX_C - C.TEMP_MIN_C), 0.0, 1.0)


def normalise_humidity(humidity_pct: Number):
    return np.clip((np.asarray(humidity_pct, dtype=float) - C.RH_MIN) / (C.RH_MAX - C.RH_MIN), 0.0, 1.0)


def compute_score(max_temp_c: Number, humidity_pct: Number, aqi: Number):
    """Vectorised score (0-100, 2 decimals). No validation; clips to ranges."""
    w = C.WEIGHTS
    aqi_n = np.clip(np.asarray(aqi, dtype=float) / C.AQI_MAX, 0.0, 1.0)
    score = (w["temperature"] * normalise_temperature(max_temp_c)
             + w["humidity"] * normalise_humidity(humidity_pct)
             + w["aqi"] * aqi_n)
    return np.round(np.clip(score, 0.0, 100.0), 2)


def calculate_risk_score(max_temp_c: float, humidity_pct: float, aqi: float) -> float:
    """Validated scalar score. Raises InvalidInputError on bad input."""
    validate_inputs(max_temp_c, humidity_pct, aqi)
    return float(compute_score(max_temp_c, humidity_pct, aqi))


def score_components(max_temp_c: float, humidity_pct: float, aqi: float) -> dict:
    """Points contributed by each component (for explaining the score)."""
    validate_inputs(max_temp_c, humidity_pct, aqi)
    w = C.WEIGHTS
    return {
        "Temperature": round(float(w["temperature"] * normalise_temperature(max_temp_c)), 2),
        "Humidity": round(float(w["humidity"] * normalise_humidity(humidity_pct)), 2),
        "AQI": round(float(w["aqi"] * min(max(float(aqi) / C.AQI_MAX, 0.0), 1.0)), 2),
    }


def categorize_score(score: float) -> str:
    """Single, configurable five-level mapping (thresholds in config.RISK_LEVELS)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        raise InvalidInputError("Score is missing.")
    if not 0.0 <= score <= 100.0:
        raise InvalidInputError(f"Score {score} outside 0-100.")
    label = C.RISK_LEVELS[0][1]
    for lower, name in C.RISK_LEVELS:
        if score >= lower:
            label = name
    return label


def categorize_series(scores: Iterable[float]) -> pd.Series:
    """Vectorised five-level mapping using the same thresholds."""
    s = pd.Series(list(scores), dtype=float)
    out = pd.Series(C.RISK_LEVELS[0][1], index=s.index, dtype=object)
    for lower, name in C.RISK_LEVELS:
        out[s >= lower] = name
    out[s.isna()] = None
    return out


def original_dataset_label(score: float) -> str:
    """The two-class rule that generated the dataset's Risk_Category column."""
    return "High" if score > C.ORIGINAL_HIGH_THRESHOLD else "Moderate"


def assess(max_temp_c: float, humidity_pct: float, aqi: float) -> dict:
    """Convenience: score + five-level category + original-rule label."""
    score = calculate_risk_score(max_temp_c, humidity_pct, aqi)
    return {
        "score": score,
        "category": categorize_score(score),
        "dataset_rule_label": original_dataset_label(score),
        "components": score_components(max_temp_c, humidity_pct, aqi),
    }
