"""Synthetic environmental scenarios for demos and tests.

SYNTHETIC DATA — NOT OBSERVED WEATHER. Rows are drawn inside documented plausible
envelopes per target level, scored with the SAME formula as real data, and accepted only
if the computed score really falls in the requested band (rejection sampling), so no
impossible values are invented just to balance classes. Real historical rows are never
modified or mixed in.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .risk_scoring import categorize_series, compute_score

SYNTHETIC_LABEL = "SYNTHETIC — NOT OBSERVED WEATHER"
MAX_PER_LEVEL = 200

# (Tmax range, RH range, AQI range) envelopes per target level.
ENVELOPES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    "Very Low": ((9.1, 22.0), (20.0, 70.0), (30.0, 150.0)),
    "Low":      ((14.0, 32.0), (20.0, 80.0), (50.0, 260.0)),
    "Moderate": ((24.0, 40.0), (20.0, 85.0), (100.0, 420.0)),
    "High":     ((32.0, 45.0), (20.0, 85.0), (150.0, 480.0)),
    "Severe":   ((40.0, 47.9), (25.0, 80.0), (250.0, 500.0)),
}


def _plausible(tmax: float, rh: float, aqi: float) -> bool:
    """Basic physical sanity rules (reject, rather than clip, implausible combinations)."""
    if tmax >= 42 and rh > 75:      # extreme heat with very high humidity is not typical for Delhi
        return False
    if tmax <= 12 and rh < 25:      # very cold and very dry together is unusual here
        return False
    return True


def generate_scenarios(n_per_level: int = 10, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Generate n_per_level scenarios for EACH of the five levels (reproducible for a seed)."""
    if not isinstance(n_per_level, (int, np.integer)) or not 1 <= n_per_level <= MAX_PER_LEVEL:
        raise ValueError(f"n_per_level must be an integer between 1 and {MAX_PER_LEVEL}.")
    rng = np.random.default_rng(seed)
    bounds = {name: (lo, (C.RISK_LEVELS[i + 1][0] if i + 1 < len(C.RISK_LEVELS) else 100.0000001))
              for i, (lo, name) in enumerate(C.RISK_LEVELS)}
    rows = []
    for level in C.LEVEL_ORDER:
        (t_lo, t_hi), (h_lo, h_hi), (a_lo, a_hi) = ENVELOPES[level]
        lo, hi = bounds[level]
        accepted, attempts = 0, 0
        while accepted < n_per_level:
            attempts += 1
            if attempts > 20000 * n_per_level:
                raise RuntimeError(f"Could not generate enough plausible '{level}' scenarios.")
            # Round BEFORE scoring so the published numbers are exactly what was scored.
            tmax = round(rng.uniform(t_lo, t_hi), 1)
            rh = round(rng.uniform(h_lo, h_hi), 1)
            aqi = float(round(rng.uniform(a_lo, a_hi)))
            if not _plausible(tmax, rh, aqi):
                continue
            score = float(compute_score(tmax, rh, aqi))
            if not lo <= score < hi:
                continue
            mean_t = tmax - rng.uniform(4.0, 9.0)
            pm25 = aqi * rng.uniform(0.30, 0.50)
            pm10 = pm25 * rng.uniform(1.8, 2.4)
            rows.append({
                "Target_Level": level,
                "Max_Temperature_C": round(tmax, 1), "Mean_Temperature_C": round(mean_t, 1),
                "Relative_Humidity_%": round(rh, 1), "AQI": round(aqi, 0),
                "PM2.5": round(pm25, 1), "PM10": round(pm10, 1),
            })
            accepted += 1
    df = pd.DataFrame(rows)
    df["Risk_Score"] = compute_score(df["Max_Temperature_C"], df["Relative_Humidity_%"], df["AQI"])
    df["Score_Category_5L"] = categorize_series(df["Risk_Score"])
    df.insert(0, "Scenario_ID", [f"SYN-{i + 1:04d}" for i in range(len(df))])
    df["Data_Source"] = SYNTHETIC_LABEL
    return df


def validate_synthetic(df: pd.DataFrame) -> list:
    """Return a list of problems (empty list = valid)."""
    problems = []
    for col, (lo, hi) in C.VALID_RANGES.items():
        if col in df.columns and not df[col].between(lo, hi).all():
            problems.append(f"{col} has values outside {lo}-{hi}.")
    if not df["Risk_Score"].between(0, 100).all():
        problems.append("Risk_Score outside 0-100.")
    if not (df["Data_Source"] == SYNTHETIC_LABEL).all():
        problems.append("Rows are not labelled as synthetic.")
    if (df["Score_Category_5L"] != df["Target_Level"]).any():
        problems.append("Some scenarios do not fall in their target category after rounding.")
    return problems
