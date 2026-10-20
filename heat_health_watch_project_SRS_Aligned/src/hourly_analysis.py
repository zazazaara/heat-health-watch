"""Hourly risk profile — a SIMULATION built from DAILY values.

The supplied dataset has one row per day, so true hourly observations do not exist.
This module disaggregates a day's (Tmin, Tmax, mean RH, mean AQI) into 24 hourly values
using documented assumptions:

* Temperature: smooth diurnal cycle, minimum at `tmin_hour` (default 05:00) and maximum at
  `tmax_hour` (default 15:00), rising and falling with half-cosine shapes.
* Relative humidity: the day's dew point is held constant (derived from mean RH at the mean
  temperature with the Magnus formula), so RH falls when it is hot and rises when it cools.
* AQI: daily mean multiplied by a smooth profile (higher late evening / night, small morning
  traffic bump) plus +/-3 % seeded noise. This shape is an ASSUMPTION, not measured.
* Hourly score: the same daily formula applied to the instantaneous temperature, humidity
  and AQI. The peak-risk hour is whatever the profile produces; it is never hard-coded.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .advisories import advisory_text
from .risk_scoring import categorize_series, compute_score

SIM_LABEL = "SIMULATED — derived from daily values, not observed hourly data"
LEVEL_RANK = {name: i for i, name in enumerate(C.LEVEL_ORDER)}


class HourlyInputError(ValueError):
    """Raised for invalid hourly-profile inputs."""


def _es(t_c):
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def _dewpoint(t_c: float, rh: float) -> float:
    g = math.log(max(rh, 1.0) / 100.0) + 17.62 * t_c / (243.12 + t_c)
    return 243.12 * g / (17.62 - g)


def _validate(tmin, tmax, rh_mean, aqi_mean, tmin_hour, tmax_hour):
    for name, v in [("Tmin", tmin), ("Tmax", tmax), ("Mean RH", rh_mean), ("Mean AQI", aqi_mean)]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise HourlyInputError(f"{name} is missing.")
    if tmin > tmax:
        raise HourlyInputError("Minimum temperature cannot exceed maximum temperature.")
    if not -5 <= tmax <= 55 or not -5 <= tmin <= 55:
        raise HourlyInputError("Temperatures must be between -5 and 55 °C.")
    if not 0 <= rh_mean <= 100:
        raise HourlyInputError("Relative humidity must be between 0 and 100 %.")
    if not 0 <= aqi_mean <= 500:
        raise HourlyInputError("AQI must be between 0 and 500.")
    if not (0 <= tmin_hour <= 23 and 0 <= tmax_hour <= 23 and tmin_hour < tmax_hour):
        raise HourlyInputError("Need 0 <= tmin_hour < tmax_hour <= 23.")


def simulate_hourly(tmin: float, tmax: float, rh_mean: float, aqi_mean: float,
                    date: Optional[pd.Timestamp] = None, tmin_hour: int = 5, tmax_hour: int = 15,
                    seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Return a 24-row DataFrame of simulated hourly values, score, category and advisory."""
    _validate(tmin, tmax, rh_mean, aqi_mean, tmin_hour, tmax_hour)
    rise = tmax_hour - tmin_hour
    fall = 24 - rise
    hours = np.arange(24)
    temp = np.empty(24)
    for i, h in enumerate(hours):
        if tmin_hour <= h <= tmax_hour:
            frac = (h - tmin_hour) / rise
            temp[i] = tmin + (tmax - tmin) * 0.5 * (1 - math.cos(math.pi * frac))
        else:
            since = (h - tmax_hour) % 24
            frac = since / fall
            temp[i] = tmin + (tmax - tmin) * 0.5 * (1 + math.cos(math.pi * frac))
    t_mean = (tmin + tmax) / 2.0
    td = _dewpoint(t_mean, rh_mean)
    rh = np.clip(100.0 * _es(td) / _es(temp), 5.0, 100.0)
    rng = np.random.default_rng(seed)
    shape = 1 + 0.15 * np.cos(2 * np.pi * (hours - 22) / 24) + 0.07 * np.cos(4 * np.pi * (hours - 9) / 24)
    aqi = np.clip(aqi_mean * shape * (1 + rng.uniform(-0.03, 0.03, 24)), 0, 500)
    score = compute_score(temp, rh, aqi)
    out = pd.DataFrame({
        "Date": pd.Timestamp(date).normalize() if date is not None else pd.NaT,
        "Hour": hours,
        "Time": [f"{h:02d}:00" for h in hours],
        "Temperature_C": np.round(temp, 1),
        "Relative_Humidity_%": np.round(rh, 1),
        "AQI": np.round(aqi, 0),
        "Risk_Score": score,
    })
    out["Score_Category_5L"] = categorize_series(out["Risk_Score"])
    out["Advisory"] = [advisory_text(c) for c in out["Score_Category_5L"]]
    out["Data_Source"] = SIM_LABEL
    return out


def peak_hour(df: pd.DataFrame) -> dict:
    """Hour with the highest simulated score (first one if tied)."""
    i = int(df["Risk_Score"].values.argmax())
    r = df.iloc[i]
    return {"hour": int(r["Hour"]), "time": r["Time"], "score": float(r["Risk_Score"]),
            "category": r["Score_Category_5L"]}


def risk_windows(df: pd.DataFrame, min_level: str = "High") -> List[dict]:
    """Contiguous hour ranges whose category is >= min_level (empty list if none)."""
    if min_level not in LEVEL_RANK:
        raise HourlyInputError(f"Unknown level {min_level!r}.")
    flag = df["Score_Category_5L"].map(LEVEL_RANK) >= LEVEL_RANK[min_level]
    windows, start = [], None
    for h, f in zip(df["Hour"], flag):
        if f and start is None:
            start = h
        if not f and start is not None:
            windows.append((start, h - 1)); start = None
    if start is not None:
        windows.append((start, int(df["Hour"].iloc[-1])))
    return [{"from": f"{a:02d}:00", "to": f"{b:02d}:59", "hours": b - a + 1} for a, b in windows]
