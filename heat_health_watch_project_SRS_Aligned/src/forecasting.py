"""Future temperature / humidity / AQI forecasting and risk estimation.

Method (identical for each variable, documented in docs/ARCHITECTURE.md):
    value(d) = seasonal_climatology(day_of_year(d)) + phi**h * anomaly_at_last_observation
* seasonal_climatology: OLS fit on 3 yearly sine/cosine harmonics (no long-term trend is extrapolated).
* anomaly: observed minus climatology; phi is the lag-1 autocorrelation of the anomalies.
* h: days between the last observed date and the target date. The anomaly term decays to zero,
  so far-future values converge to the climatological average (an *illustrative scenario*).

This is a forecasting model (regression on calendar features + persistence of anomalies).
It is NOT the Random Forest classifier, which only classifies days whose inputs are already known.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C
from .advisories import advisory_text
from .risk_scoring import categorize_series, compute_score

FORECAST_VARS = ["Max_Temperature_C", "Mean_Temperature_C", "Relative_Humidity_%", "AQI"]
CLIP = {"Max_Temperature_C": (-5, 55), "Mean_Temperature_C": (-5, 50),
        "Relative_Humidity_%": (0, 100), "AQI": (0, 500)}
N_HARMONICS = 3
SHORT_RANGE_DAYS = 7
EXTENDED_DAYS = 30
MIN_HISTORY_DAYS = 730
SYNTHETIC_CONTINUATION_LABEL = "Synthetic continuation (generated; not observed)"


class ForecastRequestError(ValueError):
    """Raised for invalid forecast date ranges or insufficient history."""


@dataclass
class VarModel:
    coef: np.ndarray
    phi: float
    sigma: float            # std of anomalies (for uncertainty bands)
    last_anomaly: float
    last_value: float


def _design(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = np.asarray(dates.dayofyear, dtype=float)
    cols = [np.ones(len(doy))]
    for k in range(1, N_HARMONICS + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def _daily_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df.set_index(C.DATE_COL)[col].astype(float).sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full).interpolate(limit_direction="both")


def fit_var(series: pd.Series) -> VarModel:
    X = _design(series.index)
    coef, *_ = np.linalg.lstsq(X, series.values, rcond=None)
    anomaly = series.values - X @ coef
    a0, a1 = anomaly[:-1], anomaly[1:]
    phi = float(np.clip(np.corrcoef(a0, a1)[0, 1], 0.0, 0.99)) if len(anomaly) > 3 else 0.0
    return VarModel(coef=coef, phi=phi, sigma=float(np.std(anomaly)),
                    last_anomaly=float(anomaly[-1]), last_value=float(series.values[-1]))


def predict_var(model: VarModel, last_date: pd.Timestamp, target_dates: pd.DatetimeIndex, var: str):
    h = np.asarray((target_dates - last_date).days, dtype=float)
    seasonal = _design(target_dates) @ model.coef
    mean = seasonal + (model.phi ** h) * model.last_anomaly
    sd = model.sigma * np.sqrt(np.clip(1 - model.phi ** (2 * h), 0, None))
    lo, hi = CLIP[var]
    z = 1.2816  # 80 % interval under a normal-error assumption
    return (np.clip(mean, lo, hi), np.clip(mean - z * sd, lo, hi), np.clip(mean + z * sd, lo, hi), h)


def check_history(df: pd.DataFrame) -> None:
    if len(df) < MIN_HISTORY_DAYS:
        raise ForecastRequestError(
            f"Only {len(df)} daily records available; at least {MIN_HISTORY_DAYS} (about 2 years) "
            "are needed to estimate the seasonal pattern.")


def fit_all(df: pd.DataFrame) -> Dict[str, VarModel]:
    check_history(df)
    return {v: fit_var(_daily_series(df, v)) for v in FORECAST_VARS}


def generate_synthetic_continuation(
    observed_df: pd.DataFrame,
    end_date,
    seed: int = C.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a clearly labelled, seasonality-calibrated continuation after real data.

    Original rows are never changed. New rows are generated from seasonal patterns learned
    from the original observations plus correlated random anomalies. They are for scenario
    demonstration only, not a substitute for measured weather or an official forecast.
    """
    if observed_df.empty:
        raise ForecastRequestError("Cannot extend an empty dataset.")
    last_date = pd.Timestamp(observed_df[C.DATE_COL].max()).normalize()
    try:
        end = pd.Timestamp(end_date).normalize()
    except Exception as exc:
        raise ForecastRequestError(f"Invalid synthetic end date: {exc}") from exc
    if pd.isna(end):
        raise ForecastRequestError("A valid synthetic end date is required.")
    if end <= last_date:
        return pd.DataFrame(columns=[C.DATE_COL, *FORECAST_VARS, "Data_Source"])

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), end, freq="D")
    rng = np.random.default_rng(seed)
    generated: Dict[str, np.ndarray] = {}

    for var in FORECAST_VARS:
        series = _daily_series(observed_df, var)
        model = fit_var(series)
        seasonal = _design(future_dates) @ model.coef
        # Preserve seasonal shape but allow realistic day-to-day variability. The AR(1)
        # anomalies are simulated, rather than treating a smooth curve as observed data.
        innovation_sd = model.sigma * np.sqrt(max(1.0 - model.phi ** 2, 0.05))
        anomalies = np.empty(len(future_dates), dtype=float)
        previous = model.last_anomaly
        innovations = rng.normal(0.0, innovation_sd, len(future_dates))
        for i, innovation in enumerate(innovations):
            previous = model.phi * previous + innovation
            anomalies[i] = previous
        lo, hi = CLIP[var]
        generated[var] = np.clip(seasonal + anomalies, lo, hi).round(1)

    # Mean temperature should not exceed the generated daily maximum.
    generated["Mean_Temperature_C"] = np.minimum(
        generated["Mean_Temperature_C"], generated["Max_Temperature_C"] - 0.5
    ).round(1)
    out = pd.DataFrame({C.DATE_COL: future_dates, **generated})
    out["Data_Source"] = SYNTHETIC_CONTINUATION_LABEL
    return out


# ---------------------------------------------------------------- validation
def validate_request(start, end, last_obs_date) -> tuple:
    """Return (start, end) as Timestamps or raise ForecastRequestError."""
    try:
        s, e, last = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(last_obs_date)
    except Exception as exc:
        raise ForecastRequestError(f"Invalid date: {exc}") from exc
    if pd.isna(s) or pd.isna(e):
        raise ForecastRequestError("Start and end dates are required.")
    if e < s:
        raise ForecastRequestError("End date must not be earlier than the start date.")
    if s <= last:
        raise ForecastRequestError(
            f"Forecast must start after the last observed date ({last.date()}). "
            "Use the historical tabs for dates inside the dataset.")
    if (e - s).days + 1 > C.MAX_FORECAST_DAYS:
        raise ForecastRequestError(f"Range too long: at most {C.MAX_FORECAST_DAYS} days per forecast.")
    if (s - last).days > C.MAX_FORECAST_OFFSET_DAYS:
        raise ForecastRequestError(
            f"Start date is more than {C.MAX_FORECAST_OFFSET_DAYS} days beyond the data; "
            "estimates that far out carry no information beyond a seasonal average.")
    return s, e


def _reliability(h: int) -> str:
    if h <= SHORT_RANGE_DAYS:
        return "Short-range (<=7 d)"
    if h <= EXTENDED_DAYS:
        return "Extended (8-30 d)"
    return "Illustrative scenario (>30 d)"


def _status(var: str, h: int, overrides: Dict[str, float]) -> str:
    if var in overrides:
        return "User-supplied"
    return "Forecast (seasonal + AR1)" if h <= EXTENDED_DAYS else "Assumed: seasonal average"


def forecast_range(df: pd.DataFrame, start, end, overrides: Optional[Dict[str, float]] = None,
                   models: Optional[Dict[str, VarModel]] = None) -> pd.DataFrame:
    """Forecast daily values and risk for [start, end] (inclusive). Never returns an empty frame."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    for k, v in overrides.items():
        if k not in ("Relative_Humidity_%", "AQI"):
            raise ForecastRequestError(f"Only humidity and AQI can be user-supplied, not {k}.")
        lo, hi = CLIP[k]
        if not lo <= float(v) <= hi:
            raise ForecastRequestError(f"{k} override {v} outside valid range {lo}-{hi}.")
    check_history(df)   # report insufficient history before any date-range complaint
    last = df[C.DATE_COL].max()
    s, e = validate_request(start, end, last)
    models = models or fit_all(df)
    dates = pd.date_range(s, e, freq="D")
    out = pd.DataFrame({"Date": dates})
    out["Horizon_days"] = (dates - last).days
    out["Reliability"] = [_reliability(h) for h in out["Horizon_days"]]
    for var in FORECAST_VARS:
        mean, lo, hi, _ = predict_var(models[var], last, dates, var)
        if var in overrides:
            mean = np.full(len(dates), float(overrides[var])); lo = hi = mean
        out[var] = np.round(mean, 1)
        out[f"{var}_low80"] = np.round(lo, 1)
        out[f"{var}_high80"] = np.round(hi, 1)
        out[f"{var}_status"] = [_status(var, int(h), overrides) for h in out["Horizon_days"]]
    out["Risk_Score"] = compute_score(out["Max_Temperature_C"], out["Relative_Humidity_%"], out["AQI"])
    out["Risk_Score_low80"] = compute_score(out["Max_Temperature_C_low80"], out["Relative_Humidity_%_low80"], out["AQI_low80"])
    out["Risk_Score_high80"] = compute_score(out["Max_Temperature_C_high80"], out["Relative_Humidity_%_high80"], out["AQI_high80"])
    out["Score_Category_5L"] = categorize_series(out["Risk_Score"])
    out["Dataset_Rule_Label"] = np.where(out["Risk_Score"] > C.ORIGINAL_HIGH_THRESHOLD, "High", "Moderate")
    out["RF_Class"] = "Unavailable (needs Heat Index, wind and pollutant forecasts)"
    out["Advisory"] = [advisory_text(c) for c in out["Score_Category_5L"]]
    has_synthetic_history = ("Data_Source" in df.columns and
                             df["Data_Source"].astype(str).eq(SYNTHETIC_CONTINUATION_LABEL).any())
    out["Method"] = (
        "Seasonal harmonics + AR(1) anomaly decay, fitted on observed + synthetic continuation"
        if has_synthetic_history else "Seasonal harmonics + AR(1) anomaly decay"
    )
    out["History_Basis"] = (
        "Original observations + generated synthetic continuation; scenario only, not official weather"
        if has_synthetic_history else "Original observations only; experimental estimate, not official weather"
    )
    if out.empty:  # pragma: no cover - guarded by validate_request
        raise ForecastRequestError("Forecast produced no rows.")
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION) -> pd.DataFrame:
    """Chronological hold-out evaluation (models fitted on the training block only).

    Two settings per variable, each compared against two simple baselines:
    * 'short (1-7 d)'  rolling origins every 7 days inside the test block
    * 'long (test block from its start)' a single forecast origin at the end of training
    Baselines: persistence (last observed value) and seasonal-naive (value 365 days earlier).
    """
    d = df.sort_values(C.DATE_COL).reset_index(drop=True)
    cut = int(round(len(d) * (1 - test_fraction)))
    train, test = d.iloc[:cut], d.iloc[cut:]
    if len(train) < MIN_HISTORY_DAYS:
        raise ForecastRequestError("Training block too short to evaluate forecasts.")
    rows = []
    for var in FORECAST_VARS:
        full = _daily_series(d, var)
        tr = full[full.index <= train[C.DATE_COL].max()]
        model = fit_var(tr)
        te_dates = pd.DatetimeIndex(test[C.DATE_COL])
        truth_all = full.reindex(te_dates).values

        def seasonal_naive(dates):
            idx = dates - pd.Timedelta(days=365)
            return full.reindex(idx).values

        def metrics(y, p):
            m = ~(np.isnan(y) | np.isnan(p))
            e = y[m] - p[m]
            return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), int(m.sum())

        # long-range: one origin at end of training
        last_tr = tr.index.max()
        p_model, *_ = predict_var(model, last_tr, te_dates, var)
        p_pers = np.full(len(te_dates), model.last_value)
        p_sn = seasonal_naive(te_dates)
        for name, p in [("Model (seasonal+AR1)", p_model), ("Persistence", p_pers), ("Seasonal naive (t-365)", p_sn)]:
            mae, rmse, n = metrics(truth_all, p)
            rows.append({"Variable": var, "Setting": "Long range (whole test block)", "Method": name, "MAE": mae, "RMSE": rmse, "N": n})
        # short-range: rolling origins
        ys, pm, pp, ps = [], [], [], []
        origins = te_dates[::7]
        for o in origins:
            hist = full[full.index < o]
            fut = pd.date_range(o, periods=SHORT_RANGE_DAYS, freq="D")
            fut = fut[fut <= te_dates.max()]
            if len(fut) == 0:
                continue
            last_obs = hist.index.max()
            anomaly = hist.iloc[-1] - (_design(pd.DatetimeIndex([last_obs])) @ model.coef)[0]
            m_local = VarModel(model.coef, model.phi, model.sigma, float(anomaly), float(hist.iloc[-1]))
            mean, *_ = predict_var(m_local, last_obs, fut, var)
            ys.append(full.reindex(fut).values); pm.append(mean)
            pp.append(np.full(len(fut), hist.iloc[-1])); ps.append(seasonal_naive(fut))
        y, a, b, c = map(np.concatenate, (ys, pm, pp, ps))
        for name, p in [("Model (seasonal+AR1)", a), ("Persistence", b), ("Seasonal naive (t-365)", c)]:
            mae, rmse, n = metrics(y, p)
            rows.append({"Variable": var, "Setting": "Short range (1-7 d, rolling origins)", "Method": name, "MAE": mae, "RMSE": rmse, "N": n})
    return pd.DataFrame(rows)
