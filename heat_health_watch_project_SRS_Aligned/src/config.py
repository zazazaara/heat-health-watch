"""Central configuration: paths, score formula constants, thresholds, feature lists.

Everything that a viva examiner might ask "where is that defined?" lives here.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "delhi_heatwave_airquality.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_risk_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "rf_metrics.json"

# ---- Dataset columns -------------------------------------------------------
DATE_COL = "Date"
TARGET_COL = "Risk_Category"
SCORE_COL = "Heat_Health_Risk_Score"
REQUIRED_COLUMNS = [
    "Date", "Max_Temperature_C", "Min_Temperature_C", "Mean_Temperature_C",
    "Relative_Humidity_%", "Heat_Index", "Wind_Speed", "PM2.5", "PM10", "NO2",
    "SO2", "CO", "Ozone", "AQI", "Heat_Health_Risk_Score", "Risk_Category",
]
# The 11 attributes used by the original mini-project report (Section 1.6).
RF_FEATURES = [
    "Mean_Temperature_C", "Relative_Humidity_%", "Heat_Index", "Wind_Speed",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI",
]

# ---- Heat-Health Risk Score (recovered from the supplied dataset) ----------
# Score = 45*NormTemp + 25*NormHumidity + 30*(AQI/500), range 0-100.
# These constants reproduce the stored Heat_Health_Risk_Score column to within
# 0.005 (rounding) on every unique day. See docs/ARCHITECTURE.md.
TEMP_MIN_C, TEMP_MAX_C = 9.1, 47.9          # Norm_Temp  = (Tmax - 9.1) / 38.8
RH_MIN, RH_MAX = 16.1, 99.3                 # Norm_Humid = (RH - 16.1) / 83.2
AQI_MAX = 500.0
WEIGHTS = {"temperature": 45.0, "humidity": 25.0, "aqi": 30.0}   # sum = 100

# Original dataset label rule: 'High' if score > 55 else 'Moderate'.
ORIGINAL_HIGH_THRESHOLD = 55.0

# ---- Five project-defined demonstration levels (NOT an official standard) ---
# (lower bound inclusive, label). Edit here and everything follows.
RISK_LEVELS = [
    (0.0, "Very Low"),
    (20.0, "Low"),
    (40.0, "Moderate"),
    (60.0, "High"),
    (80.0, "Severe"),
]
LEVEL_ORDER = [name for _, name in RISK_LEVELS]
LEVEL_COLORS = {
    "Very Low": "#2e9e5b", "Low": "#8bc34a", "Moderate": "#f2b705",
    "High": "#ef6c00", "Severe": "#c62828",
}

# ---- Input validation ranges (physically plausible for Delhi) --------------
VALID_RANGES = {
    "Max_Temperature_C": (-5.0, 55.0),
    "Mean_Temperature_C": (-5.0, 50.0),
    "Relative_Humidity_%": (0.0, 100.0),
    "AQI": (0.0, 500.0),
}

RANDOM_SEED = 42
TEST_FRACTION = 0.20
MAX_FORECAST_DAYS = 180
MAX_FORECAST_OFFSET_DAYS = 1100   # how far past the data end a forecast may start

DISCLAIMER = (
    "General educational guidance only. It is not medical advice or a diagnosis. "
    "Follow official IMD / CPCB / local health-authority warnings and consult a "
    "health professional if you feel unwell."
)
