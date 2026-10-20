"""Preprocessing helpers. All fitting happens on training data only (inside a Pipeline)."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer

from . import config as C


def chronological_split(df: pd.DataFrame, test_fraction: float = C.TEST_FRACTION) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest (1-f) of the rows train, newest f test. No shuffling => no look-ahead leakage."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    d = df.sort_values(C.DATE_COL).reset_index(drop=True)
    cut = int(round(len(d) * (1 - test_fraction)))
    if cut < 10 or len(d) - cut < 5:
        raise ValueError("Not enough rows for a train/test split.")
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def make_imputer() -> SimpleImputer:
    """Median imputation (robust to the skewed pollutant columns); fitted inside the pipeline."""
    return SimpleImputer(strategy="median")


def feature_frame(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    features = features or C.RF_FEATURES
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing feature column(s): {missing}")
    return df[features].astype(float)
