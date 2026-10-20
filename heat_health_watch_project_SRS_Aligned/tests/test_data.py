import hashlib

import pandas as pd
import pytest

from src import config as C
from src import data_loader as D


def test_dataset_loads_with_required_columns(df):
    assert len(df) == 1428
    for col in C.REQUIRED_COLUMNS:
        assert col in df.columns


def test_date_range_and_endpoint(dataset):
    df, rep = dataset
    assert rep.date_min == "2021-01-01" and rep.date_max == "2024-11-30"
    assert df[C.DATE_COL].is_monotonic_increasing


def test_duplicate_row_removed_and_dates_unique(dataset):
    df, rep = dataset
    assert rep.rows_raw == 1429 and rep.exact_duplicate_rows == 1
    assert df[C.DATE_COL].is_unique


def test_missing_dates_reported(dataset):
    _, rep = dataset
    assert rep.missing_dates == ["2022-09-15", "2022-09-16"]


def test_no_missing_values_in_source(dataset):
    assert dataset[1].missing_values == {}


def test_target_has_only_two_original_classes(df):
    assert set(df[C.TARGET_COL].unique()) == {"Moderate", "High"}


def test_original_file_not_modified(raw_hash):
    D.load_dataset()
    assert hashlib.sha256(C.DATA_PATH.read_bytes()).hexdigest() == raw_hash


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(D.DataValidationError, match="not found"):
        D.load_raw(tmp_path / "nope.csv")


def test_missing_column_gives_clear_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2021-01-01"], "AQI": [100]}).to_csv(p, index=False)
    with pytest.raises(D.DataValidationError, match="missing required column"):
        D.load_raw(p)


def test_unparseable_dates_rejected(tmp_path):
    raw = pd.read_csv(C.DATA_PATH).head(3)
    raw["Date"] = ["x", "y", "z"]
    p = tmp_path / "baddate.csv"
    raw.to_csv(p, index=False)
    with pytest.raises(D.DataValidationError, match="unparseable"):
        D.load_raw(p)


def test_duplicate_dates_with_different_values_collapse():
    raw = D.load_raw().head(5).copy()
    dup = raw.iloc[[0]].copy()
    dup["AQI"] = dup["AQI"] + 1
    out = D.clean(pd.concat([raw, dup]))
    assert out[C.DATE_COL].is_unique and len(out) == 5


def test_five_level_column_added(df):
    assert set(df["Score_Category_5L"].unique()) <= set(C.LEVEL_ORDER)


def test_csv_export_roundtrip(df):
    b = D.to_csv_bytes(df.head(10))
    back = pd.read_csv(pd.io.common.BytesIO(b))
    assert len(back) == 10 and list(back.columns) == list(df.columns)
