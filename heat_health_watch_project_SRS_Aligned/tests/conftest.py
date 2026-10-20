import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    df, report = load_dataset()
    return df, report


@pytest.fixture(scope="session")
def df(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def raw_hash():
    import hashlib
    return hashlib.sha256(C.DATA_PATH.read_bytes()).hexdigest()
