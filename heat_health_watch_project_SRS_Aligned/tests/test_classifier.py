import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import preprocessing as P
from src import risk_classifier as RC


@pytest.fixture(scope="module")
def result(df):
    return RC.train_and_evaluate(df)


def test_uses_genuine_random_forest(result):
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(result["pipeline"].named_steps["rf"], RandomForestClassifier)


def test_learns_only_the_original_two_classes(result, df):
    assert set(result["pipeline"].classes_) == set(df[C.TARGET_COL].unique()) == {"Moderate", "High"}
    assert len(result["pipeline"].classes_) == 2   # never pretends to learn five classes


def test_chronological_split_has_no_overlap(df):
    tr, te = P.chronological_split(df)
    assert tr[C.DATE_COL].max() < te[C.DATE_COL].min()
    assert len(tr) + len(te) == len(df)


def test_split_rejects_bad_fraction(df):
    with pytest.raises(ValueError):
        P.chronological_split(df, 1.5)


def test_metrics_are_from_held_out_block(result):
    m = result["metrics"]
    assert sum(map(sum, m["confusion_matrix"])) == m["test_rows"] == 286
    assert m["train_period"][1] < m["test_period"][0]
    assert m["accuracy_test"] > m["majority_class_baseline_accuracy"]


def test_metrics_internally_consistent(result):
    m = result["metrics"]
    cm = np.array(m["confusion_matrix"])
    assert np.isclose(m["accuracy_test"], np.trace(cm) / cm.sum())
    for lab, i in zip(m["classes"], range(len(m["classes"]))):
        assert np.isclose(m["per_class"][lab]["recall"], cm[i, i] / cm[i].sum())
        assert np.isclose(m["per_class"][lab]["precision"], cm[i, i] / cm[:, i].sum())


def test_feature_importance_covers_all_features(result):
    fi = result["metrics"]["feature_importance"]
    assert set(fi) == set(C.RF_FEATURES) and np.isclose(sum(fi.values()), 1.0)


def test_training_is_reproducible(df):
    a, b = RC.train_and_evaluate(df), RC.train_and_evaluate(df)
    assert a["metrics"]["confusion_matrix"] == b["metrics"]["confusion_matrix"]


def test_example_prediction(result, df):
    row = df.iloc[-1][C.RF_FEATURES].to_dict()
    out = RC.predict_one(result["pipeline"], row)
    assert out["predicted_label"] in {"Moderate", "High"}
    assert np.isclose(sum(out["vote_share"].values()), 1.0)


def test_prediction_rejects_missing_or_bad_input(result, df):
    row = df.iloc[0][C.RF_FEATURES].to_dict()
    bad = dict(row); bad.pop("AQI")
    with pytest.raises(ValueError, match="Missing feature"):
        RC.predict_one(result["pipeline"], bad)
    bad = dict(row); bad["AQI"] = float("nan")
    with pytest.raises(ValueError):
        RC.predict_one(result["pipeline"], bad)
    bad = dict(row); bad["Relative_Humidity_%"] = 150
    with pytest.raises(ValueError, match="outside valid range"):
        RC.predict_one(result["pipeline"], bad)


def test_imputer_is_fitted_on_training_only(df):
    pipe = RC.build_pipeline()
    tr, te = P.chronological_split(df)
    pipe.fit(P.feature_frame(tr), tr[C.TARGET_COL])
    med_train = tr[C.RF_FEATURES].median().values
    assert np.allclose(pipe.named_steps["impute"].statistics_, med_train)


def test_nan_inputs_handled_by_pipeline_imputer(df):
    res = RC.train_and_evaluate(df)
    X = df[C.RF_FEATURES].head(3).copy(); X.iloc[0, 0] = np.nan
    assert len(res["pipeline"].predict(X)) == 3


def test_save_and_load_roundtrip(tmp_path, monkeypatch, df):
    monkeypatch.setattr(C, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(C, "METRICS_PATH", tmp_path / "m.json")
    r1 = RC.load_or_train(df)
    assert r1["source"] == "trained now" and (tmp_path / "m.joblib").exists()
    r2 = RC.load_or_train(df)
    assert r2["source"].startswith("loaded")
    assert json.loads((tmp_path / "m.json").read_text())["accuracy_test"] == r1["metrics"]["accuracy_test"]


def test_corrupt_model_file_triggers_retrain(tmp_path, monkeypatch, df):
    (tmp_path / "m.joblib").write_bytes(b"not a model")
    (tmp_path / "m.json").write_text("{}")
    monkeypatch.setattr(C, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(C, "METRICS_PATH", tmp_path / "m.json")
    assert RC.load_or_train(df)["source"] == "trained now"


def test_saved_project_model_matches_current_data(df):
    saved = json.loads(C.METRICS_PATH.read_text())
    fresh = RC.train_and_evaluate(df)["metrics"]
    assert saved["confusion_matrix"] == fresh["confusion_matrix"]


def test_model_saved_under_other_sklearn_version_is_retrained(tmp_path, monkeypatch, df):
    monkeypatch.setattr(C, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(C, "METRICS_PATH", tmp_path / "m.json")
    RC.load_or_train(df)
    meta = json.loads((tmp_path / "m.json").read_text())
    meta["sklearn_version"] = "0.0.1"
    (tmp_path / "m.json").write_text(json.dumps(meta))
    assert RC.load_or_train(df)["source"] == "trained now"
