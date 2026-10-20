"""Random Forest risk classifier (genuine scikit-learn RandomForestClassifier).

Target: the dataset's own `Risk_Category` column (only 'Moderate' and 'High' exist).
Split: chronological 80/20 (oldest rows train, newest rows test).
Run `python -m src.risk_classifier` to retrain and save the model + metrics.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Optional

import joblib
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from . import config as C
from .preprocessing import chronological_split, feature_frame, make_imputer


def build_pipeline(seed: int = C.RANDOM_SEED) -> Pipeline:
    return Pipeline([
        ("impute", make_imputer()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    ])


def train_and_evaluate(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> Dict:
    """Fit on the chronological training block and evaluate ONLY on the held-out block."""
    if C.TARGET_COL not in df.columns:
        raise KeyError(f"Target column {C.TARGET_COL} not found.")
    train, test = chronological_split(df)
    classes = sorted(train[C.TARGET_COL].unique())
    if len(classes) < 2:
        raise ValueError("Training data contains fewer than two classes.")

    pipe = build_pipeline(seed)
    pipe.fit(feature_frame(train), train[C.TARGET_COL])

    X_test, y_test = feature_frame(test), test[C.TARGET_COL]
    pred = pipe.predict(X_test)
    labels = list(pipe.classes_)
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    majority = y_test.value_counts().max() / len(y_test)

    rf = pipe.named_steps["rf"]
    importances = dict(sorted(zip(C.RF_FEATURES, map(float, rf.feature_importances_)),
                              key=lambda kv: kv[1], reverse=True))
    metrics = {
        "target": C.TARGET_COL,
        "features": C.RF_FEATURES,
        "classes": labels,
        "split": "chronological 80/20 (oldest rows train, newest rows test)",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_period": [str(train[C.DATE_COL].min().date()), str(train[C.DATE_COL].max().date())],
        "test_period": [str(test[C.DATE_COL].min().date()), str(test[C.DATE_COL].max().date())],
        "test_class_counts": {k: int(v) for k, v in y_test.value_counts().items()},
        "accuracy_test": float(accuracy_score(y_test, pred)),
        "majority_class_baseline_accuracy": float(majority),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])} for i, lab in enumerate(labels)},
        "macro_f1": float(np.mean(f)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "feature_importance": importances,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "note": ("Risk_Category in the supplied data is a deterministic threshold on "
                 "Heat_Health_Risk_Score (>55 = High), which is itself a linear function of Max "
                 "temperature, humidity and AQI. The Random Forest therefore re-learns that rule "
                 "approximately from correlated inputs; it is not independent clinical validation."),
    }
    return {"pipeline": pipe, "metrics": metrics}


def save_artifacts(result: Dict) -> None:
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], C.MODEL_PATH)
    C.METRICS_PATH.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")


def load_or_train(df: pd.DataFrame, force_retrain: bool = False) -> Dict:
    """Load the saved model if it is usable; otherwise retrain (reproducible workflow)."""
    if not force_retrain and C.MODEL_PATH.exists() and C.METRICS_PATH.exists():
        try:
            pipe = joblib.load(C.MODEL_PATH)
            metrics = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
            same_features = list(metrics.get("features", [])) == C.RF_FEATURES
            same_sklearn = metrics.get("sklearn_version") == sklearn.__version__
            if same_features and same_sklearn:
                return {"pipeline": pipe, "metrics": metrics, "source": "loaded from models/"}
        except Exception:
            pass  # incompatible scikit-learn version etc. -> retrain below
    result = train_and_evaluate(df)
    try:
        save_artifacts(result)
    except OSError:
        pass
    result["source"] = "trained now"
    return result


def validate_feature_row(row: Dict[str, float]) -> Dict[str, float]:
    """Check that every required feature is present and a finite number."""
    clean = {}
    for f in C.RF_FEATURES:
        if f not in row or row[f] is None:
            raise ValueError(f"Missing feature: {f}")
        v = float(row[f])
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Feature {f} must be finite.")
        if f in C.VALID_RANGES and not C.VALID_RANGES[f][0] <= v <= C.VALID_RANGES[f][1]:
            raise ValueError(f"Feature {f}={v} outside valid range {C.VALID_RANGES[f]}.")
        clean[f] = v
    return clean


def predict_one(pipeline: Pipeline, row: Dict[str, float]) -> Dict:
    """Predict the learned (two-class) label with class probabilities.

    Probabilities are the forest's vote shares, not calibrated medical risk.
    """
    clean = validate_feature_row(row)
    X = pd.DataFrame([clean], columns=C.RF_FEATURES)
    label = pipeline.predict(X)[0]
    proba = dict(zip(pipeline.classes_, map(float, pipeline.predict_proba(X)[0])))
    return {"predicted_label": str(label), "vote_share": proba}


if __name__ == "__main__":  # pragma: no cover
    from .data_loader import load_dataset
    data, _ = load_dataset()
    res = train_and_evaluate(data)
    save_artifacts(res)
    m = res["metrics"]
    print(f"Test accuracy: {m['accuracy_test']:.4f} (majority baseline {m['majority_class_baseline_accuracy']:.4f})")
    print("Confusion matrix", m["classes"], m["confusion_matrix"])
