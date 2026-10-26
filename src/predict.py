"""Score one patient with the saved model and log it to PostgreSQL.

Run:  python src/predict.py
"""
import json

import joblib
import pandas as pd

from config import FEATURES, MODEL_META_PATH, MODEL_PATH
from db import log_prediction


def _threshold() -> float:
    """Screening threshold saved by tune.py; falls back to 0.5 if not available."""
    try:
        return float(json.loads(MODEL_META_PATH.read_text())["threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.5


def predict_risk(record: dict, log: bool = True) -> float:
    """Return the probability of diabetes. Use None for unknown values."""
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([record], columns=FEATURES).astype(float)
    probability = float(model.predict_proba(row)[0, 1])
    if log:
        log_prediction(record, probability)
    return probability


def predict_label(record: dict, log: bool = True) -> dict:
    """Probability plus a high-risk flag using the screening threshold."""
    p = predict_risk(record, log=log)
    thr = _threshold()
    return {"risk_probability": round(p, 3), "threshold": thr, "flagged_high_risk": p >= thr}


if __name__ == "__main__":
    example = {
        "pregnancies": 2, "glucose": 150, "blood_pressure": 80,
        "skin_thickness": None, "insulin": None, "bmi": 33.5,
        "diabetes_pedigree_function": 0.5, "age": 45,
    }
    result = predict_label(example)
    print(f"Estimated diabetes risk: {result['risk_probability']:.1%}")
    print(f"Screening threshold {result['threshold']:.2f} -> "
          f"{'FLAG for follow-up' if result['flagged_high_risk'] else 'below threshold'}")
