"""Train, compare, save the model, and log results to PostgreSQL.

Run:  python src/train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (FEATURES, FIG_DIR, METRICS_PATH, MODEL_PATH,
                    RANDOM_STATE, TARGET, TEST_SIZE)
from data_loader import load_clean
from db import log_model_runs


def make_pipeline(model) -> Pipeline:
    # Imputation and scaling live INSIDE the pipeline, so they are fitted on
    # training data only (this prevents data leakage into the test set).
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_clean()                     # <- comes from PostgreSQL
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = make_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 3),
            "cv_roc_auc_std": round(float(cv_auc.std()), 3),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "test_recall": round(float(recall_score(y_test, pred)), 3),
            "test_f1": round(float(f1_score(y_test, pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        }
        fitted[name] = pipe
        print(name, results[name])

    # Choose the winner by cross-validated AUC, NOT by the test score
    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    joblib.dump(fitted[best], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({**results, "selected_model": best}, indent=2))
    log_model_runs(results, best, n_train=len(X_train), n_test=len(X_test))

    ConfusionMatrixDisplay.from_estimator(fitted[best], X_test, y_test)
    plt.title(f"Confusion Matrix - {best}")
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Selected {best}; saved model, metrics.json and rows in model_runs")


if __name__ == "__main__":
    main()
