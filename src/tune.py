"""Modelling stage 2: tune, select, set a screening threshold, interpret, save.

Run:  python src/tune.py        (also called by src/pipeline.py)

Leakage rules followed here:
  * The test set is touched ONCE, at the very end.
  * Hyperparameters are chosen by cross-validation on the training set only.
  * The decision threshold is chosen from out-of-fold training predictions.
"""
import json
from datetime import datetime, timezone

import joblib
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluate as ev
from config import (FEATURES, FIG_DIR, MODEL_CARD_PATH, MODEL_META_PATH, MODEL_PATH,
                    N_BOOTSTRAP, N_SEARCH_ITER, RANDOM_STATE, TARGET, TARGET_RECALL,
                    TEST_SIZE, TUNED_METRICS_PATH)
from data_loader import load_clean
from db import apply_schema, log_tuned_runs


def make_pipeline(model) -> Pipeline:
    # Imputation + scaling are INSIDE the pipeline, so every CV fold fits them
    # on its own training part only (no leakage).
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def search_spaces() -> dict:
    """Model + hyperparameter distributions to search. Keys use the pipeline step name."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": loguniform(1e-2, 1e2)},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 4, 5, 6, 8, None],
                "model__min_samples_leaf": [1, 3, 5, 10, 15],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__max_iter": [100, 200, 300],
                "model__min_samples_leaf": [10, 20, 40],
                "model__l2_regularization": [0.0, 1.0, 5.0],
            },
        ),
    }


def _plain(params: dict) -> dict:
    """Make numpy scalars JSON-safe."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def write_model_card(meta: dict, results: dict, at_default: dict, at_thr: dict,
                     brier_score: float, importance: dict) -> None:
    best = results[meta["model_name"]]
    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
    MODEL_CARD_PATH.write_text(f"""# Model card: diabetes risk screening prototype

**Model:** {meta['model_name']}  |  **Trained:** {meta['trained_at']}
**Data:** Pima Indians Diabetes dataset (768 women, 21+, Pima heritage, Arizona USA), read from PostgreSQL view `clean_diabetes`.

## Intended use
Research / portfolio demonstration of an SQL-backed ML pipeline. **Not a diagnostic tool.**

## Performance (held-out test set, n = {best['n_test']})
| Metric | Value |
|---|---|
| ROC-AUC | {best['test_roc_auc']} (95% bootstrap CI {best['auc_ci_low']} - {best['auc_ci_high']}) |
| Cross-validated ROC-AUC (train) | {best['cv_roc_auc_mean']} +/- {best['cv_roc_auc_std']} |
| Brier score | {brier_score} |

| At threshold | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|
| 0.50 (default) | {at_default['recall']} | {at_default['precision']} | {at_default['specificity']} | {at_default['f1']} |
| {at_thr['threshold']} (screening, target recall {TARGET_RECALL}) | {at_thr['recall']} | {at_thr['precision']} | {at_thr['specificity']} | {at_thr['f1']} |

The screening threshold was chosen from out-of-fold training predictions, not from the test set.

## Most influential features (permutation importance)
{', '.join(f'{k} ({v})' for k, v in top)}

## Limitations
- Small dataset: the test set holds {best['n_test']} patients, so metrics carry wide uncertainty (see the CI).
- Single population (Pima women in Arizona): no evidence of generalisation to Nigerian or other cohorts.
- Zeros in glucose, blood pressure, skin thickness, insulin and BMI were treated as missing and imputed with the training median.
- Outcome definition and collection year predate current diagnostic criteria.
""")


def main() -> None:
    apply_schema()  # idempotent: adds the new model_runs columns on older databases
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- 1. tune every candidate with cross-validated random search ----
    results, fitted, test_proba = {}, {}, {}
    for name, (model, space) in search_spaces().items():
        search = RandomizedSearchCV(
            make_pipeline(model), space, n_iter=N_SEARCH_ITER, scoring="roc_auc",
            cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        lo, hi = ev.bootstrap_auc_ci(y_test, proba, N_BOOTSTRAP)
        results[name] = {
            "best_params": _plain(search.best_params_),
            "cv_roc_auc_mean": round(float(search.best_score_), 3),
            "cv_roc_auc_std": round(float(cv_std), 3),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "test_recall": round(float(recall_score(y_test, pred)), 3),
            "test_f1": round(float(f1_score(y_test, pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "auc_ci_low": lo, "auc_ci_high": hi,
            "n_train": len(X_train), "n_test": len(X_test),
        }
        fitted[name], test_proba[name] = best, proba
        print(f"{name}: CV AUC {results[name]['cv_roc_auc_mean']} | "
              f"test AUC {results[name]['test_roc_auc']} ({lo}-{hi}) | {search.best_params_}")

    # ---- 2. select by cross-validated AUC (never by test score) ----
    selected = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    final, final_proba = fitted[selected], test_proba[selected]

    # ---- 3. screening threshold from OUT-OF-FOLD training predictions ----
    oof = cross_val_predict(clone(final), X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = ev.choose_threshold(y_train, oof, TARGET_RECALL)
    at_default = ev.metrics_at_threshold(y_test, final_proba, 0.5)
    at_thr = ev.metrics_at_threshold(y_test, final_proba, threshold)
    print(f"Selected {selected}; screening threshold {threshold:.3f}")
    print("Test @0.50     :", at_default)
    print(f"Test @{threshold:.3f}:", at_thr)

    # ---- 4. interpretation, calibration, curves ----
    importance = ev.plot_permutation_importance(
        final, X_test, y_test, FIG_DIR / "permutation_importance.png")
    ev.plot_roc_pr(test_proba, y_test, FIG_DIR / "roc_pr_curves.png")
    ev.plot_calibration(y_test, final_proba, FIG_DIR / "calibration_curve.png",
                        title=f"Calibration - {selected}")
    ev.plot_confusion(y_test, final_proba, threshold,
                      FIG_DIR / "confusion_matrix_tuned.png", title=selected)
    brier_score = ev.brier(y_test, final_proba)

    # ---- 5. persist: model file, metadata, JSON metrics, DB rows, model card ----
    joblib.dump(final, MODEL_PATH)
    meta = {
        "model_name": selected,
        "threshold": round(threshold, 4),
        "target_recall": TARGET_RECALL,
        "features": FEATURES,
        "best_params": results[selected]["best_params"],
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    TUNED_METRICS_PATH.write_text(json.dumps({
        "models": results, "selected_model": selected,
        "test_at_default_threshold": at_default,
        "test_at_screening_threshold": at_thr,
        "brier_score": brier_score,
        "permutation_importance": importance,
    }, indent=2))
    log_tuned_runs(results, selected, threshold, len(X_train), len(X_test))
    write_model_card(meta, results, at_default, at_thr, brier_score, importance)
    print(f"Saved {MODEL_PATH.name}, model_meta.json, metrics_tuned.json, MODEL_CARD.md")


if __name__ == "__main__":
    main()
