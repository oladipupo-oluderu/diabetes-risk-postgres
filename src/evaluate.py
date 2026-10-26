"""Evaluation helpers: threshold choice, metrics, bootstrap CI and plots.

Everything here is a small, testable function so train/tune scripts stay readable.
"""
import matplotlib
matplotlib.use("Agg")  # no display needed (works inside Docker)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             brier_score_loss, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score,
                             precision_recall_curve, roc_auc_score)


# ----------------------------------------------------------------- numbers
def choose_threshold(y_true, proba, target_recall: float = 0.80) -> float:
    """Pick the decision threshold for a screening goal.

    Among all thresholds that still catch at least `target_recall` of the true
    positives, return the one with the highest precision (fewest false alarms).
    Ties go to the higher threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    ok = np.flatnonzero(recall >= target_recall)
    if len(ok) == 0:
        return 0.5
    best = max(ok, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[best])


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """Confusion-matrix based metrics for a given probability threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float((tp + tn) / len(y_true)), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "specificity": round(float(specificity), 3),
        "f1": round(float(f1_score(y_true, pred)), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_auc_ci(y_true, proba, n_boot: int = 1000, seed: int = 42):
    """95% bootstrap confidence interval for ROC-AUC.

    A test set of ~150 patients is small, so a single AUC number hides a lot of
    uncertainty. Resampling the test set shows how much it could move.
    """
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:  # need both classes to compute AUC
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def brier(y_true, proba) -> float:
    """Brier score: mean squared error of the probabilities (lower is better)."""
    return round(float(brier_score_loss(y_true, proba)), 3)


# ------------------------------------------------------------------- plots
def plot_roc_pr(proba_by_model: dict, y_true, path) -> None:
    """ROC and precision-recall curves for several models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in proba_by_model.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC curve (test set)")
    axes[1].set_title("Precision-recall curve (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_calibration(y_true, proba, path, title="Calibration") -> None:
    """Do predicted risks match observed frequencies? (diagonal = perfect)"""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label=f"Model (Brier {brier(y_true, proba)})")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed diabetes rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(y_true, proba, threshold: float, path, title="Confusion matrix") -> None:
    pred = (np.asarray(proba) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, pred, colorbar=False)
    plt.title(f"{title} (threshold {threshold:.2f})")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_permutation_importance(pipe, X_test, y_test, path, seed: int = 42):
    """Which features does the model actually rely on?

    Shuffle one feature at a time on the TEST set and measure how much AUC drops.
    Works for any model, and runs the imputer/scaler inside the pipeline.
    """
    result = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                    n_repeats=30, random_state=seed)
    order = np.argsort(result.importances_mean)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(X_test.columns)[order], result.importances_mean[order],
             xerr=result.importances_std[order], color="steelblue")
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title("Permutation importance (test set)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return dict(zip(X_test.columns, np.round(result.importances_mean, 4).tolist()))
