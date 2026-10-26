import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate as ev  # noqa: E402


def test_choose_threshold_meets_target_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.9])
    thr = ev.choose_threshold(y, p, target_recall=0.75)
    assert ev.metrics_at_threshold(y, p, thr)["recall"] >= 0.75


def test_choose_threshold_prefers_precision_among_valid_thresholds():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    thr = ev.choose_threshold(y, p, target_recall=1.0)
    m = ev.metrics_at_threshold(y, p, thr)
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_metrics_at_threshold_counts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    m = ev.metrics_at_threshold(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == 0.5


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 200), 0, 1)
    lo, hi = ev.bootstrap_auc_ci(y, p, n_boot=200)
    assert 0.5 < lo < hi <= 1.0
