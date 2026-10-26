"""Database helpers: engine, readiness check, schema, and logging tables."""
import json
import time
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from config import DATABASE_URL, SCHEMA_SQL


@lru_cache(maxsize=1)
def get_engine():
    """One shared engine (connection pool) for the whole process."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def wait_for_db(retries: int = 30, delay: float = 2.0) -> None:
    """Block until PostgreSQL accepts connections (useful right after `docker compose up`)."""
    engine = get_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            print(f"Waiting for PostgreSQL ({attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError(
        "Could not connect to PostgreSQL. Check DATABASE_URL and that the server is running."
    )


def server_version() -> str:
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT version()")).scalar_one()


def apply_schema() -> None:
    """Run sql/01_schema.sql. Idempotent: tables are kept, views are rebuilt."""
    with get_engine().begin() as conn:  # begin() = one transaction, auto-commit on success
        conn.exec_driver_sql(SCHEMA_SQL.read_text())


def log_model_runs(results: dict, selected: str, n_train: int, n_test: int) -> None:
    """Store each model's metrics in the model_runs table."""
    sql = text("""
        INSERT INTO model_runs
            (model_name, selected, cv_roc_auc_mean, cv_roc_auc_std,
             test_accuracy, test_recall, test_f1, test_roc_auc, n_train, n_test)
        VALUES
            (:name, :selected, :cv_mean, :cv_std,
             :acc, :rec, :f1, :auc, :n_train, :n_test)
    """)
    rows = [
        {
            "name": name,
            "selected": name == selected,
            "cv_mean": float(m["cv_roc_auc_mean"]),
            "cv_std": float(m["cv_roc_auc_std"]),
            "acc": float(m["test_accuracy"]),
            "rec": float(m["test_recall"]),
            "f1": float(m["test_f1"]),
            "auc": float(m["test_roc_auc"]),
            "n_train": n_train,
            "n_test": n_test,
        }
        for name, m in results.items()
    ]
    with get_engine().begin() as conn:
        conn.execute(sql, rows)


def log_tuned_runs(results: dict, selected: str, threshold: float,
                   n_train: int, n_test: int) -> None:
    """Store tuned-model metrics (stage='tuned'), incl. best params and AUC interval."""
    sql = text("""
        INSERT INTO model_runs
            (model_name, selected, stage, best_params, decision_threshold,
             cv_roc_auc_mean, cv_roc_auc_std, test_accuracy, test_recall,
             test_f1, test_roc_auc, auc_ci_low, auc_ci_high, n_train, n_test)
        VALUES
            (:name, :selected, 'tuned', CAST(:params AS JSONB), :thr,
             :cv_mean, :cv_std, :acc, :rec, :f1, :auc, :lo, :hi, :n_train, :n_test)
    """)
    rows = [
        {
            "name": name,
            "selected": name == selected,
            "params": json.dumps(m["best_params"]),
            "thr": float(threshold) if name == selected else None,
            "cv_mean": float(m["cv_roc_auc_mean"]),
            "cv_std": float(m["cv_roc_auc_std"]),
            "acc": float(m["test_accuracy"]),
            "rec": float(m["test_recall"]),
            "f1": float(m["test_f1"]),
            "auc": float(m["test_roc_auc"]),
            "lo": float(m["auc_ci_low"]),
            "hi": float(m["auc_ci_high"]),
            "n_train": n_train,
            "n_test": n_test,
        }
        for name, m in results.items()
    ]
    with get_engine().begin() as conn:
        conn.execute(sql, rows)


def log_prediction(features: dict, probability: float) -> None:
    """Store one prediction (inputs as JSONB) in prediction_log."""
    sql = text("""
        INSERT INTO prediction_log (input_features, risk_probability)
        VALUES (CAST(:features AS JSONB), :p)
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {"features": json.dumps(features), "p": float(probability)})
