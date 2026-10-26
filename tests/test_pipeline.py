import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import build_db  # noqa: E402
import db  # noqa: E402
import train  # noqa: E402
from data_loader import load_clean, run_query  # noqa: E402
from predict import predict_risk  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def database():
    try:
        db.wait_for_db(retries=2, delay=1)
    except RuntimeError:
        pytest.skip("PostgreSQL is not reachable (set DATABASE_URL)")
    build_db.build_database()


def test_really_running_on_postgresql():
    assert db.server_version().startswith("PostgreSQL")


def test_row_count():
    assert run_query("SELECT COUNT(*) AS n FROM raw_diabetes")["n"].iloc[0] == 768


def test_raw_layer_keeps_zeros_and_clean_layer_removes_them():
    raw_zeros = run_query("SELECT COUNT(*) AS n FROM raw_diabetes WHERE glucose = 0")["n"].iloc[0]
    assert raw_zeros > 0
    df = load_clean()
    for col in ["glucose", "blood_pressure", "bmi"]:
        assert (df[col] == 0).sum() == 0
    assert df["glucose"].isna().sum() == raw_zeros


def test_bmi_categories_are_valid():
    cats = set(run_query("SELECT DISTINCT bmi_category AS c FROM risk_groups")["c"])
    assert cats <= {"Underweight", "Normal", "Overweight", "Obese", "Unknown"}


def test_constraint_rejects_bad_outcome():
    with pytest.raises(Exception):
        with db.get_engine().begin() as conn:
            conn.execute(text(
                "INSERT INTO raw_diabetes (pregnancies, glucose, blood_pressure, "
                "skin_thickness, insulin, bmi, diabetes_pedigree_function, age, outcome) "
                "VALUES (1, 100, 70, 20, 80, 25.0, 0.3, 30, 5)"))


def test_train_logs_runs_and_predict_logs_prediction(tmp_path, monkeypatch):
    # Work on temporary files so running the tests never overwrites the real
    # trained model, metrics or figures produced by `python src/pipeline.py`.
    import predict
    tmp_model = tmp_path / "model.joblib"
    monkeypatch.setattr(train, "MODEL_PATH", tmp_model)
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(train, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_model)
    train.main()
    n_runs = run_query("SELECT COUNT(*) AS n FROM model_runs WHERE selected")["n"].iloc[0]
    assert n_runs >= 1
    p = predict_risk({
        "pregnancies": 2, "glucose": 150, "blood_pressure": 80,
        "skin_thickness": None, "insulin": None, "bmi": 33.5,
        "diabetes_pedigree_function": 0.5, "age": 45,
    })
    assert 0.0 <= p <= 1.0
    assert run_query("SELECT COUNT(*) AS n FROM prediction_log")["n"].iloc[0] >= 1
