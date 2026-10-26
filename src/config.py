"""Central place for paths and settings so no script hardcodes them."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "diabetes.csv"
SCHEMA_SQL = ROOT / "sql" / "01_schema.sql"
QUERIES_SQL = ROOT / "sql" / "analysis_queries.sql"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"

# Connection string is read from the environment (12-factor style), so the
# same code runs on your laptop and inside Docker Compose without edits.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://diabetes_user:diabetes_pass@localhost:5432/diabetes_db",
)

TARGET = "outcome"
FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Modelling stage 2 (tuning, threshold, interpretation) ---
TUNED_METRICS_PATH = ROOT / "reports" / "metrics_tuned.json"
MODEL_META_PATH = ROOT / "models" / "model_meta.json"
MODEL_CARD_PATH = ROOT / "reports" / "MODEL_CARD.md"
# Screening goal: catch at least this share of true diabetics, then keep
# precision as high as possible. Change it to trade recall against false alarms.
TARGET_RECALL = 0.80
N_SEARCH_ITER = 25      # random-search candidates per model
N_BOOTSTRAP = 1000      # resamples for the test-AUC confidence interval
