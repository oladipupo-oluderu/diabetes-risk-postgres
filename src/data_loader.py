"""Data access layer: the ONLY place the ML code reads from PostgreSQL."""
import pandas as pd
from sqlalchemy import text

from db import get_engine


def run_query(sql: str) -> pd.DataFrame:
    """Run any SELECT and return a DataFrame."""
    with get_engine().connect() as conn:
        df = pd.read_sql(text(sql), conn)
    # NUMERIC/SMALLINT columns containing NULLs can arrive as object dtype;
    # convert every column that looks numeric to a real number type.
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted
    return df


def load_clean() -> pd.DataFrame:
    """Cleaned view: impossible zeros are already NULL."""
    return run_query("SELECT * FROM clean_diabetes ORDER BY patient_id")


def load_risk_groups() -> pd.DataFrame:
    """Cleaned data plus the WHO BMI category column."""
    return run_query("SELECT * FROM risk_groups ORDER BY patient_id")
