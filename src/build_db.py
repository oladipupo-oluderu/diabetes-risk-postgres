"""Step 1 of the pipeline: CSV -> PostgreSQL.

Run:  python src/build_db.py
"""
from config import RAW_CSV, FEATURES, TARGET
from db import apply_schema, get_engine, server_version, wait_for_db

COLUMNS = FEATURES + [TARGET]  # same order as the CSV columns


def load_raw() -> int:
    """Bulk-load the CSV with COPY, PostgreSQL's fastest ingestion path."""
    copy_sql = (
        f"COPY raw_diabetes ({', '.join(COLUMNS)}) "
        "FROM STDIN WITH (FORMAT csv, HEADER true)"
    )
    raw = get_engine().raw_connection()  # underlying psycopg2 connection
    try:
        with raw.cursor() as cur, open(RAW_CSV, newline="") as f:
            # Reload from scratch so the script can be re-run safely
            cur.execute("TRUNCATE raw_diabetes RESTART IDENTITY")
            cur.copy_expert(copy_sql, f)
            cur.execute("SELECT COUNT(*) FROM raw_diabetes")
            n = cur.fetchone()[0]
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return n


def build_database() -> None:
    wait_for_db()
    print("Connected to:", server_version().split(",")[0])
    apply_schema()
    n = load_raw()
    print(f"Loaded {n} rows into raw_diabetes")


if __name__ == "__main__":
    build_database()
