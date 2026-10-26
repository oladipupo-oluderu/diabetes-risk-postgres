"""Run the named SQL queries in sql/analysis_queries.sql and print the results."""
import re

from config import QUERIES_SQL
from data_loader import run_query


def load_queries() -> dict:
    """Split the SQL file into {name: query} using the '-- name:' markers."""
    text_ = QUERIES_SQL.read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text_, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main() -> None:
    for name, sql in load_queries().items():
        print(f"\n=== {name} ===")
        print(run_query(sql).to_string(index=False))


if __name__ == "__main__":
    main()
