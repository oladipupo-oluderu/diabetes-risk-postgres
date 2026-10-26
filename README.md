# Diabetes Risk Prediction Pipeline (PostgreSQL + Docker Compose) 
[![DOI](https://zenodo.org/badge/DOI/ DOI: 10.5281/zenodo.23674618.svg)](https://doi.org/ DOI: 10.5281/zenodo.23674618) 

End-to-end ML project on the Pima Indians Diabetes dataset (768 patients, 8 clinical
features). Raw data lives in **PostgreSQL**, cleaning is done with SQL views, models are
trained from data queried out of the database, and results and predictions are written
back to it. Everything runs with one command through Docker Compose.

## Data Architecture & Pipeline Design

```
 data/raw/diabetes.csv                  (source file, never edited)
          |
          |  src/build_db.py            (applies sql/01_schema.sql, then COPY)
          v
 +-----------------------------------------------------------+
 | PostgreSQL 16   (container "db", volume "pgdata")         |
 |                                                           |
 |  raw_diabetes    TABLE   typed columns + CHECK constraints|
 |        |                 untouched source data            |
 |  clean_diabetes  VIEW    impossible 0s -> NULL (NULLIF)   |
 |        |                                                  |
 |  risk_groups     VIEW    + WHO BMI category               |
 |                                                           |
 |  model_runs      TABLE   metrics of every training run    |
 |  prediction_log  TABLE   inputs (JSONB) + risk per call   |
 +-----------------------------------------------------------+
          |                          ^            ^
          | SQLAlchemy               |            |
          | pd.read_sql              | INSERT     | INSERT
          v                          |            |
   pandas DataFrame (src/data_loader.py)          |
          |                          |            |
          +--> src/eda.py      -> reports/figures/*.png
          +--> src/analysis.py -> named SQL queries (CTE, FILTER, percentile_cont)
          +--> src/train.py    -> impute -> scale -> model (5-fold CV)
                                   |--> models/diabetes_model.joblib
                                   |--> reports/metrics.json
                                   '--> model_runs rows
                                                 |
                                       src/predict.py ----------> prediction_log
```

### Design decisions
- **Raw vs clean layers.** Glucose, blood pressure, skin thickness, insulin and BMI contain
  0s that are physiologically impossible, i.e. missing values coded as 0. The raw table
  keeps them; a SQL view turns them into NULL. Cleaning is documented and reversible.
- **Constraints in the schema.** `CHECK` constraints and `NOT NULL` stop bad rows at the
  database door (for example `outcome` must be 0 or 1).
- **COPY for loading.** PostgreSQL's `COPY ... FROM STDIN` is much faster than row-by-row
  inserts and is the standard bulk-load path.
- **Config through `DATABASE_URL`.** The same code runs locally and in Docker unchanged.
- **Imputation inside the model pipeline**, fitted on training folds only, to avoid leakage.
- **Model selected by cross-validated ROC-AUC**, not the test set, so the test set stays an
  honest final check.
- **Recall is reported** because in screening, missing a diabetic patient costs more than
  a false alarm.

## Quick start with Docker Compose

```bash
cp .env.example .env          # then edit the password
docker compose up --build --abort-on-container-exit app
```

This starts PostgreSQL, waits until it is healthy, then runs the whole pipeline.
Figures and metrics appear in `reports/`, the model in `models/`.

```bash
docker compose run --rm app pytest -q          # run the tests
docker compose exec db psql -U diabetes_user -d diabetes_db
docker compose down -v                         # stop and delete the database volume
```

If port 5432 is already used on your machine, set `DB_PORT=5433` in `.env`.

## Run without Docker (local PostgreSQL)

```bash
psql -U postgres -f sql/00_local_setup.sql     # creates user + database once
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://diabetes_user:diabetes_pass@localhost:5432/diabetes_db
python src/pipeline.py
python src/predict.py
pytest -q
```

## Look inside the database

```sql
SELECT * FROM risk_groups LIMIT 5;
SELECT model_name, selected, cv_roc_auc_mean, test_roc_auc, test_recall
FROM model_runs ORDER BY trained_at DESC;
SELECT input_features->>'glucose' AS glucose, risk_probability FROM prediction_log;
```

## Results

Reported in `reports/metrics.json` and stored in `model_runs`. In the reference run, both
Logistic Regression and Random Forest reached a test ROC-AUC of about 0.81-0.82 with recall
of about 0.70-0.76; Logistic Regression was selected on cross-validated AUC. Exact numbers
can shift slightly with library versions. This is a research prototype, not a diagnostic tool.

Example SQL finding: the diabetes rate rises from about 7% (normal BMI) to about 46% (obese).

## Project structure

```
diabetes-risk-postgres/
├── data/raw/diabetes.csv
├── sql/            00_local_setup.sql, 01_schema.sql, analysis_queries.sql
├── src/            config, db, build_db, data_loader, analysis, eda, train, predict, pipeline
├── tests/          test_pipeline.py (runs against a live PostgreSQL)
├── reports/        figures/, metrics.json
├── models/         saved model (generated)
├── Dockerfile, docker-compose.yml, .env.example, requirements.txt, Makefile
```

## Data source

Pima Indians Diabetes Database (UCI / NIDDK). Cite the original source in any write-up.

## Author

Demola | GitHub: oladipupo-oluderu
