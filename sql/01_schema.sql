-- PostgreSQL schema for the diabetes risk pipeline.
-- Safe to re-run: tables use IF NOT EXISTS, views are dropped and recreated.

-- ---------------------------------------------------------------
-- Layer 1: raw data, stored exactly as received (never modified)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_diabetes (
    patient_id                  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pregnancies                 SMALLINT         NOT NULL CHECK (pregnancies >= 0),
    glucose                     SMALLINT         NOT NULL CHECK (glucose >= 0),
    blood_pressure              SMALLINT         NOT NULL CHECK (blood_pressure >= 0),
    skin_thickness              SMALLINT         NOT NULL CHECK (skin_thickness >= 0),
    insulin                     SMALLINT         NOT NULL CHECK (insulin >= 0),
    bmi                         DOUBLE PRECISION NOT NULL CHECK (bmi >= 0),
    diabetes_pedigree_function  DOUBLE PRECISION NOT NULL CHECK (diabetes_pedigree_function >= 0),
    age                         SMALLINT         NOT NULL CHECK (age BETWEEN 1 AND 120),
    outcome                     SMALLINT         NOT NULL CHECK (outcome IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_raw_diabetes_outcome ON raw_diabetes (outcome);

-- ---------------------------------------------------------------
-- Layer 2 and 3: views (dropped in reverse dependency order)
-- ---------------------------------------------------------------
DROP VIEW IF EXISTS risk_groups;
DROP VIEW IF EXISTS clean_diabetes;

-- A 0 in these columns is physiologically impossible, so it is a missing
-- value coded as 0. NULLIF(x, 0) returns NULL when x = 0.
CREATE VIEW clean_diabetes AS
SELECT
    patient_id,
    pregnancies,
    NULLIF(glucose, 0)         AS glucose,
    NULLIF(blood_pressure, 0)  AS blood_pressure,
    NULLIF(skin_thickness, 0)  AS skin_thickness,
    NULLIF(insulin, 0)         AS insulin,
    NULLIF(bmi, 0)             AS bmi,
    diabetes_pedigree_function,
    age,
    outcome
FROM raw_diabetes;

-- WHO BMI categories for clinical risk grouping
CREATE VIEW risk_groups AS
SELECT
    c.*,
    CASE
        WHEN bmi IS NULL THEN 'Unknown'
        WHEN bmi < 18.5  THEN 'Underweight'
        WHEN bmi < 25    THEN 'Normal'
        WHEN bmi < 30    THEN 'Overweight'
        ELSE 'Obese'
    END AS bmi_category
FROM clean_diabetes AS c;

-- ---------------------------------------------------------------
-- ML bookkeeping tables (results and predictions live in the database)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    run_id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trained_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    model_name        TEXT             NOT NULL,
    selected          BOOLEAN          NOT NULL,
    cv_roc_auc_mean   DOUBLE PRECISION,
    cv_roc_auc_std    DOUBLE PRECISION,
    test_accuracy     DOUBLE PRECISION,
    test_recall       DOUBLE PRECISION,
    test_f1           DOUBLE PRECISION,
    test_roc_auc      DOUBLE PRECISION,
    n_train           INTEGER,
    n_test            INTEGER
);

-- Added in modelling stage 2. ADD COLUMN IF NOT EXISTS keeps old databases working.
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS stage              TEXT NOT NULL DEFAULT 'baseline';
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS best_params        JSONB;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS decision_threshold DOUBLE PRECISION;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS auc_ci_low         DOUBLE PRECISION;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS auc_ci_high        DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS prediction_log (
    prediction_id     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    predicted_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    input_features    JSONB            NOT NULL,
    risk_probability  DOUBLE PRECISION NOT NULL CHECK (risk_probability BETWEEN 0 AND 1)
);
