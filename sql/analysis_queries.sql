-- name: prevalence_by_bmi
-- Diabetes rate by WHO BMI category
SELECT bmi_category,
       COUNT(*)                                   AS n_patients,
       ROUND(100.0 * AVG(outcome), 1)             AS diabetes_rate_pct
FROM risk_groups
GROUP BY bmi_category
ORDER BY diabetes_rate_pct DESC

-- name: glucose_by_outcome
-- Median and mean glucose by outcome (percentile_cont is PostgreSQL-specific)
SELECT outcome,
       ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY glucose::double precision))::numeric, 1) AS median_glucose,
       ROUND(AVG(glucose), 1)                                                                      AS mean_glucose
FROM clean_diabetes
GROUP BY outcome
ORDER BY outcome

-- name: missing_value_audit
-- How many NULLs did cleaning create? (COUNT ... FILTER is PostgreSQL syntax)
SELECT COUNT(*) FILTER (WHERE glucose        IS NULL) AS missing_glucose,
       COUNT(*) FILTER (WHERE blood_pressure IS NULL) AS missing_blood_pressure,
       COUNT(*) FILTER (WHERE skin_thickness IS NULL) AS missing_skin_thickness,
       COUNT(*) FILTER (WHERE insulin        IS NULL) AS missing_insulin,
       COUNT(*) FILTER (WHERE bmi            IS NULL) AS missing_bmi
FROM clean_diabetes

-- name: prevalence_by_age_group
-- Diabetes rate by age band, using a CTE
WITH banded AS (
    SELECT outcome,
           CASE WHEN age < 30 THEN '21-29'
                WHEN age < 40 THEN '30-39'
                WHEN age < 50 THEN '40-49'
                WHEN age < 60 THEN '50-59'
                ELSE '60+' END AS age_group
    FROM clean_diabetes
)
SELECT age_group,
       COUNT(*)                        AS n_patients,
       ROUND(100.0 * AVG(outcome), 1)  AS diabetes_rate_pct
FROM banded
GROUP BY age_group
ORDER BY age_group
