# Model card: diabetes risk screening prototype

**Model:** logistic_regression  |  **Trained:** 2026-10-25 18:22 UTC
**Data:** Pima Indians Diabetes dataset (768 women, 21+, Pima heritage, Arizona USA), read from PostgreSQL view `clean_diabetes`.

## Intended use
Research / portfolio demonstration of an SQL-backed ML pipeline. **Not a diagnostic tool.**

## Performance (held-out test set, n = 154)
| Metric | Value |
|---|---|
| ROC-AUC | 0.811 (95% bootstrap CI 0.741 - 0.872) |
| Cross-validated ROC-AUC (train) | 0.845 +/- 0.015 |
| Brier score | 0.181 |

| At threshold | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|
| 0.50 (default) | 0.704 | 0.603 | 0.75 | 0.65 |
| 0.411 (screening, target recall 0.8) | 0.815 | 0.571 | 0.67 | 0.672 |

The screening threshold was chosen from out-of-fold training predictions, not from the test set.

## Most influential features (permutation importance)
glucose (0.1547), bmi (0.028), pregnancies (0.0262)

## Limitations
- Small dataset: the test set holds 154 patients, so metrics carry wide uncertainty (see the CI).
- Single population (Pima women in Arizona): no evidence of generalisation to Nigerian or other cohorts.
- Zeros in glucose, blood pressure, skin thickness, insulin and BMI were treated as missing and imputed with the training median.
- Outcome definition and collection year predate current diagnostic criteria.
