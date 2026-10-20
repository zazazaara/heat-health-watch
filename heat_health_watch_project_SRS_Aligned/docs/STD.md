# Software Test Document (STD)
**Project:** Heat-Health Risk Monitoring, Prediction and Early Warning System (Delhi)
**Course:** Object-Oriented Software Engineering (OOSE) · TY B.Tech AI & DS

> **Template note.** The faculty STD template was not included in the supplied files, so this document follows the
> section list given in the project brief. Re-map the headings if the faculty format differs.
> Only test cases that were actually executed are marked Passed (see section 9).

## 1. Overview and objectives
Verify that the implemented system meets the requirements in `docs/REQUIREMENTS_TRACEABILITY.md`: correct data
handling, a reproducible risk score, an honestly evaluated Random Forest classifier, validated future-date
forecasting, clearly labelled synthetic/simulated outputs, and a dashboard that starts and responds.

## 2. Scope and approach
Automated unit and integration tests with **pytest**, plus headless dashboard tests with Streamlit's `AppTest`
and a real server launch (health endpoint). Deterministic seeds are used throughout (`RANDOM_SEED = 42`).

## 3. Features tested
Data loading/validation/export · duplicate and missing-date handling · score formula and bounds · five-level
categories and boundaries · advisories · Random Forest training, held-out evaluation and prediction · forecast
date validation, output schema, status labels · synthetic data reproducibility and labelling · hourly simulation
validation · dashboard start-up and every action button.

## 4. Features not tested
* Visual appearance in a real browser (layout, colours, chart overlap) — not verified by automation; check manually.
* Live external weather/AQI APIs — not implemented (out of scope, optional future work).
* Accuracy of forecasts against future real weather — impossible to test; only chronological hold-out evaluation exists.
* Authentication, SMS/e-mail, hardware — out of scope.
* Usability testing with real users; performance/load testing.

## 5. Environment and tools
Python 3.12.3 · pandas 3.0.2 · numpy 2.4.4 · scikit-learn 1.8.0 · streamlit 1.65.0 · plotly 7.1.0 · pytest 9.1.1 ·
Linux sandbox. (Target machine: Windows 10/11 + PowerShell — not executed there by the author of this document.)

## 6. Test cases (IDs, steps, expected and actual results)
Each case is one pytest function: **steps** = arrange inputs/fixtures and call the function named in the table;
**expected** = the assertion in that function (see `tests/`); **actual** = outcome of the recorded run in
`docs/test_results.xml`.

| ID | Feature | Requirement | Test function | What is checked | Actual result |
|---|---|---|---|---|---|
| TC-001 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_app_starts_without_exceptions` | app starts without exceptions | Passed |
| TC-002 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_endpoint_banner_shows_last_date` | endpoint banner shows last date | Passed |
| TC-003 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_calculator_button_returns_result` | calculator button returns result | Passed |
| TC-004 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_rf_prediction_button` | rf prediction button | Passed |
| TC-005 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_forecast_button_returns_rows` | forecast button returns rows | Passed |
| TC-006 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_forecast_invalid_range_shows_error` | forecast invalid range shows error | Passed |
| TC-007 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_hourly_button` | hourly button | Passed |
| TC-008 | Dashboard (headless AppTest) | FR2/FR5/FR7 | `test_app.py::test_synthetic_button` | synthetic button | Passed |
| TC-009 | Random Forest classification | FR4 | `test_classifier.py::test_uses_genuine_random_forest` | uses genuine random forest | Passed |
| TC-010 | Random Forest classification | FR4 | `test_classifier.py::test_learns_only_the_original_two_classes` | learns only the original two classes | Passed |
| TC-011 | Random Forest classification | FR4 | `test_classifier.py::test_chronological_split_has_no_overlap` | chronological split has no overlap | Passed |
| TC-012 | Random Forest classification | FR4 | `test_classifier.py::test_split_rejects_bad_fraction` | split rejects bad fraction | Passed |
| TC-013 | Random Forest classification | FR4 | `test_classifier.py::test_metrics_are_from_held_out_block` | metrics are from held out block | Passed |
| TC-014 | Random Forest classification | FR4 | `test_classifier.py::test_metrics_internally_consistent` | metrics internally consistent | Passed |
| TC-015 | Random Forest classification | FR4 | `test_classifier.py::test_feature_importance_covers_all_features` | feature importance covers all features | Passed |
| TC-016 | Random Forest classification | FR4 | `test_classifier.py::test_training_is_reproducible` | training is reproducible | Passed |
| TC-017 | Random Forest classification | FR4 | `test_classifier.py::test_example_prediction` | example prediction | Passed |
| TC-018 | Random Forest classification | FR4 | `test_classifier.py::test_prediction_rejects_missing_or_bad_input` | prediction rejects missing or bad input | Passed |
| TC-019 | Random Forest classification | FR4 | `test_classifier.py::test_imputer_is_fitted_on_training_only` | imputer is fitted on training only | Passed |
| TC-020 | Random Forest classification | FR4 | `test_classifier.py::test_nan_inputs_handled_by_pipeline_imputer` | nan inputs handled by pipeline imputer | Passed |
| TC-021 | Random Forest classification | FR4 | `test_classifier.py::test_save_and_load_roundtrip` | save and load roundtrip | Passed |
| TC-022 | Random Forest classification | FR4 | `test_classifier.py::test_corrupt_model_file_triggers_retrain` | corrupt model file triggers retrain | Passed |
| TC-023 | Random Forest classification | FR4 | `test_classifier.py::test_saved_project_model_matches_current_data` | saved project model matches current data | Passed |
| TC-024 | Random Forest classification | FR4 | `test_classifier.py::test_model_saved_under_other_sklearn_version_is_retrained` | model saved under other sklearn version is retrained | Passed |
| TC-025 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_dataset_loads_with_required_columns` | dataset loads with required columns | Passed |
| TC-026 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_date_range_and_endpoint` | date range and endpoint | Passed |
| TC-027 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_duplicate_row_removed_and_dates_unique` | duplicate row removed and dates unique | Passed |
| TC-028 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_missing_dates_reported` | missing dates reported | Passed |
| TC-029 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_no_missing_values_in_source` | no missing values in source | Passed |
| TC-030 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_target_has_only_two_original_classes` | target has only two original classes | Passed |
| TC-031 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_original_file_not_modified` | original file not modified | Passed |
| TC-032 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_missing_file_gives_clear_error` | missing file gives clear error | Passed |
| TC-033 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_missing_column_gives_clear_error` | missing column gives clear error | Passed |
| TC-034 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_unparseable_dates_rejected` | unparseable dates rejected | Passed |
| TC-035 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_duplicate_dates_with_different_values_collapse` | duplicate dates with different values collapse | Passed |
| TC-036 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_five_level_column_added` | five level column added | Passed |
| TC-037 | Data loading, validation, export | FR1/FR8/FR9 | `test_data.py::test_csv_export_roundtrip` | csv export roundtrip | Passed |
| TC-038 | Future forecasting | NEW-1 | `test_forecasting.py::test_rejects_end_before_start` | rejects end before start | Passed |
| TC-039 | Future forecasting | NEW-1 | `test_forecasting.py::test_rejects_start_inside_history` | rejects start inside history | Passed |
| TC-040 | Future forecasting | NEW-1 | `test_forecasting.py::test_rejects_too_long_range` | rejects too long range | Passed |
| TC-041 | Future forecasting | NEW-1 | `test_forecasting.py::test_rejects_start_too_far_out` | rejects start too far out | Passed |
| TC-042 | Future forecasting | NEW-1 | `test_forecasting.py::test_rejects_invalid_date` | rejects invalid date | Passed |
| TC-043 | Future forecasting | NEW-1 | `test_forecasting.py::test_insufficient_history_message` | insufficient history message | Passed |
| TC-044 | Future forecasting | NEW-1 | `test_forecasting.py::test_output_length_and_schema` | output length and schema | Passed |
| TC-045 | Future forecasting | NEW-1 | `test_forecasting.py::test_values_within_valid_ranges` | values within valid ranges | Passed |
| TC-046 | Future forecasting | NEW-1 | `test_forecasting.py::test_status_distinguishes_forecast_assumed_and_user_supplied` | status distinguishes forecast assumed and user supplied | Passed |
| TC-047 | Future forecasting | NEW-1 | `test_forecasting.py::test_reliability_labels` | reliability labels | Passed |
| TC-048 | Future forecasting | NEW-1 | `test_forecasting.py::test_override_validation` | override validation | Passed |
| TC-049 | Future forecasting | NEW-1 | `test_forecasting.py::test_rf_class_marked_unavailable` | rf class marked unavailable | Passed |
| TC-050 | Future forecasting | NEW-1 | `test_forecasting.py::test_far_future_converges_to_seasonal_average` | far future converges to seasonal average | Passed |
| TC-051 | Future forecasting | NEW-1 | `test_forecasting.py::test_summer_forecast_warmer_than_winter` | summer forecast warmer than winter | Passed |
| TC-052 | Future forecasting | NEW-1 | `test_forecasting.py::test_evaluation_table_and_baselines` | evaluation table and baselines | Passed |
| TC-053 | Future forecasting | NEW-1 | `test_forecasting.py::test_forecast_is_deterministic` | forecast is deterministic | Passed |
| TC-054 | Hourly simulation | NEW-2 | `test_hourly.py::test_24_rows_and_schema` | 24 rows and schema | Passed |
| TC-055 | Hourly simulation | NEW-2 | `test_hourly.py::test_marked_as_simulated` | marked as simulated | Passed |
| TC-056 | Hourly simulation | NEW-2 | `test_hourly.py::test_temperature_extremes_match_inputs` | temperature extremes match inputs | Passed |
| TC-057 | Hourly simulation | NEW-2 | `test_hourly.py::test_values_within_physical_ranges` | values within physical ranges | Passed |
| TC-058 | Hourly simulation | NEW-2 | `test_hourly.py::test_humidity_lowest_when_hottest` | humidity lowest when hottest | Passed |
| TC-059 | Hourly simulation | NEW-2 | `test_hourly.py::test_deterministic_for_seed` | deterministic for seed | Passed |
| TC-060 | Hourly simulation | NEW-2 | `test_hourly.py::test_peak_hour_is_argmax_not_hardcoded` | peak hour is argmax not hardcoded | Passed |
| TC-061 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_inputs_rejected[args0]` | invalid inputs rejected[args0] | Passed |
| TC-062 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_inputs_rejected[args1]` | invalid inputs rejected[args1] | Passed |
| TC-063 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_inputs_rejected[args2]` | invalid inputs rejected[args2] | Passed |
| TC-064 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_inputs_rejected[args3]` | invalid inputs rejected[args3] | Passed |
| TC-065 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_inputs_rejected[args4]` | invalid inputs rejected[args4] | Passed |
| TC-066 | Hourly simulation | NEW-2 | `test_hourly.py::test_invalid_hours_rejected` | invalid hours rejected | Passed |
| TC-067 | Hourly simulation | NEW-2 | `test_hourly.py::test_risk_windows_contiguous_and_threshold` | risk windows contiguous and threshold | Passed |
| TC-068 | Hourly simulation | NEW-2 | `test_hourly.py::test_no_windows_on_mild_day` | no windows on mild day | Passed |
| TC-069 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_formula_reproduces_every_stored_score` | formula reproduces every stored score | Passed |
| TC-070 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_original_label_rule_matches_every_row` | original label rule matches every row | Passed |
| TC-071 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_score_bounds_at_extremes[9.1-16.1-0]` | score bounds at extremes[9.1-16.1-0] | Passed |
| TC-072 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_score_bounds_at_extremes[47.9-99.3-500]` | score bounds at extremes[47.9-99.3-500] | Passed |
| TC-073 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_score_bounds_at_extremes[-5-0-0]` | score bounds at extremes[-5-0-0] | Passed |
| TC-074 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_score_bounds_at_extremes[55-100-500]` | score bounds at extremes[55-100-500] | Passed |
| TC-075 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_max_and_min_score_values` | max and min score values | Passed |
| TC-076 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[0-Very Low]` | all five categories and boundaries[0-Very Low] | Passed |
| TC-077 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[19.99-Very Low]` | all five categories and boundaries[19.99-Very Low] | Passed |
| TC-078 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[20-Low]` | all five categories and boundaries[20-Low] | Passed |
| TC-079 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[39.99-Low]` | all five categories and boundaries[39.99-Low] | Passed |
| TC-080 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[40-Moderate]` | all five categories and boundaries[40-Moderate] | Passed |
| TC-081 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[59.99-Moderate]` | all five categories and boundaries[59.99-Moderate] | Passed |
| TC-082 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[60-High]` | all five categories and boundaries[60-High] | Passed |
| TC-083 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[79.99-High]` | all five categories and boundaries[79.99-High] | Passed |
| TC-084 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[80-Severe]` | all five categories and boundaries[80-Severe] | Passed |
| TC-085 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_all_five_categories_and_boundaries[100-Severe]` | all five categories and boundaries[100-Severe] | Passed |
| TC-086 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_vectorised_matches_scalar` | vectorised matches scalar | Passed |
| TC-087 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_thresholds_are_configurable` | thresholds are configurable | Passed |
| TC-088 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[None]` | invalid temperature rejected[None] | Passed |
| TC-089 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[nan]` | invalid temperature rejected[nan] | Passed |
| TC-090 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[inf]` | invalid temperature rejected[inf] | Passed |
| TC-091 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[hot]` | invalid temperature rejected[hot] | Passed |
| TC-092 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[-50]` | invalid temperature rejected[-50] | Passed |
| TC-093 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_temperature_rejected[80]` | invalid temperature rejected[80] | Passed |
| TC-094 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[-1-100]` | invalid humidity or aqi rejected[-1-100] | Passed |
| TC-095 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[101-100]` | invalid humidity or aqi rejected[101-100] | Passed |
| TC-096 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[50--1]` | invalid humidity or aqi rejected[50--1] | Passed |
| TC-097 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[50-501]` | invalid humidity or aqi rejected[50-501] | Passed |
| TC-098 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[None-100]` | invalid humidity or aqi rejected[None-100] | Passed |
| TC-099 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_invalid_humidity_or_aqi_rejected[50-None]` | invalid humidity or aqi rejected[50-None] | Passed |
| TC-100 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_components_sum_to_score` | components sum to score | Passed |
| TC-101 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_score_is_deterministic` | score is deterministic | Passed |
| TC-102 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_categorize_rejects_bad_scores` | categorize rejects bad scores | Passed |
| TC-103 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_for_every_category[Very Low]` | advisory for every category[Very Low] | Passed |
| TC-104 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_for_every_category[Low]` | advisory for every category[Low] | Passed |
| TC-105 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_for_every_category[Moderate]` | advisory for every category[Moderate] | Passed |
| TC-106 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_for_every_category[High]` | advisory for every category[High] | Passed |
| TC-107 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_for_every_category[Severe]` | advisory for every category[Severe] | Passed |
| TC-108 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisory_unknown_category` | advisory unknown category | Passed |
| TC-109 | Risk score, 5 categories, advisories | FR3/FR6 | `test_risk_scoring.py::test_advisories_avoid_diagnosis_claims` | advisories avoid diagnosis claims | Passed |
| TC-110 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_reproducible_with_same_seed` | reproducible with same seed | Passed |
| TC-111 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_different_seed_gives_different_data` | different seed gives different data | Passed |
| TC-112 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_all_five_levels_present_with_requested_count` | all five levels present with requested count | Passed |
| TC-113 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_rows_labelled_synthetic` | rows labelled synthetic | Passed |
| TC-114 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_validation_passes_and_values_plausible` | validation passes and values plausible | Passed |
| TC-115 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_validation_detects_problems` | validation detects problems | Passed |
| TC-116 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_invalid_counts_rejected[0]` | invalid counts rejected[0] | Passed |
| TC-117 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_invalid_counts_rejected[-1]` | invalid counts rejected[-1] | Passed |
| TC-118 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_invalid_counts_rejected[201]` | invalid counts rejected[201] | Passed |
| TC-119 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_invalid_counts_rejected[2.5]` | invalid counts rejected[2.5] | Passed |
| TC-120 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_invalid_counts_rejected[5]` | invalid counts rejected[5] | Passed |
| TC-121 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_scores_use_the_same_formula` | scores use the same formula | Passed |
| TC-122 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_generation_never_touches_historical_data` | generation never touches historical data | Passed |
| TC-123 | Synthetic scenarios | NEW-3 | `test_synthetic_data.py::test_csv_export_keeps_label` | csv export keeps label | Passed |

## 7. Execution status and evidence
| Metric | Value |
|---|---|
| Total executed | 123 |
| Passed | 123 |
| Failed | 0 |
| Skipped | 0 |
| Duration (s) | 27.7 |

Evidence: `docs/test_results.xml` (JUnit XML from the final run). Additional manual/automated checks: launching
`python -m streamlit run app.py` and requesting `/_stcore/health` returned HTTP 200 `ok`.

## 8. Incident / defect reports
| ID | Found by | Description | Type | Resolution |
|---|---|---|---|---|
| INC-01 | `test_missing_dates_reported` (1st run) | Test expected the wrong missing dates (2022-09-18/19). The CSV skips 2022-09-15 and 2022-09-16. | Test-script defect (code was right) | Expected value corrected after checking the CSV. |
| INC-02 | `test_insufficient_history_message` | With too little history the forecaster complained about the date range first, hiding the real cause. | Product defect | `check_history()` now runs before date validation; test passes. |
| INC-03 | Code review of tests | A hourly test ended in `... or True`, so it could never fail. | Test-quality defect | Replaced by a real assertion (peak hour differs from hottest hour on a flat-temperature polluted day). |
| INC-04 | Design review | A model saved by another scikit-learn version could be loaded unsafely. | Robustness risk | Metrics store the scikit-learn version; mismatch triggers retraining (test added). |

## 9. Verification versus validation
* **Verification** ("built the product right"): unit tests of formula, thresholds, validation, split, schema;
  exact reproduction of all stored scores and labels by the documented formula.
* **Validation** ("built the right product"): requirements traceability, the honest two-class RF limitation,
  labelled synthetic/simulated data, held-out metrics against a majority baseline. Validation against real
  stakeholders (citizens, health authorities) and clinical outcomes was **not** performed.

## 10. Object-oriented testing strategies
* **Class/module-level (unit):** each module is tested through its public interface (scoring, advisories, loader, classifier, forecaster, synthetic, hourly).
* **Integration/cluster testing:** loader → scoring → classifier → forecast → dashboard exercised together by `test_app.py`.
* **State-based testing:** session-state results after each button action; error paths leave no stale result.
* **Boundary-value and equivalence-class testing:** category boundaries (19.99/20/…/80), input range limits.
* **Regression:** the full suite is re-run after every fix (final run recorded above).

## 11. Outcome and conclusion
All 123 of 123 executed automated tests passed in the final run after fixing the incidents in section 8.
This shows the implemented functions behave as specified; it does not prove real-world predictive validity.
Known limitations are listed in `docs/KNOWN_LIMITATIONS.md`.

## 12. References
* Supplied lab reports and project report (OOSE Experiments 1–7, SDD, project report) and `delhi_heatwave_airquality.csv`.
* scikit-learn `RandomForestClassifier` documentation; Streamlit `AppTest` documentation; pytest documentation.
* Magnus formula for dew point (used in the hourly simulation, Alduchov & Eskridge, 1996).
