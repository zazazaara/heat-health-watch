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
{{ROWS}}

## 7. Execution status and evidence
| Metric | Value |
|---|---|
| Total executed | {{TOTAL}} |
| Passed | {{PASSED}} |
| Failed | {{FAILED}} |
| Skipped | {{SKIPPED}} |
| Duration (s) | {{DURATION}} |

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
All {{PASSED}} of {{TOTAL}} executed automated tests passed in the final run after fixing the incidents in section 8.
This shows the implemented functions behave as specified; it does not prove real-world predictive validity.
Known limitations are listed in `docs/KNOWN_LIMITATIONS.md`.

## 12. References
* Supplied lab reports and project report (OOSE Experiments 1–7, SDD, project report) and `delhi_heatwave_airquality.csv`.
* scikit-learn `RandomForestClassifier` documentation; Streamlit `AppTest` documentation; pytest documentation.
* Magnus formula for dew point (used in the hourly simulation, Alduchov & Eskridge, 1996).
