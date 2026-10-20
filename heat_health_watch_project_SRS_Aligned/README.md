# Heat-Health Risk Monitoring System (Delhi)

OOSE mini project · Streamlit dashboard · scikit-learn Random Forest · historical risk monitoring · pytest.

## Run on Windows (PowerShell)
```powershell
cd heat_health_watch_project_SRS_Aligned
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Run the tests: `python -m pytest -q` (≈30 s). Retrain the saved model: `python -m src.risk_classifier`.
Rebuild the STD from a fresh run: `python -m pytest -q --junitxml=docs/test_results.xml; python scripts/build_std.py`.
No internet or API key is needed. The original CSV in `data/` is never modified.


## Alignment with the submitted OOSE report

The core workflow follows the report's historical-monitoring scope: load historical CSV data, display environmental measurements and risk information, classify risk with Random Forest, visualize trends, and show general health advisories. The existing report/UML diagrams are not modified. The app does not claim a live weather/AQI API, SQL database, login system, or administrator CRUD because these are not implemented in this runnable version.

The five-level score calculator, synthetic scenarios, hourly simulation and future-date estimate tab are retained as **optional extensions**. The Random Forest learns only the two labels present in the supplied dataset (Moderate and High); the five-level categories are produced by a separate score rule. Synthetic and hourly values are labelled as such. The future-date tab now generates a separate synthetic continuation after the original data endpoint, then estimates dates after that continuation. It is labelled experimental and must not be presented as an official weather forecast. See `docs/REQUIREMENTS_TRACEABILITY.md` for the scope mapping.

## What the app contains
| Tab | Purpose |
|---|---|
| Overview | KPIs, latest advisory, distributions, temperature and score trends for the selected date range |
| Risk Calculator | Enter Tmax, humidity, AQI → score, 5-level category, original two-class rule, advisory |
| Trends | Temperature, humidity, pollutants, score, seasonal comparison |
| ML Risk Classifier | Random Forest info, held-out metrics, confusion matrix, importances, prediction, actual-vs-predicted |
| Forecast (Experimental) | Optional future-date scenario estimates, clearly separated from historical observations; not an official weather forecast |
| Hourly (Simulated) | 24-hour profile built from daily values; peak-risk hour; time-wise advisory; CSV |
| Synthetic Scenarios | Reproducible demo data covering all five levels; CSV |
| Data Explorer | Filter/search/sort historical records, quality report, CSV |

## Key facts about the supplied data (verified, not assumed)
* 1,429 rows → 1,428 after removing 1 exact duplicate (2022-05-01); 2021-01-01 to **2024-11-30**; 2 calendar days missing (2022-09-15/16). **Daily only — no hourly data.**
* `Risk_Category` has **two classes** (Moderate 890, High 539 before dedupe). It is exactly `High` when `Heat_Health_Risk_Score > 55`.
* The stored score equals **45·NormTemp + 25·NormHumidity + 30·AQI/500** (NormTemp from *Max* temperature: (Tmax−9.1)/38.8; NormHumidity: (RH−16.1)/83.2). Our implementation reproduces all stored scores exactly (max error 0).
* Real scores span 34–69, so on the 5-level scale real days are only Low/Moderate/High. Very Low and Severe appear only in synthetic scenarios.

## Score, categories, classifier, forecast — in one paragraph each
**Score.** Weighted blend of normalised temperature (45 pts), humidity (25 pts) and AQI (30 pts); 0–100; a project heuristic, not a probability of illness. Thresholds 20/40/60/80 are project-defined and editable in `src/config.py` (`RISK_LEVELS`).
**Classifier.** `RandomForestClassifier(300 trees)` inside a pipeline with median imputation; 11 features from the original report; chronological 80/20 split (train 2021-01-01→2024-02-18, test 2024-02-19→2024-11-30). Held-out accuracy **89.86 %** vs majority-class baseline 59.09 %; High precision 0.95 / recall 0.79; Moderate precision 0.87 / recall 0.97. Because the label is a deterministic threshold on a score of Tmax, humidity and AQI, the forest is re-learning that rule from correlated inputs — a valid ML exercise, not clinical validation.
**Forecast.** For each of Tmax, Tmean, humidity and AQI: seasonal harmonics (3 sine/cosine pairs, no trend) + AR(1) decaying anomaly. Hold-out MAE for Tmax: 1.94 °C at 1–7 days vs 2.19 °C persistence; whole-test-block 2.43 °C vs 3.40 °C (seasonal naive). Humidity and AQI are forecast by the same method or supplied by the user, and every row states which. After ~30 days the output is just the seasonal average and is labelled an *illustrative scenario*. **Classification and forecasting are different things:** the classifier labels a day whose inputs are known; the forecaster estimates inputs for future days.
**Synthetic / simulated.** The five-level demonstration scenarios are rejection-sampled inside plausible envelopes. Hourly profiles use a diurnal cycle, constant dew point and assumed AQI profile. The Forecast tab can append a separate, clearly labelled synthetic daily continuation in memory to extend the timeline; this does not modify the original CSV and is not used for classifier training or evaluation. Forecasts based on it are exploratory scenarios, not observed weather.

## Viva demonstration sequence
1. Sidebar: dataset range and endpoint banner. 2. Overview: KPIs, two distributions (note only two real labels). 3. Calculator: try 42 °C / 60 % / 210 AQI. 4. ML tab: warning, metrics vs baseline, confusion matrix, actual-vs-predicted. 5. Forecast: default 14 days, then a date far out to show the illustrative warning; try an invalid range. 6. Hourly: pick a hot day, discuss the SIMULATED banner and peak hour. 7. Synthetic: generate, show all five levels, download. 8. Data Explorer + quality report. 9. Run `pytest`.

## Folder structure
`app.py` · `src/` (config, data_loader, preprocessing, risk_scoring, risk_classifier, forecasting, synthetic_data, hourly_analysis, advisories, visualization) · `tests/` · `models/` · `data/` · `docs/` · `scripts/`

See `docs/ARCHITECTURE.md`, `docs/REQUIREMENTS_TRACEABILITY.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/STD.md`.

## Synthetic continuation and future estimates

The Forecast tab can append a generated synthetic daily continuation after the original CSV, then estimate dates after that continuation. The continuation is calibrated to seasonal patterns in the original observations and includes correlated day-to-day variation, so it is not a forced straight-line extrapolation. Synthetic rows are not observed weather; outputs are exploratory scenarios only. The original CSV, classifier training, and forecasting evaluation remain based on original observations.
