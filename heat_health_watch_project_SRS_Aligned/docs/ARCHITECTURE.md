# Architecture and implementation notes

## Core architecture

The application follows a simple layered structure: **Presentation** (`app.py`, `visualization.py`) → **application logic** (`risk_scoring`, `risk_classifier`, `advisories`, data validation) → **data** (`data/delhi_heatwave_airquality.csv`). Optional demonstration modules provide experimental future-date estimates, synthetic scenarios and hourly simulations. Constants (paths, formula weights, thresholds, features and limits) live in `src/config.py`.

## Data pipeline

CSV (read-only) → parse dates → drop exact duplicates → sort and validate → quality report → add the project-defined five-level score category. Median imputation happens inside the classifier pipeline, so it is fitted on training rows only.

## Risk score

`score = 45·clip((Tmax−9.1)/38.8) + 25·clip((RH−16.1)/83.2) + 30·clip(AQI/500)`, rounded to 2 decimal places on a 0–100 scale. The formula reproduces the supplied dataset's `Heat_Health_Risk_Score` column. It is a project heuristic, not a clinical probability or official health standard.

The four categories named in the submitted report (Low, Moderate, High and Severe) remain represented. The app also includes **Very Low** as an additional project-defined level so the requested five-level calculator and synthetic demonstration can be retained. The supplied real data does not cover every level.

## Random Forest classifier

A pipeline with median imputation and `RandomForestClassifier` is trained/evaluated using a chronological split. The target column in the supplied data contains only `Moderate` and `High`; therefore the classifier cannot honestly be described as a five-class model. The five-level score calculator is a separate rule-based component.

## Optional experimental extensions

- **Future-date estimates:** the Forecast tab preserves original observations, generates a clearly labelled synthetic continuation using seasonal harmonics plus correlated AR(1)-style anomalies, then fits a seasonal + AR(1) estimator on original + synthetic rows to estimate dates after the continuation. The timeline visually separates observed, synthetic and forecast values. This is an exploratory scenario, not an official weather forecast; the synthetic extension is never written into the original CSV.
- **Hourly simulation:** constructs an assumed diurnal profile from daily minimum/maximum temperature, humidity and AQI. It is not based on measured hourly observations.
- **Synthetic scenarios:** generated for demonstration/testing across the five project-defined levels. They are clearly labelled and are not written into the historical dataset or used to train/evaluate the Random Forest. The separate forecast continuation is used only as an explicitly synthetic input to the optional forecast scenario.

## Data/API/database scope

The runnable implementation reads the supplied local CSV and does not modify it. It does not implement a live weather/AQI API, SQL database, authentication or administrator CRUD. The original report's UML diagrams are left unchanged as requested; elements representing these planned components should not be claimed as implemented functionality during the demo.
