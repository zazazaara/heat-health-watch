# Known limitations and future scope

1. **Two real classes only.** The RF cannot predict Very Low/Low/Severe; the 5-level scale is a rule on the score. Real data never reaches Very Low or Severe (scores 34–69).
2. **Label is a rule.** `Risk_Category` = score > 55, and the score is a function of Tmax, RH and AQI. The RF's 89.86 % accuracy reflects re-learning that rule from correlated inputs (it does not receive Tmax directly). It is not clinical validation.
3. **Heat_Index column is not a physical heat index** (range 79–339, correlation −0.95 with temperature). It is used as supplied because the original report used it; its definition is undocumented.
4. **Data ends 2024-11-30.** Any date after that is an extrapolation; beyond ~30 days it is only the seasonal average. Only ~3.9 years of history inform the seasonal pattern; no long-term trend/climate change is modelled.
5. **Forecast evaluation is single-split.** Short-range gains over persistence are modest (e.g. Tmax MAE 1.94 vs 2.19 °C).
6. **Hourly output is a simulation** from daily values with assumed profiles; the peak hour reflects those assumptions.
7. **Thresholds are project-defined**, not medical or government standards. Advisories are general guidance only.
8. **Synthetic data** is plausible-by-construction, not observed. The Random Forest is not trained on synthetic data. The optional forecast model does use a clearly labelled synthetic continuation to demonstrate an extended timeline, so forecasts from it are scenario outputs rather than verified predictions.
9. **Not verified:** appearance in a real browser; execution on Windows; lowest package versions in `requirements.txt` (tested with the versions in the STD).
10. **Future scope:** optional live weather/AQI API, hourly observations, probabilistic/ML forecasters (e.g. gradient boosting with lags), rolling-origin cross-validation, SQLite storage.

11. **OOSE report alignment:** the submitted report and UML diagrams are left unchanged. The core app implements historical CSV monitoring, rule-based risk scoring, the two-class Random Forest, trends and advisories. Forecast estimates, hourly simulation, synthetic scenarios and the fifth Very Low score category are optional extensions; they are labelled as experimental/synthetic where applicable. The local-CSV implementation does not provide live API access, SQL persistence, authentication or administrator CRUD.


## Synthetic continuation used before experimental forecast

The Forecast tab can generate a seasonality-calibrated synthetic continuation after the original dataset's last date. These generated values are explicitly labelled and shown separately from original observations. Forecasts fitted using this continuation are scenario outputs; they must not be described as actual weather predictions. The Random Forest classifier and reported hold-out metrics continue to use the original dataset only.
