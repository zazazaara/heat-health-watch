"""Heat-Health Risk Monitoring, Prediction and Early Warning System (Delhi) — Streamlit dashboard.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # reliable imports from any working dir

import pandas as pd
import streamlit as st

from src import config as C
from src import forecasting as F
from src import hourly_analysis as H
from src import risk_classifier as RC
from src import synthetic_data as S
from src import visualization as V
from src.advisories import advisory_for_category
from src.data_loader import DataValidationError, load_dataset, to_csv_bytes
from src.risk_scoring import InvalidInputError, assess

st.set_page_config(page_title="Heat-Health Risk Monitoring System", page_icon="🌡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.tag {display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:0.9rem;}
.banner-syn {background:#fff3e0;border-left:6px solid #ef6c00;padding:10px 14px;border-radius:6px;margin-bottom:10px;}
.banner-sim {background:#e3f2fd;border-left:6px solid #1565c0;padding:10px 14px;border-radius:6px;margin-bottom:10px;}
.banner-ill {background:#fce4ec;border-left:6px solid #c2185b;padding:10px 14px;border-radius:6px;margin-bottom:10px;}
</style>""", unsafe_allow_html=True)


def tag(category: str) -> str:
    return f"<span class='tag' style='background:{C.LEVEL_COLORS.get(category, '#607d8b')}'>{category}</span>"


# ------------------------------------------------------------------ data / model
@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    df, report = load_dataset()
    return df, report.as_dict()


@st.cache_resource(show_spinner="Preparing Random Forest model…")
def get_model(_df: pd.DataFrame):
    return RC.load_or_train(_df)


@st.cache_data(show_spinner="Evaluating forecasting model on the hold-out block…")
def get_forecast_eval(_df: pd.DataFrame):
    return F.evaluate(_df)


@st.cache_resource(show_spinner=False)
def get_forecast_models(_df: pd.DataFrame):
    return F.fit_all(_df)


@st.cache_data(show_spinner=False)
def get_synthetic_continuation(_df: pd.DataFrame, end_date, seed: int):
    return F.generate_synthetic_continuation(_df, end_date, seed=seed)


try:
    data, quality = get_data()
except DataValidationError as exc:
    st.error(f"**Dataset problem:** {exc}")
    st.stop()

LAST_DATE = data[C.DATE_COL].max()
FIRST_DATE = data[C.DATE_COL].min()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌡️ Heat-Health Risk")
    st.caption("Delhi · Historical Environmental Monitoring & Risk Assessment")
    st.markdown(f"**Dataset coverage**  \n{FIRST_DATE.date()} → **{LAST_DATE.date()}**  \n"
                f"{quality['rows_clean']:,} daily records · *daily, not hourly*")
    sel = st.date_input("Historical date range", value=(FIRST_DATE.date(), LAST_DATE.date()),
                        min_value=FIRST_DATE.date(), max_value=LAST_DATE.date())
    st.divider()
    st.caption(C.DISCLAIMER)

if not isinstance(sel, (tuple, list)) or len(sel) != 2:
    st.info("Select both a start and an end date in the sidebar to continue.")
    st.stop()
start_d, end_d = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])
view = data[(data[C.DATE_COL] >= start_d) & (data[C.DATE_COL] <= end_d)]

st.title("Heat-Health Risk Monitoring, Prediction and Early Warning System")
st.info(f"📅 **Historical data ends on {LAST_DATE.date():%d %B %Y}.** Anything dated after that is an estimate "
        "(see the Forecast tab); anything hourly is a simulation.", icon=None)
if view.empty:
    st.warning("No records in the selected date range. Widen the range in the sidebar.")
    st.stop()

tabs = st.tabs(["Overview", "Risk Calculator", "Trends", "ML Risk Classifier", "Forecast (Experimental)",
                "Hourly (Simulated)", "Synthetic Scenarios", "Data Explorer"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    last = view.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records in range", f"{len(view):,}")
    c2.metric("Latest mean temp", f"{last['Mean_Temperature_C']:.1f} °C")
    c3.metric("Latest AQI", f"{last['AQI']:.0f}")
    c4.metric("Latest risk score", f"{last[C.SCORE_COL]:.2f}")
    c5.markdown("**Latest category**<br>" + tag(last["Score_Category_5L"]), unsafe_allow_html=True)
    st.caption(f"Latest record in selected range: **{last[C.DATE_COL].date()}** · original dataset label: "
               f"**{last[C.TARGET_COL]}**")
    adv = advisory_for_category(last["Score_Category_5L"])
    st.success(f"**Advisory:** {adv['summary']}")
    st.caption(adv["disclaimer"])
    a, b = st.columns(2)
    counts5 = view["Score_Category_5L"].value_counts().to_dict()
    a.plotly_chart(V.distribution_bar(counts5, "Days per project-defined level (5-level scale)", C.LEVEL_ORDER),
                   width="stretch")
    b.plotly_chart(V.distribution_bar(view[C.TARGET_COL].value_counts().to_dict(),
                                      "Original dataset labels (Risk_Category)", ["Moderate", "High"]),
                   width="stretch")
    if counts5.get("Very Low", 0) == 0 and counts5.get("Severe", 0) == 0:
        st.caption("ℹ️ The real data never reaches Very Low or Severe on the 5-level scale (scores span "
                   f"{view[C.SCORE_COL].min():.1f}–{view[C.SCORE_COL].max():.1f}). Use *Synthetic Scenarios* to see those levels.")
    st.plotly_chart(V.line_chart(view, C.DATE_COL, ["Max_Temperature_C", "Mean_Temperature_C"], "Temperature", "°C"),
                    width="stretch")
    st.plotly_chart(V.score_with_levels(view, C.DATE_COL, C.SCORE_COL, "Heat-Health Risk Score over time"),
                    width="stretch")

# ------------------------------------------------------------------ Risk calculator
with tabs[1]:
    st.subheader("Risk calculator")
    st.caption("Enter conditions to compute the Heat-Health Risk Score = 45·NormTemp + 25·NormHumidity + 30·(AQI/500).")
    ca, cb, cc = st.columns(3)
    t_in = ca.number_input("Max temperature (°C)", -5.0, 55.0, 38.0, 0.5, key="calc_t")
    h_in = cb.number_input("Relative humidity (%)", 0.0, 100.0, 45.0, 1.0, key="calc_h")
    a_in = cc.number_input("AQI", 0.0, 500.0, 250.0, 5.0, key="calc_a")
    if st.button("Calculate risk", type="primary", key="calc_btn"):
        try:
            st.session_state["calc"] = assess(t_in, h_in, a_in)
        except InvalidInputError as exc:
            st.session_state.pop("calc", None)
            st.error(str(exc))
    if "calc" in st.session_state:
        r = st.session_state["calc"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score (0-100)", f"{r['score']:.2f}")
        m2.markdown("**Project-defined level**<br>" + tag(r["category"]), unsafe_allow_html=True)
        m3.metric("Original dataset rule (score > 55)", r["dataset_rule_label"])
        st.bar_chart(pd.Series(r["components"], name="points"))
        adv = advisory_for_category(r["category"])
        st.success(adv["summary"])
        for item in adv["actions"]:
            st.write("• " + item)
        st.caption(adv["disclaimer"])

# ------------------------------------------------------------------ Trends
with tabs[2]:
    st.subheader("Historical trends")
    st.plotly_chart(V.line_chart(view, C.DATE_COL, ["Max_Temperature_C", "Mean_Temperature_C", "Min_Temperature_C"],
                                 "Temperature over time", "°C"), width="stretch")
    st.plotly_chart(V.line_chart(view, C.DATE_COL, ["Relative_Humidity_%"], "Relative humidity over time", "%"),
                    width="stretch")
    pol = st.multiselect("Pollutants / AQI", ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"],
                         default=["AQI", "PM2.5", "PM10"], key="pol_sel")
    if pol:
        st.plotly_chart(V.line_chart(view, C.DATE_COL, pol, "Air quality over time"), width="stretch")
    else:
        st.info("Select at least one pollutant to display the chart.")
    st.plotly_chart(V.score_with_levels(view, C.DATE_COL, C.SCORE_COL, "Risk score trend"), width="stretch")
    seas = view.groupby("Season")[["Max_Temperature_C", "AQI", C.SCORE_COL]].mean().round(1)
    if len(seas) >= 2:
        st.markdown("**Seasonal comparison (mean values in selected range)**")
        st.dataframe(seas, width="stretch")
        st.bar_chart(seas[C.SCORE_COL], y_label="Mean risk score")
    else:
        st.caption("Seasonal comparison needs at least two seasons in the selected range.")

# ------------------------------------------------------------------ ML classifier
with tabs[3]:
    st.subheader("Random Forest risk classifier")
    model = get_model(data)
    pipe, m = model["pipeline"], model["metrics"]
    st.warning("The dataset's `Risk_Category` has only **two classes (Moderate, High)**. The Random Forest learns those "
               "two; it does **not** predict Very Low / Low / Severe. Those five levels come from the rule-based score "
               "(Calculator, Forecast, Synthetic tabs).")
    st.markdown(f"**Target:** `{m['target']}` · **Features ({len(m['features'])}):** {', '.join(m['features'])}  \n"
                f"**Split:** {m['split']} · train {m['train_rows']} rows ({m['train_period'][0]} → {m['train_period'][1]}) · "
                f"test {m['test_rows']} rows ({m['test_period'][0]} → {m['test_period'][1]}) · model {model['source']}")
    ca, cb = st.columns(2)
    ca.markdown("**Actual class distribution — all data**")
    ca.dataframe(data[C.TARGET_COL].value_counts().rename("days"), width="stretch")
    cb.markdown("**Actual class distribution — held-out test block**")
    cb.dataframe(pd.Series(m["test_class_counts"], name="days"), width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Test accuracy (held-out)", f"{m['accuracy_test']:.2%}")
    k2.metric("Majority-class baseline", f"{m['majority_class_baseline_accuracy']:.2%}")
    k3.metric("Macro F1", f"{m['macro_f1']:.3f}")
    st.dataframe(pd.DataFrame(m["per_class"]).T.round(3), width="stretch")
    p1, p2 = st.columns(2)
    p1.plotly_chart(V.confusion_heatmap(m["confusion_matrix"], m["classes"]), width="stretch")
    p2.plotly_chart(V.importance_bar(m["feature_importance"]), width="stretch")
    st.caption(m["note"])

    st.markdown("### Predict from inputs")
    med = data[C.RF_FEATURES].median()
    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(C.RF_FEATURES):
        vals[f] = cols[i % 4].number_input(f.replace("_", " "), value=float(round(med[f], 2)), key=f"rf_{f}")
    tmax_in = st.number_input("Max temperature (°C) — only used for the separate rule-based score", -5.0, 55.0,
                              float(round(data["Max_Temperature_C"].median(), 1)), key="rf_tmax")
    if st.button("Predict risk class", type="primary", key="rf_btn"):
        try:
            st.session_state["rf_pred"] = (RC.predict_one(pipe, vals), assess(tmax_in, vals["Relative_Humidity_%"], vals["AQI"]))
        except (ValueError, InvalidInputError) as exc:
            st.session_state.pop("rf_pred", None)
            st.error(str(exc))
    if "rf_pred" in st.session_state:
        pred, rule = st.session_state["rf_pred"]
        x1, x2 = st.columns(2)
        x1.metric("Random Forest PREDICTED class (learned)", pred["predicted_label"])
        x1.caption("Vote shares: " + ", ".join(f"{k} {v:.0%}" for k, v in pred["vote_share"].items()) +
                   " — not a calibrated probability of illness.")
        x2.markdown("**Rule-based score level (separate, project-defined)**<br>" + tag(rule["category"]) +
                    f" &nbsp; score {rule['score']:.2f}", unsafe_allow_html=True)

    st.markdown("### Actual vs predicted for a historical day")
    pick = st.date_input("Choose a date in the dataset", value=LAST_DATE.date(), min_value=FIRST_DATE.date(),
                         max_value=LAST_DATE.date(), key="rf_hist_date")
    row = data[data[C.DATE_COL] == pd.Timestamp(pick)]
    if row.empty:
        st.info("That date is not present in the dataset (a missing calendar day). Choose another date.")
    else:
        r = row.iloc[0]
        pr = RC.predict_one(pipe, {f: r[f] for f in C.RF_FEATURES})
        in_test = pd.Timestamp(pick) >= pd.Timestamp(m["test_period"][0])
        y1, y2 = st.columns(2)
        y1.metric("ACTUAL label (dataset)", r[C.TARGET_COL])
        y2.metric("PREDICTED label (Random Forest)", pr["predicted_label"])
        st.caption("This date is in the **held-out test block**." if in_test else
                   "⚠️ This date was in the **training block**; agreement here is not an unbiased test.")

# ------------------------------------------------------------------ Forecast
with tabs[4]:
    st.subheader("Synthetic continuation + experimental future estimates")
    today = max(pd.Timestamp.today().normalize(), LAST_DATE + pd.Timedelta(days=1))
    st.markdown(f"<div class='banner-ill'><b>Transparent scenario workflow.</b> Original observations end on "
                f"<b>{LAST_DATE.date()}</b>. The app can generate a seasonality-calibrated synthetic continuation "
                "from the original data, then estimate dates after that continuation. Synthetic rows are not real "
                "measurements, and the resulting forecast is scenario-based—not an official weather forecast.</div>",
                unsafe_allow_html=True)
    synthetic_end = st.date_input(
        "Generate synthetic daily continuation through",
        value=today.date(), min_value=(LAST_DATE + pd.Timedelta(days=1)).date(),
        max_value=today.date(), key="f_synthetic_end",
        help="Generated values preserve original rows and follow seasonal patterns learned from the original dataset.",
    )
    synthetic = get_synthetic_continuation(data, synthetic_end, C.RANDOM_SEED)
    observed_for_chart = data[[C.DATE_COL, "Max_Temperature_C"]].copy()
    if not synthetic.empty:
        synthetic_for_chart = synthetic[[C.DATE_COL, "Max_Temperature_C"]].copy()
        history = pd.concat([
            data[[C.DATE_COL, *F.FORECAST_VARS]].copy(),
            synthetic[[C.DATE_COL, *F.FORECAST_VARS, "Data_Source"]].copy(),
        ], ignore_index=True).sort_values(C.DATE_COL).reset_index(drop=True)
    else:
        synthetic_for_chart = pd.DataFrame(columns=[C.DATE_COL, "Max_Temperature_C"])
        history = data[[C.DATE_COL, *F.FORECAST_VARS]].copy()
    st.caption(f"Original data: {len(data):,} rows through {LAST_DATE.date()} · "
               f"Synthetic continuation: {len(synthetic):,} generated daily rows through {pd.Timestamp(synthetic_end).date()}.")
    st.warning("The synthetic continuation is a simulation calibrated to historical seasonality, not a claim about what actually happened. "
               "The forecast uses original + synthetic rows, so treat its output as an exploratory scenario.")

    d0 = (pd.Timestamp(synthetic_end) + pd.Timedelta(days=1)).date()
    fa, fb = st.columns(2)
    f_start = fa.date_input("Forecast start date (after synthetic continuation)", value=d0, min_value=d0, key=f"f_start_{synthetic_end}")
    default_end = (pd.Timestamp(d0) + pd.Timedelta(days=13)).date()
    f_end = fb.date_input("Forecast end date", value=default_end, min_value=d0, key=f"f_end_{synthetic_end}")
    mode = st.radio("Humidity and AQI inputs", ["Forecast them (seasonal + AR1)", "I supply assumptions"],
                    horizontal=True, key="f_mode")
    over = {}
    if mode.startswith("I supply"):
        oa, ob = st.columns(2)
        over["Relative_Humidity_%"] = oa.number_input("Assumed relative humidity (%)", 0.0, 100.0, 50.0, key="f_rh")
        over["AQI"] = ob.number_input("Assumed AQI", 0.0, 500.0, 250.0, key="f_aqi")
    if st.button("Generate synthetic continuation + forecast", type="primary", key="f_btn"):
        try:
            models = F.fit_all(history)
            res = F.forecast_range(history, f_start, f_end, over or None, models)
            st.session_state["forecast"] = res
            st.session_state["forecast_context"] = {
                "synthetic_end": str(synthetic_end), "start": str(f_start), "end": str(f_end),
            }
        except F.ForecastRequestError as exc:
            st.session_state.pop("forecast", None)
            st.session_state.pop("forecast_context", None)
            st.error(f"Forecast not calculated: {exc}")
    context = st.session_state.get("forecast_context", {})
    current_context = {"synthetic_end": str(synthetic_end), "start": str(f_start), "end": str(f_end)}
    if "forecast" in st.session_state and context == current_context:
        fr = st.session_state["forecast"]
        n_ill = int((fr["Reliability"] == "Illustrative scenario (>30 d)").sum())
        if n_ill:
            st.warning(f"{n_ill} of {len(fr)} forecast days are more than 30 days after the synthetic continuation; "
                       "these are illustrative seasonal scenarios, not reliable weather predictions.")
        st.plotly_chart(V.forecast_context_chart(observed_for_chart, synthetic_for_chart, fr), width="stretch")
        st.plotly_chart(V.forecast_score_chart(fr), width="stretch")
        show_cols = ["Date", "Horizon_days", "Reliability", "Max_Temperature_C", "Max_Temperature_C_status",
                     "Relative_Humidity_%", "Relative_Humidity_%_status", "AQI", "AQI_status", "Risk_Score",
                     "Score_Category_5L", "Dataset_Rule_Label", "RF_Class", "Advisory", "Method", "History_Basis"]
        show = fr[show_cols].copy()
        show["Date"] = show["Date"].dt.date
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button("⬇ Download scenario estimates CSV", to_csv_bytes(fr), "experimental_scenario_estimates.csv", "text/csv", key="f_dl")
        st.caption("The Random Forest class is unavailable for future dates because forecasts for all required ML features "
                   "(Heat Index, wind and pollutant concentrations) are not generated. Forecast bands are approximate 80% intervals.")
        st.caption(C.DISCLAIMER)
    elif "forecast" in st.session_state:
        st.info("Forecast settings changed. Click **Generate synthetic continuation + forecast** to refresh the chart.")
    with st.expander("Experimental model evaluation (original observed data only)"):
        ev = get_forecast_eval(data)
        st.dataframe(ev.round(2), width="stretch", hide_index=True)
        st.caption("Evaluation uses only the original observed dataset. Synthetic rows are never used as ground truth.")

# ------------------------------------------------------------------ Hourly
with tabs[5]:
    st.subheader("Hourly risk analysis (simulation)")
    st.markdown(f"<div class='banner-sim'><b>{H.SIM_LABEL}.</b> The dataset has one row per day; hourly values below "
                "are generated from documented assumptions (diurnal temperature cycle, constant dew point, assumed AQI "
                "profile). The peak-risk hour is computed from the profile, not fixed.</div>", unsafe_allow_html=True)
    src = st.radio("Source of daily values", ["Historical day from the dataset", "Custom values"], horizontal=True, key="h_src")
    if src.startswith("Historical"):
        hd = st.date_input("Date", value=LAST_DATE.date(), min_value=FIRST_DATE.date(), max_value=LAST_DATE.date(), key="h_date")
        hrow = data[data[C.DATE_COL] == pd.Timestamp(hd)]
        hin = None if hrow.empty else (float(hrow.iloc[0]["Min_Temperature_C"]), float(hrow.iloc[0]["Max_Temperature_C"]),
                                       float(hrow.iloc[0]["Relative_Humidity_%"]), float(hrow.iloc[0]["AQI"]))
        if hin is None:
            st.info("That date is not in the dataset. Choose another date.")
        else:
            st.caption(f"Daily inputs: Tmin {hin[0]} °C, Tmax {hin[1]} °C, RH {hin[2]} %, AQI {hin[3]:.0f}")
    else:
        hd = st.date_input("Label date", value=LAST_DATE.date(), key="h_cdate")
        c1, c2, c3, c4 = st.columns(4)
        hin = (c1.number_input("Tmin (°C)", -5.0, 55.0, 28.0, key="h_tmin"), c2.number_input("Tmax (°C)", -5.0, 55.0, 41.0, key="h_tmax"),
               c3.number_input("Mean RH (%)", 0.0, 100.0, 35.0, key="h_rh"), c4.number_input("Mean AQI", 0.0, 500.0, 250.0, key="h_aqi"))
    if st.button("Generate simulated hourly profile", type="primary", key="h_btn"):
        if hin is None:
            st.error("No daily values available for the chosen date.")
        else:
            try:
                st.session_state["hourly"] = H.simulate_hourly(*hin, date=pd.Timestamp(hd))
            except H.HourlyInputError as exc:
                st.session_state.pop("hourly", None)
                st.error(str(exc))
    if "hourly" in st.session_state:
        hh = st.session_state["hourly"]
        pk = H.peak_hour(hh)
        k1, k2, k3 = st.columns(3)
        k1.metric("Highest simulated risk hour", pk["time"])
        k2.metric("Score at that hour", f"{pk['score']:.2f}")
        k3.markdown("**Level at peak**<br>" + tag(pk["category"]), unsafe_allow_html=True)
        st.plotly_chart(V.hourly_chart(hh), width="stretch")
        wins = H.risk_windows(hh, "High")
        if wins:
            st.warning("Time-wise advisory — simulated **High or Severe** window(s): " +
                       "; ".join(f"{w['from']}–{w['to']}" for w in wins) + ". " + advisory_for_category("High")["summary"])
        else:
            st.success("No simulated hour reaches High. " + advisory_for_category(pk["category"])["summary"])
        st.dataframe(hh[["Time", "Temperature_C", "Relative_Humidity_%", "AQI", "Risk_Score", "Score_Category_5L", "Data_Source"]],
                     width="stretch", hide_index=True)
        st.download_button("⬇ Download hourly CSV", to_csv_bytes(hh), "hourly_simulation.csv", "text/csv", key="h_dl")
        st.caption(C.DISCLAIMER)

# ------------------------------------------------------------------ Synthetic
with tabs[6]:
    st.subheader("Synthetic scenarios")
    st.markdown(f"<div class='banner-syn'><b>{S.SYNTHETIC_LABEL}.</b> Generated for demonstration and testing across all five "
                "project-defined levels. Never mixed with the historical records and not used to train or test the "
                "Random Forest.</div>", unsafe_allow_html=True)
    sa, sb = st.columns(2)
    n_lvl = sa.number_input("Scenarios per level", 1, S.MAX_PER_LEVEL, 10, key="syn_n")
    seed = sb.number_input("Random seed", 0, 1_000_000, C.RANDOM_SEED, key="syn_seed")
    if st.button("Generate synthetic scenarios", type="primary", key="syn_btn"):
        try:
            st.session_state["syn"] = S.generate_scenarios(int(n_lvl), int(seed))
        except (ValueError, RuntimeError) as exc:
            st.session_state.pop("syn", None)
            st.error(str(exc))
    if "syn" in st.session_state:
        sy = st.session_state["syn"]
        probs = S.validate_synthetic(sy)
        if probs:
            st.error("Validation failed: " + "; ".join(probs))
        else:
            st.caption(f"{len(sy)} scenarios · validation passed · seed {seed}")
        st.plotly_chart(V.synthetic_scatter(sy), width="stretch")
        st.dataframe(sy, width="stretch", hide_index=True)
        st.download_button("⬇ Download synthetic CSV", to_csv_bytes(sy), "synthetic_scenarios.csv", "text/csv", key="syn_dl")

# ------------------------------------------------------------------ Data explorer
with tabs[7]:
    st.subheader("Data explorer (historical records)")
    e1, e2, e3 = st.columns(3)
    lab = e1.multiselect("Original label", ["Moderate", "High"], default=["Moderate", "High"], key="ex_lab")
    lvl = e2.multiselect("5-level category", C.LEVEL_ORDER, default=C.LEVEL_ORDER, key="ex_lvl")
    q = e3.text_input("Search (date, season, conditions)", key="ex_q")
    ex = view[view[C.TARGET_COL].isin(lab) & view["Score_Category_5L"].isin(lvl)]
    if q.strip():
        ql = q.strip().lower()
        mask = (ex[C.DATE_COL].dt.strftime("%Y-%m-%d").str.contains(ql) | ex["Season"].str.lower().str.contains(ql)
                | ex["Conditions"].str.lower().str.contains(ql))
        ex = ex[mask]
    show_cols = ["Date", "Max_Temperature_C", "Mean_Temperature_C", "Relative_Humidity_%", "AQI", "PM2.5", "PM10",
                 C.SCORE_COL, C.TARGET_COL, "Score_Category_5L", "Season", "Conditions"]
    cols_sel = st.multiselect("Columns", list(data.columns), default=show_cols, key="ex_cols")
    if ex.empty:
        st.info("No rows match the current filters.")
    elif not cols_sel:
        st.info("Select at least one column.")
    else:
        st.caption(f"{len(ex):,} rows (click a column header to sort)")
        st.dataframe(ex[cols_sel], width="stretch", hide_index=True)
        st.download_button("⬇ Download filtered CSV", to_csv_bytes(ex[cols_sel]), "historical_filtered.csv", "text/csv", key="ex_dl")
    with st.expander("Data quality report"):
        st.write(f"Rows in file: **{quality['rows_raw']}** · after removing exact duplicates: **{quality['rows_clean']}**")
        st.write(f"Date range: **{quality['date_min']} → {quality['date_max']}**")
        st.write("Missing values per column: " + (str(quality["missing_values"]) if quality["missing_values"] else "none"))
        for n in quality["notes"]:
            st.write("• " + n)
        st.write("Original file is never modified; cleaning is in memory only.")
