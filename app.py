"""
Sri Lanka Rainfall Forecast - Ensemble Learning Dashboard
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Sri Lanka Rainfall - Ensemble ML", layout="wide")
st.markdown(
    "<style>"
    "#MainMenu, [data-testid='stAppDeployButton'], [data-testid='stToolbarActions'], [data-testid='stStatusWidget'], footer {display: none !important;}"
    "</style>",
    unsafe_allow_html=True,
)

MODEL_FILES = {
    "Random Forest": "random_forest",
    "Extra Trees": "extra_trees",
    "Bagging (Decision Trees)": "bagging",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "ANN (MLP)": "ann",
    "Stacking (6 models → Ridge)": "stacking",
}
SEASON_NAMES = {
    "NE_monsoon": "North-East monsoon (Dec-Feb)",
    "First_inter_monsoon": "First inter-monsoon (Mar-Apr)",
    "SW_monsoon": "South-West monsoon (May-Sep)",
    "Second_inter_monsoon": "Second inter-monsoon (Oct-Nov)",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def season_of(m):
    if m in (12, 1, 2):
        return "NE_monsoon"
    if m in (3, 4):
        return "First_inter_monsoon"
    if m in (5, 6, 7, 8, 9):
        return "SW_monsoon"
    return "Second_inter_monsoon"


# ------------------------------------------------------------------ loading
@st.cache_data
def load_tables():
    metrics = pd.read_csv(ROOT / "results/metrics.csv", index_col=0)
    by_zone = pd.read_csv(ROOT / "results/rmse_by_zone.csv", index_col=0)
    by_season = pd.read_csv(ROOT / "results/rmse_by_season.csv", index_col=0)
    imp = pd.read_csv(ROOT / "results/feature_importance.csv", index_col=0)
    weights = pd.read_csv(ROOT / "results/stacking_weights.csv", index_col=0)
    rf_curve = pd.read_csv(ROOT / "results/rf_trees_curve.csv")
    xgb_curve = pd.read_csv(ROOT / "results/xgb_rounds_curve.csv")
    test = pd.read_csv(ROOT / "data/processed/test_predictions.csv", parse_dates=["date", "target_date"])
    clim = pd.read_csv(ROOT / "data/processed/monthly_climatology.csv")
    cities = pd.read_csv(ROOT / "data/processed/city_info.csv", index_col=0)
    return metrics, by_zone, by_season, imp, weights, rf_curve, xgb_curve, test, clim, cities


@st.cache_resource
def load_models():
    import gc
    cfg = json.load(open(ROOT / "models/feature_config.json"))
    models = {}
    for name, f in MODEL_FILES.items():
        mdl = joblib.load(ROOT / f"models/{f}.pkl")
        if hasattr(mdl, "n_jobs"):
            mdl.n_jobs = 1
        models[name] = mdl
    arma_info = joblib.load(ROOT / "models/arma_params.pkl")["arma_params"]
    rf_resid_mdl = joblib.load(ROOT / "models/arma_rf_residual.pkl")
    if hasattr(rf_resid_mdl, "n_jobs"):
        rf_resid_mdl.n_jobs = 1
    gc.collect()
    return cfg, models, arma_info, rf_resid_mdl


metrics, by_zone, by_season, imp, weights, rf_curve, xgb_curve, test, clim, cities = load_tables()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Sri Lanka Rainfall")
st.sidebar.caption("Attanayake AMSL (224016H)")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model comparison",
        "Test set check (2023-2024)",
        "Predict tomorrow's rain",
        "About",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: ERA5 reanalysis via Open-Meteo, 12 cities, 2015-2024.")

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Predicting tomorrow's rainfall in Sri Lanka")
    st.write(
        "This project compares **bagging**, **boosting** and **stacking** ensembles (plus an ANN and an ARMA "
        "time-series model) for one question: *given today's weather in a Sri Lankan city, how much rain will "
        "fall tomorrow?* Models were trained on 2015-2022 and tested on 2023-2024, which they had never seen."
    )
    best = metrics.index[0]
    base = metrics.loc["Climatology (city-month mean)"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", len(cities))
    c2.metric("Daily records", f"{len(cities) * 3653:,}")
    c3.metric("Best model (RMSE)", best.split(" (")[0])
    c4.metric("Best test RMSE", f"{metrics.loc[best, 'RMSE']:.2f} mm",
              f"{metrics.loc[best, 'RMSE'] - base['RMSE']:.2f} vs climatology", delta_color="inverse")

    st.subheader("Rainfall pattern by city")
    sel = st.multiselect("Cities", list(cities.index), default=["Colombo", "Ratnapura", "Trincomalee", "Jaffna"])
    if sel:
        tbl = clim[clim.city.isin(sel)].pivot(index="month", columns="city", values="rain")
        tbl.index = MONTHS
        tbl = tbl.reindex(MONTHS)
        st.line_chart(tbl, y_label="average rain per day (mm)")
    st.caption("Wet-zone cities peak in the South-West monsoon and October-November; dry-zone cities get most "
               "of their rain from October to January (North-East monsoon).")

    st.subheader("Cities used")
    show = cities.rename(columns={"avg_annual_rain_mm": "avg annual rain (mm)", "rainy_days_pct": "rainy days (%)"})
    st.dataframe(show.sort_values("avg annual rain (mm)", ascending=False), width="stretch")

elif page == "Model comparison":
    st.title("Model comparison (test years 2023-2024)")
    st.write("Lower RMSE / MAE is better, higher R² is better. *Rain-day F1* checks whether the model gets "
             "rainy (≥ 1 mm) vs dry days right. *Heavy MAE* is the error on days with 25 mm or more.")

    fams = ["All"] + sorted(metrics.family.unique())
    fam = st.selectbox("Filter by family", fams)
    m = metrics if fam == "All" else metrics[metrics.family == fam]
    st.dataframe(
        m[["family", "RMSE", "MAE", "R2", "RainDay_Acc", "RainDay_F1", "Heavy_MAE", "Fit_sec"]]
        .style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}", "RainDay_Acc": "{:.3f}",
                       "RainDay_F1": "{:.3f}", "Heavy_MAE": "{:.1f}", "Fit_sec": "{:.0f}"}, na_rep="-")
        .highlight_min(subset=["RMSE", "MAE", "Heavy_MAE"], color="#d4edda")
        .highlight_max(subset=["R2", "RainDay_F1"], color="#d4edda"),
        width="stretch", height=min(40 + 35 * len(m), 700))

    metric = st.radio("Chart metric", ["RMSE", "MAE", "R2"], horizontal=True)
    st.bar_chart(m[metric].sort_values(), horizontal=True, height=450)

    st.subheader("Where are the errors?")
    c1, c2 = st.columns(2)
    c1.write("RMSE by climatic zone (mm)")
    c1.dataframe(by_zone.style.format("{:.2f}").background_gradient(cmap="Reds", axis=None), width="stretch")
    c2.write("RMSE by monsoon season (mm)")
    c2.dataframe(by_season.rename(columns=lambda c: c.replace("_", " ")).style.format("{:.2f}")
                 .background_gradient(cmap="Reds", axis=None), width="stretch")
    st.caption("Errors are much larger in the wet zone and in the inter-monsoon / monsoon months simply because "
               "there is more (and more variable) rain to predict there.")

    st.subheader("Bagging vs boosting behaviour")
    c1, c2 = st.columns(2)
    c1.write("**Random Forest** - test RMSE as more trees are averaged")
    c1.line_chart(rf_curve.set_index("trees"), y_label="RMSE (mm)")
    c1.caption("Error falls then flattens: averaging removes variance, more trees never hurt.")
    c2.write("**XGBoost** - train vs test RMSE as boosting rounds are added")
    c2.line_chart(xgb_curve.set_index("round"), y_label="RMSE (mm)")
    c2.caption("Train error keeps falling, test error levels off: boosting reduces bias but can start fitting noise.")

    st.subheader("Stacking weights and feature importance")
    c1, c2 = st.columns([1, 2])
    c1.write("Weight the Ridge meta-learner gives each base model")
    c1.bar_chart(weights["weight"], horizontal=True)
    c2.write("Top features")
    which = c2.selectbox("Importance from", list(imp.columns))
    c2.bar_chart(imp[which].sort_values(ascending=False).head(12), horizontal=True)

elif page == "Test set check (2023-2024)":
    st.title("Test set check: Predicted vs. Actual (2023-2024)")
    st.write("Pick a city and a day from the unseen 2023-2024 test period. Every model uses that day's weather to "
             "predict **the next day's rainfall**, and we compare the prediction against the real rainfall that occurred.")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", sorted(test.city.unique()), index=sorted(test.city.unique()).index("Colombo"))
    sub = test[test.city == city].sort_values("date")
    day = c2.date_input("Day (2023-2024)", value=pd.Timestamp("2024-05-20"),
                        min_value=sub.date.min(), max_value=sub.date.max())
    row = sub[sub.date == pd.Timestamp(day)].iloc[0]

    st.write(f"**Weather on {pd.Timestamp(day):%d %b %Y} in {city}** - {SEASON_NAMES[row.season]}")
    c = st.columns(6)
    c[0].metric("Rain", f"{row.rain:.1f} mm")
    c[1].metric("Rain last 7 days", f"{row.rain_sum_7d:.0f} mm")
    c[2].metric("Humidity", f"{row.humidity:.0f} %")
    c[3].metric("Cloud cover", f"{row.cloud_cover:.0f} %")
    c[4].metric("Max temp", f"{row.temp_max:.1f} °C")
    c[5].metric("Pressure", f"{row.pressure:.0f} hPa")

    pcols = [col for col in sub.columns if col.startswith("pred__")]
    res = pd.DataFrame({"model": [p.replace("pred__", "") for p in pcols],
                        "predicted (mm)": [row[p] for p in pcols]})
    res["error (mm)"] = res["predicted (mm)"] - row.rain_tomorrow
    res = res.sort_values("error (mm)", key=abs)
    st.write(f"**Actual rain on {row.target_date:%d %b %Y}: {row.rain_tomorrow:.1f} mm**")
    st.dataframe(res.style.format({"predicted (mm)": "{:.1f}", "error (mm)": "{:+.1f}"}),
                 width="stretch", hide_index=True)

    st.subheader("Six weeks around this day")
    models_plot = st.multiselect("Models to show", [p.replace("pred__", "") for p in pcols],
                                 default=["Random Forest", "XGBoost", "Stacking (6 models → Ridge)"])
    w = sub[(sub.date >= pd.Timestamp(day) - pd.Timedelta(days=21)) & (sub.date <= pd.Timestamp(day) + pd.Timedelta(days=21))]
    chart = w.set_index("target_date")[["rain_tomorrow"] + ["pred__" + mm for mm in models_plot]]
    chart.columns = ["actual"] + models_plot
    st.line_chart(chart, y_label="rain (mm)")

elif page == "Predict tomorrow's rain":
    st.title("Predict tomorrow's rain (Live model calculator)")
    st.write("Enter today's weather readings below to get a live forecast for **tomorrow's rainfall** from all "
             "11 trained models (including the ARMA + Random Forest hybrid, RF + ANN hybrid, and Stacking ensemble).")
    cfg, models, arma_info, rf_resid_mdl = load_models()

    c1, c2 = st.columns(2)
    city = c1.selectbox("City", sorted(test.city.unique()))
    month = c2.selectbox("Month", MONTHS, index=4)
    m_num = MONTHS.index(month) + 1
    # use the median day of that city & month in the test years as the starting point
    base = test[(test.city == city) & (test.date.dt.month == m_num)]
    row = base.iloc[(base["rain"] - base["rain"].median()).abs().argsort().iloc[0]].copy()

    st.write("**Today's weather**")
    c = st.columns(4)
    rain = c[0].slider("Rain today (mm)", 0.0, 150.0, float(round(row.rain, 1)), 0.5)
    rain_hours = c[1].slider("Hours of rain today", 0.0, 24.0, float(row.rain_hours), 1.0)
    humidity = c[2].slider("Humidity (%)", 50.0, 100.0, float(round(row.humidity)), 1.0)
    cloud = c[3].slider("Cloud cover (%)", 0.0, 100.0, float(round(row.cloud_cover)), 1.0)
    c = st.columns(4)
    temp_max = c[0].slider("Max temperature (°C)", 10.0, 40.0, float(round(row.temp_max, 1)), 0.5)
    pressure = c[1].slider("Sea-level pressure (hPa)", 1000.0, 1020.0, float(round(row.pressure, 1)), 0.5)
    rain7 = c[2].slider("Rain in the last 7 days (mm)", 0.0, 500.0, float(round(row.rain_sum_7d)), 5.0)
    wind = c[3].slider("Max wind speed (km/h)", 0.0, 60.0, float(round(row.wind_max)), 1.0)

    # update the row and keep the derived features consistent
    d_rain = rain - row.rain
    row["rain"], row["rain_hours"], row["humidity"], row["cloud_cover"] = rain, rain_hours, humidity, cloud
    row["humidity_3d"] += (humidity - base.humidity.median()) / 3
    row["cloud_3d"] += (cloud - base.cloud_cover.median()) / 3
    row["temp_max"], row["pressure"], row["wind_max"] = temp_max, pressure, wind
    row["temp_range"] = row.temp_max - row.temp_min
    row["temp_mean"] = (row.temp_max + row.temp_min) / 2
    row["dewpoint_depression"] = row.temp_mean - row.dew_point
    row["rain_sum_3d"] = max(row.rain_sum_3d + d_rain, rain)
    row["rain_sum_7d"] = max(rain7, row.rain_sum_3d)
    row["rain_sum_14d"] = max(row.rain_sum_14d, row.rain_sum_7d)
    row["rain_sum_30d"] = max(row.rain_sum_30d, row.rain_sum_14d)
    row["month"] = m_num
    doy = pd.Timestamp(2024, m_num, 15).dayofyear
    row["doy_sin"], row["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    row["season"] = season_of(m_num)
    for s in SEASON_NAMES:
        row["season_" + s] = int(row.season == s)

    X = pd.DataFrame([row[cfg["FEATURES"]].astype(float)])
    X_cb = pd.DataFrame([row[cfg["CB_FEATURES"]]])
    for col in cfg["BASE_FEATURES"]:
        X_cb[col] = X_cb[col].astype(float)

    out = {}
    for name, mdl in models.items():
        p = mdl.predict(X_cb if name == "CatBoost" else X)[0]
        out[name] = max(float(p), 0.0)
    out["Hybrid: RF + ANN (average)"] = (out["Random Forest"] + out["ANN (MLP)"]) / 2.0
    ap = arma_info[city]
    arma_base = float(ap["const"]) + float(ap["ar.L1"]) * (rain - float(ap["const"])) + float(ap["ar.L2"]) * (float(row["rain_lag1"]) - float(ap["const"]))
    out["Hybrid: ARMA + RF (residuals)"] = max(arma_base + float(rf_resid_mdl.predict(X)[0]), 0.0)
    out = pd.Series(out).sort_values()

    st.write(f"**Predicted rain for tomorrow in {city}** ({SEASON_NAMES[row.season]})")
    c1, c2 = st.columns([1, 2])
    c1.metric("Stacking ensemble", f"{out['Stacking (6 models → Ridge)']:.1f} mm")
    c1.metric("Range across models", f"{out.min():.1f} - {out.max():.1f} mm")
    clim_val = clim[(clim.city == city) & (clim.month == m_num)].rain.iloc[0]
    c1.caption(f"Normal for {city} in {month}: {clim_val:.1f} mm/day")
    c2.bar_chart(out, horizontal=True, y_label="", x_label="mm")
    st.caption("Tip: raise the rain hours, humidity and cloud cover and watch the tree models jump in steps while "
               "the ANN changes smoothly.")

else:
    st.title("About this project")
    st.markdown("""
**Goal.** Compare and combine ensemble learning methods on a real Sri Lankan problem: next-day rainfall.

**Data.** Daily weather for 12 Sri Lankan cities (Colombo, Galle, Ratnapura, Kandy, Nuwara Eliya, Kurunegala,
Badulla, Anuradhapura, Jaffna, Trincomalee, Batticaloa, Hambantota), 2015-2024, from the Open-Meteo historical API
(ECMWF ERA5 reanalysis).

**Models.**
- Bagging: Bagging (decision trees), Random Forest, Extra Trees
- Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Neural network: MLP
- Time series: ARMA(2,1) per city
- Hybrids: ARMA + Random Forest (RF learns ARMA's residuals), Random Forest + ANN (average)
- Stacking: RF, Extra Trees, XGBoost, LightGBM, CatBoost and ANN combined by a Ridge meta-learner

**Files.** The full analysis is in `ensemble_rainfall_sri_lanka.ipynb`. Trained models are in `models/` as pickle files.

**AI declaration.** Code and frontend partially generated with Claude via Antigravity, then reviewed and tested by me.
The comparison and interpretation of the results are my own.
""")
