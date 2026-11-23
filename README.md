# Predicting Next-Day Rainfall in Sri Lanka with Ensemble Learning

**Module:** Machine Learning - Ensemble Learning Assignment  
**Student:** Attanayake AMSL (`224016H`)  
**Live Hosted Dashboard:** https://automataassignmentml-dgz85rykfhwevvlamnghgz.streamlit.app/  
**Dataset:** Real Daily Meteorological Data for 12 Sri Lankan Cities (2015-2024, 43,836 observations) via ECMWF ERA5 Reanalysis / Open-Meteo Historical Weather API.

---

## 1. Project Overview

This project predicts **tomorrow's daily rainfall (in mm)** across **12 cities in Sri Lanka** covering all three climatic zones defined by the Department of Meteorology, Sri Lanka:
- **Wet Zone:** Colombo, Galle, Ratnapura, Kandy, Nuwara Eliya
- **Intermediate Zone:** Kurunegala, Badulla
- **Dry Zone:** Anuradhapura, Jaffna, Trincomalee, Batticaloa, Hambantota

We compare and combine **17 models across 7 families** using a strict time-based split (**Train: 2015-2022 [34,980 rows]**, **Walk-Forward Validation: 2020-2022**, **Test: 2023-2024 [8,760 rows]**):

1. **Baselines:** Persistence (`today = tomorrow`), City-Month Climatology, Linear Regression, Single Decision Tree
2. **Bagging:** Bagging (Decision Trees), Random Forest, Extra Trees
3. **Boosting:** AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost (using native categorical features for `city`, `zone`, `season`)
4. **Neural Network:** Multi-Layer Perceptron (`MLPRegressor`, 64→32 ReLU layers)
5. **Time-Series:** Per-city `ARMA(2,1)` one-step-ahead forecasting
6. **Hybrid Models:**
   - **Hybrid `ARMA(2,1) + Random Forest` (Residual Learning):** ARMA models multi-day rainy spells; Random Forest predicts the leftover ARMA errors using today's weather features.
   - **Hybrid `Random Forest + ANN`:** Averages step-shaped tree predictions with smooth neural-network predictions.
7. **Stacking Ensemble:** Combines 6 base models (`Random Forest`, `Extra Trees`, `XGBoost`, `LightGBM`, `CatBoost`, `ANN`) using 5-fold out-of-fold predictions into a non-negative **Ridge Regression** meta-learner.

---

## 2. Summary of Test Results (2023-2024, 8,760 Unseen City-Days)

| Model | Family | RMSE (mm) ↓ | MAE (mm) ↓ | R² ↑ | Rain-Day Acc ↑ | Rain-Day F1 ↑ | Heavy-Rain MAE (≥25 mm) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking (6 models → Ridge)** | **Stacking** | **8.081** | 4.560 | **0.394** | **0.767** | **0.842** | **21.34** |
| **Hybrid: RF + ANN (average)** | **Hybrid** | 8.113 | **4.552** | 0.390 | 0.761 | 0.839 | 21.78 |
| **Random Forest** | **Bagging** | 8.148 | 4.655 | 0.384 | 0.745 | 0.833 | 22.09 |
| **Hybrid: ARMA + RF (residuals)** | **Hybrid** | 8.151 | 4.639 | 0.384 | 0.741 | 0.831 | 22.12 |
| **LightGBM** | **Boosting** | 8.175 | 4.703 | 0.380 | 0.656 | 0.792 | 22.90 |
| **CatBoost** | **Boosting** | 8.177 | 4.652 | 0.380 | 0.727 | 0.824 | 22.37 |
| **Extra Trees** | **Bagging** | 8.180 | 4.632 | 0.380 | 0.740 | 0.830 | 22.82 |
| **Bagging (Decision Trees)** | **Bagging** | 8.189 | 4.679 | 0.378 | 0.747 | 0.834 | 21.90 |
| **XGBoost** | **Boosting** | 8.214 | 4.702 | 0.374 | 0.704 | 0.814 | 23.14 |
| **Linear Regression** | **Baseline** | 8.215 | 4.725 | 0.374 | 0.737 | 0.825 | 22.81 |
| **ANN (MLP)** | **Neural network** | 8.279 | 4.602 | 0.364 | 0.769 | 0.839 | 21.64 |
| **Gradient Boosting** | **Boosting** | 8.304 | 4.678 | 0.361 | 0.724 | 0.823 | 22.86 |
| **ARMA(2,1) per city** | **Time series** | 8.448 | 4.959 | 0.338 | 0.662 | 0.795 | 23.49 |
| **AdaBoost** | **Boosting** | 8.785 | 5.705 | 0.284 | 0.656 | 0.792 | 18.49 |
| **Climatology (city-month mean)** | **Baseline** | 9.492 | 5.508 | 0.165 | 0.665 | 0.796 | 28.74 |
| **Persistence (today=tomorrow)** | **Baseline** | 9.690 | 5.160 | 0.129 | 0.772 | 0.826 | 19.85 |
| **Single Decision Tree** | **Baseline** | 11.104 | 6.075 | -0.143 | 0.724 | 0.797 | 25.90 |

---

## 3. Repository Structure

```text
SL_Weather_Ensemble_ML/
├── app.py                                 # Streamlit interactive dashboard (frontend + ML pickle backend)
├── ensemble_rainfall_sri_lanka.ipynb      # Full Jupyter Notebook (EDA, training, charts, and deep interpretation)
├── requirements.txt                       # Python dependencies for local run & Streamlit Community Cloud
├── data/
│   ├── fetch_weather.py                   # Script used to download 2015-2024 Sri Lankan data from Open-Meteo
│   ├── raw/
│   │   └── sri_lanka_weather_daily.csv    # 43,836 raw daily weather records across 12 Sri Lankan cities
│   └── processed/
│       ├── city_info.csv                  # City coordinates, elevation, climatic zone, annual rainfall stats
│       ├── monthly_climatology.csv        # Monthly average daily rainfall per city
│       └── test_predictions.csv           # 2023-2024 test set with predictions from all 17 models
├── models/
│   ├── stacking.pkl                       # Trained StackingRegressor (6 base models -> Ridge)
│   ├── random_forest.pkl                  # Trained RandomForestRegressor
│   ├── extra_trees.pkl                    # Trained ExtraTreesRegressor
│   ├── bagging.pkl                        # Trained BaggingRegressor
│   ├── xgboost.pkl                        # Trained XGBRegressor
│   ├── lightgbm.pkl                       # Trained LGBMRegressor
│   ├── catboost.pkl                       # Trained CatBoostRegressor
│   ├── gradient_boosting.pkl              # Trained GradientBoostingRegressor
│   ├── ann.pkl                            # Trained StandardScaler + MLPRegressor pipeline
│   ├── arma_params.pkl                    # Fitted ARMA(2,1) parameters for all 12 cities
│   ├── arma_rf_residual.pkl               # Trained Random Forest on ARMA(2,1) residuals
│   ├── linear_regression.pkl              # Trained Linear Regression baseline
│   └── feature_config.json                # Feature lists used by the models
└── results/
    ├── metrics.csv                        # Full evaluation metrics table
    ├── rmse_by_zone.csv                   # RMSE broken down by Wet / Intermediate / Dry zone
    ├── rmse_by_season.csv                 # RMSE broken down by Sri Lankan monsoon seasons
    ├── feature_importance.csv             # RF, XGBoost, and Stacking permutation importances
    ├── stacking_weights.csv               # Learned Ridge meta-learner weights
    ├── rf_trees_curve.csv                 # Bagging variance reduction curve (1 to 200 trees)
    └── xgb_rounds_curve.csv               # Boosting train vs. test RMSE curve (1 to 1,500 rounds)
```

---

## 4. Live Hosted Dashboard & Local Execution

### Live Dashboard
- **Hosted Application URL:** https://automataassignmentml-dgz85rykfhwevvlamnghgz.streamlit.app/

### How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py
```
