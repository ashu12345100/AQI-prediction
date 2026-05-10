# Air Quality Index Prediction Using Machine Learning
### College Project — Complete Beginner Guide

---

## What This Project Does
This project predicts the **Air Quality Index (AQI)** of a city based on
pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3) and weather data
(Temperature, Humidity) using five different Machine Learning algorithms.

---

## Project Structure
```
aqi_project/
│
├── generate_dataset.py   ← STEP 1: Creates the dataset
├── main.py               ← STEP 2: Trains models & makes graphs
├── requirements.txt      ← List of required Python libraries
├── README.md             ← This file
│
├── aqi_dataset.csv       ← Created after running generate_dataset.py
│
└── output/               ← Created after running main.py
    ├── fig1_aqi_distribution.png
    ├── fig2_correlation_heatmap.png
    ├── fig3_pollutant_boxplots.png
    ├── fig4_aqi_by_city_season.png
    ├── fig5_model_comparison.png
    ├── fig6_actual_vs_predicted.png
    ├── fig7_feature_importance.png
    ├── fig8_cross_validation.png
    └── model_results.csv
```

---

## How to Run (Step by Step)

### Step 1 — Install Python
Download Python 3.9 or later from: https://www.python.org/downloads/
During installation, tick "Add Python to PATH".

### Step 2 — Install Required Libraries
Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 3 — Run the Dataset Generator
```
python generate_dataset.py
```
This creates `aqi_dataset.csv` with 1000 records.

### Step 4 — Run the Main Project
```
python main.py
```
This trains all models and saves 8 graphs + results in the `output/` folder.

---

## Machine Learning Models Used

| Model | What It Does |
|---|---|
| Linear Regression | Finds a straight-line relationship between inputs and AQI |
| Decision Tree | Makes a tree of if/else decisions to predict AQI |
| Random Forest | Combines 100 decision trees for better accuracy |
| Gradient Boosting | Builds trees one by one, each fixing the previous one's errors |
| SVR | Finds the best boundary using a mathematical kernel trick |

---

## Features (Input Variables)

| Feature | Unit | Why Important |
|---|---|---|
| PM2.5 | µg/m³ | Fine particles — most harmful to lungs |
| PM10 | µg/m³ | Coarser particles, affects breathing |
| NO2 | ppb | From vehicles, causes respiratory issues |
| SO2 | ppb | From industries, causes acid rain |
| CO | ppm | Carbon monoxide, toxic at high levels |
| O3 | ppb | Ground-level ozone, lung irritant |
| Temperature | °C | Affects pollutant formation |
| Humidity | % | Affects particle dispersion |

---

## AQI Scale

| AQI Range | Category | Health Impact |
|---|---|---|
| 0 – 50 | Good | No risk |
| 51 – 100 | Moderate | Acceptable, minor concern for sensitive groups |
| 101 – 150 | Unhealthy (Sensitive) | Sensitive groups at risk |
| 151 – 200 | Unhealthy | Everyone may be affected |
| 201 – 300 | Very Unhealthy | Health alert for all |
| 301 – 500 | Hazardous | Emergency conditions |

---

## Results Summary

Best model: **Gradient Boosting** (R² = 0.91)

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.65 | 11.65 | 22.81 |
| Decision Tree | 0.83 | 7.53 | 16.09 |
| Random Forest | 0.84 | 8.00 | 15.34 |
| Gradient Boosting | **0.91** | **6.73** | **11.66** |
| SVR | 0.72 | 7.77 | 20.53 |

---

## Output Graphs Explained

1. **fig1_aqi_distribution** — Shows how AQI values are spread across the dataset
2. **fig2_correlation_heatmap** — Shows which pollutants are most related to AQI
3. **fig3_pollutant_boxplots** — Shows the range and outliers of each pollutant
4. **fig4_aqi_by_city_season** — Compares AQI across Indian cities and seasons
5. **fig5_model_comparison** — Compares all 5 models side by side
6. **fig6_actual_vs_predicted** — Shows how close predictions were to real values
7. **fig7_feature_importance** — Shows which features matter most (Random Forest)
8. **fig8_cross_validation** — Validates model stability using 5-fold testing

---

## References
1. US EPA AQI Calculation Formula — https://www.airnow.gov/
2. CPCB India Air Quality Standards — https://cpcb.nic.in/
3. Scikit-learn Documentation — https://scikit-learn.org/
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
5. Friedman, J. (2001). Greedy Function Approximation: Gradient Boosting Machine.

---

*Project for academic submission. Dataset is synthetically generated
based on realistic Indian city air quality patterns.*
