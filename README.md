# 🌿 Air Quality Index Prediction Using Machine Learning

> College Project | Python | Scikit-learn | Streamlit | Machine Learning

---

## 📌 Project Overview

This project predicts the **Air Quality Index (AQI)** of a location based on pollutant
concentrations and weather data using **5 Machine Learning algorithms**.

AQI is a number used by government agencies to communicate how polluted the air is.
Higher AQI = more pollution = greater health risk.

---

## 📁 Project Structure

```
aqi_project/
│
├── generate_dataset.py   → Creates the dataset (run this FIRST)
├── main.py               → Trains all ML models and saves graphs
├── app.py                → Interactive Streamlit web application
├── aqi_dataset.csv       → Generated dataset (1000 rows, 11 columns)
├── requirements.txt      → Required Python libraries
├── README.md             → This file
│
└── output/               → Auto-created after running main.py
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

## ▶️ How to Run

### Step 1 — Install Python
Download from: https://www.python.org/downloads/
> ⚠️ Tick **"Add Python to PATH"** during installation!

### Step 2 — Install Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### Step 3 — Generate Dataset
```bash
python generate_dataset.py
```

### Step 4 — Run ML Pipeline
```bash
python main.py
```

### Step 5 — Launch Web App
```bash
python -m streamlit run app.py
```
Opens at: `http://localhost:8501`

---

## 📊 Dataset Description

| Feature | Unit | Description |
|---|---|---|
| PM2.5 | µg/m³ | Fine particles — most harmful to lungs |
| PM10 | µg/m³ | Coarser particles, affects breathing |
| NO2 | ppb | Nitrogen dioxide — mainly from vehicles |
| SO2 | ppb | Sulphur dioxide — from industrial sources |
| CO | ppm | Carbon monoxide — toxic at high levels |
| O3 | ppb | Ground-level ozone — lung irritant |
| Temperature | °C | Affects pollutant formation |
| Humidity | % | Affects particle dispersion |
| City | — | Indian city name (8 cities) |
| Season | — | Season of measurement (4 seasons) |
| **AQI** | — | **Target variable to predict** |

- **Total Records:** 1000
- **Cities:** Delhi, Mumbai, Pune, Chennai, Kolkata, Hyderabad, Ahmedabad, Bengaluru
- **Seasons:** Winter, Summer, Monsoon, Post-Monsoon

---

## 🤖 Machine Learning Algorithms Used

### 1. Linear Regression — R² = 0.65
Finds a straight-line relationship between inputs and AQI.
Simple but limited for non-linear AQI patterns.

### 2. Decision Tree — R² = 0.83
Makes predictions using a tree of if-else decisions.
Example: *If PM2.5 > 50 AND NO2 > 40 → AQI = 120*

### 3. Random Forest — R² = 0.84
Builds 100 Decision Trees and averages their predictions.
More stable and accurate than a single tree.

### 4. ✅ Gradient Boosting — R² = 0.91 (BEST)
Builds trees one by one, each fixing the previous one's mistakes.
Best accuracy in this project.

### 5. SVR (Support Vector Regression) — R² = 0.72
Uses mathematical kernel functions to find the optimal prediction boundary.

---

## 📈 Model Results

| Model | R² Score | MAE | RMSE | MAPE% |
|---|---|---|---|---|
| Linear Regression | 0.6523 | 11.65 | 22.81 | 39.02 |
| Decision Tree | 0.8270 | 7.53 | 16.09 | 18.09 |
| Random Forest | 0.8427 | 8.00 | 15.34 | 18.42 |
| **Gradient Boosting** | **0.9091** | **6.73** | **11.66** | **17.87** |
| SVR | 0.7185 | 7.77 | 20.53 | 19.19 |

> ✅ **Best Model: Gradient Boosting** — R² = 0.91 means the model correctly explains **91% of AQI variation**

---

## 🎨 Output Graphs

| Graph | Description |
|---|---|
| fig1_aqi_distribution | Histogram and category bar chart of AQI values |
| fig2_correlation_heatmap | Correlation between all features and AQI |
| fig3_pollutant_boxplots | Range and outliers of each pollutant |
| fig4_aqi_by_city_season | AQI comparison across cities and seasons |
| fig5_model_comparison | Side-by-side comparison of all 5 models |
| fig6_actual_vs_predicted | Scatter plot of predictions vs real values |
| fig7_feature_importance | Which pollutants matter most (Random Forest) |
| fig8_cross_validation | 5-fold cross-validation stability scores |

---

## 🌐 Streamlit Web App

The `app.py` launches an interactive dashboard with 3 tabs:

- **Predict AQI** — Move sliders for each pollutant → get real-time AQI prediction with color-coded health category
- **Data Analysis** — View dataset, AQI histogram, correlation heatmap, city comparison
- **Model Results** — Compare all models, feature importance, actual vs predicted charts

---

## 🟢 AQI Scale Reference

| AQI Range | Category | Health Impact |
|---|---|---|
| 0 – 50 | 🟢 Good | No risk |
| 51 – 100 | 🟡 Moderate | Minor concern for sensitive groups |
| 101 – 150 | 🟠 Unhealthy (Sensitive) | Sensitive groups at risk |
| 151 – 200 | 🔴 Unhealthy | Everyone may be affected |
| 201 – 300 | 🟣 Very Unhealthy | Health alert for all |
| 301 – 500 | 🔴 Hazardous | Emergency conditions |

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.0.0
```

---

## 📚 References

1. US EPA AQI Calculation — https://www.airnow.gov/
2. CPCB India Air Quality Standards — https://cpcb.nic.in/
3. Scikit-learn Documentation — https://scikit-learn.org/
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
5. Friedman, J. (2001). Gradient Boosting Machine. *Annals of Statistics*.
6. Streamlit Documentation — https://docs.streamlit.io/

---

*Project submitted for academic evaluation. Dataset synthetically generated based on realistic Indian city air quality patterns.*
