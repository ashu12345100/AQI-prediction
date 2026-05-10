"""
app.py  —  AQI Prediction Streamlit Web App
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.linear_model    import LinearRegression
from sklearn.svm             import SVR
from sklearn.metrics         import r2_score, mean_absolute_error, mean_squared_error

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Air Quality Index Prediction")
st.markdown("**College Project — Machine Learning**")
st.markdown("---")

# ── Load & train (cached so it runs only once) ───────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("aqi_dataset.csv")

    le_city   = LabelEncoder()
    le_season = LabelEncoder()
    df['City_enc']   = le_city.fit_transform(df['City'])
    df['Season_enc'] = le_season.fit_transform(df['Season'])

    features = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','City_enc','Season_enc']
    X = df[features]
    y = df['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression' : LinearRegression(),
        'Decision Tree'     : DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model' : model,
            'r2'    : round(r2_score(y_test, y_pred), 4),
            'mae'   : round(mean_absolute_error(y_test, y_pred), 2),
            'rmse'  : round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'y_pred': y_pred,
        }
    return df, results, X_test, y_test, le_city, le_season

df, results, X_test, y_test, le_city, le_season = load_and_train()

# ── AQI helper ────────────────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:   return "Good",                  "#4CAF50"
    elif aqi <= 100: return "Moderate",              "#FFEB3B"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "#FF9800"
    elif aqi <= 200: return "Unhealthy",             "#F44336"
    elif aqi <= 300: return "Very Unhealthy",        "#9C27B0"
    else:            return "Hazardous",             "#B71C1C"

# ════════════════════════════════════════════════════════
# SIDEBAR — user inputs
# ════════════════════════════════════════════════════════
st.sidebar.header("🔬 Enter Pollutant Values")

pm25 = st.sidebar.slider("PM2.5 (µg/m³)",  0.0, 300.0, 45.0, 1.0)
pm10 = st.sidebar.slider("PM10  (µg/m³)",  0.0, 500.0, 80.0, 1.0)
no2  = st.sidebar.slider("NO₂   (ppb)",    0.0, 200.0, 35.0, 1.0)
so2  = st.sidebar.slider("SO₂   (ppb)",    0.0, 100.0, 10.0, 0.5)
co   = st.sidebar.slider("CO    (ppm)",    0.0,  10.0,  1.2, 0.1)
o3   = st.sidebar.slider("O₃    (ppb)",    0.0, 200.0, 40.0, 1.0)
temp = st.sidebar.slider("Temperature (°C)", 5.0, 48.0, 30.0, 0.5)
hum  = st.sidebar.slider("Humidity (%)",    10.0,100.0, 60.0, 1.0)

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    list(results.keys()),
    index=2
)

# ════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 Predict AQI", "📊 Data Analysis", "🤖 Model Results"])

# ── TAB 1: PREDICT ───────────────────────────────────────
with tab1:
    st.subheader("AQI Prediction")

    sample = pd.DataFrame([{
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2,
        'CO': co, 'O3': o3, 'Temperature': temp, 'Humidity': hum,
        'City_enc': 0, 'Season_enc': 0
    }])

    model = results[model_choice]['model']
    predicted_aqi = model.predict(sample)[0]
    category, color = aqi_category(predicted_aqi)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted AQI",  f"{predicted_aqi:.1f}")
    col2.metric("Category",       category)
    col3.metric("Model Used",     model_choice)

    st.markdown(f"""
    <div style='background-color:{color}22; border-left: 6px solid {color};
                padding: 16px; border-radius: 8px; margin-top: 12px;'>
        <h3 style='color:{color}; margin:0;'>{category}</h3>
        <p style='margin:6px 0 0; font-size:15px;'>AQI Value: <b>{predicted_aqi:.1f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### AQI Scale Reference")
    scale_data = {
        "Range"   : ["0–50","51–100","101–150","151–200","201–300","301–500"],
        "Category": ["Good","Moderate","Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"],
        "Health Impact": [
            "No risk","Minor concern for sensitive groups","Sensitive groups at risk",
            "Everyone may be affected","Health alert for all","Emergency conditions"
        ]
    }
    st.dataframe(pd.DataFrame(scale_data), use_container_width=True, hide_index=True)

# ── TAB 2: DATA ANALYSIS ─────────────────────────────────
with tab2:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Features",      "10")
    col3.metric("Avg AQI",       f"{df['AQI'].mean():.1f}")
    col4.metric("Max AQI",       f"{df['AQI'].max():.1f}")

    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### AQI Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['AQI'], bins=30, color='#2196F3', edgecolor='white', alpha=0.85)
        ax.axvline(df['AQI'].mean(), color='red', linestyle='--', label=f"Mean={df['AQI'].mean():.1f}")
        ax.set_xlabel('AQI'); ax.set_ylabel('Frequency')
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','AQI']
        corr = df[numeric].corr()
        sns.heatmap(corr, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    linewidths=0.5, annot_kws={'size': 7})
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("### Average AQI by City")
    city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(city_aqi.index, city_aqi.values, color='#2196F3', edgecolor='white', alpha=0.85)
    ax.set_xlabel('City'); ax.set_ylabel('Average AQI')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── TAB 3: MODEL RESULTS ─────────────────────────────────
with tab3:
    st.subheader("Model Performance Comparison")

    metrics_df = pd.DataFrame([
        {'Model': name, 'R² Score': v['r2'], 'MAE': v['mae'], 'RMSE': v['rmse']}
        for name, v in results.items()
    ]).sort_values('R² Score', ascending=False)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    best = metrics_df.iloc[0]
    st.success(f"✅ Best Model: **{best['Model']}** with R² = {best['R² Score']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### R² Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(metrics_df['Model'], metrics_df['R² Score'],
               color=['#4CAF50','#2196F3','#FF9800','#E91E63'], edgecolor='white', alpha=0.85)
        ax.set_ylabel('R² Score'); ax.set_ylim(0, 1)
        plt.xticks(rotation=20, ha='right', fontsize=9); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### Feature Importance (Random Forest)")
        rf = results['Random Forest']['model']
        features = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','City','Season']
        fi = pd.Series(rf.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        fi.plot(kind='barh', ax=ax, color='#2196F3', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Importance'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("### Actual vs Predicted AQI")
    best_name  = metrics_df.iloc[0]['Model']
    best_preds = results[best_name]['y_pred']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_test, best_preds, alpha=0.4, color='#2196F3', s=15)
    mn, mx = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
    ax.plot([mn,mx],[mn,mx],'r--', label='Perfect fit')
    ax.set_xlabel('Actual AQI'); ax.set_ylabel('Predicted AQI')
    ax.set_title(f'{best_name} — Actual vs Predicted')
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()
