"""
============================================================
  AIR QUALITY INDEX PREDICTION USING MACHINE LEARNING
  College Project — main.py
  Run: python main.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('output', exist_ok=True)

# ── colours ──────────────────────────────────────────────
COLORS = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0']
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("  STEP 1: LOADING DATASET")
print("="*60)

df = pd.read_csv('aqi_dataset.csv')
print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")
print(df.describe().round(2).to_string())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nMissing values:\n", df.isnull().sum())
print("\nAQI Statistics:")
print(f"  Mean   : {df['AQI'].mean():.2f}")
print(f"  Median : {df['AQI'].median():.2f}")
print(f"  Std Dev: {df['AQI'].std():.2f}")
print(f"  Min    : {df['AQI'].min():.2f}")
print(f"  Max    : {df['AQI'].max():.2f}")

def aqi_category(aqi):
    if aqi <= 50:   return 'Good'
    elif aqi <= 100: return 'Moderate'
    elif aqi <= 150: return 'Unhealthy (Sensitive)'
    elif aqi <= 200: return 'Unhealthy'
    elif aqi <= 300: return 'Very Unhealthy'
    else:            return 'Hazardous'

df['AQI_Category'] = df['AQI'].apply(aqi_category)
print("\nAQI Category Distribution:")
print(df['AQI_Category'].value_counts().to_string())

# ── Figure 1: AQI distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('AQI Distribution Analysis', fontsize=14, fontweight='bold')

axes[0].hist(df['AQI'], bins=30, color='#2196F3', edgecolor='white', alpha=0.85)
axes[0].set_xlabel('AQI Value'); axes[0].set_ylabel('Frequency')
axes[0].set_title('AQI Histogram')
axes[0].axvline(df['AQI'].mean(), color='red', linestyle='--', label=f"Mean={df['AQI'].mean():.1f}")
axes[0].legend()

cat_counts = df['AQI_Category'].value_counts()
cat_order  = ['Good','Moderate','Unhealthy (Sensitive)','Unhealthy','Very Unhealthy','Hazardous']
cat_colors = ['#4CAF50','#FFEB3B','#FF9800','#F44336','#9C27B0','#B71C1C']
present    = [(c, col) for c, col in zip(cat_order, cat_colors) if c in cat_counts]
cats, cols = zip(*present)
axes[1].bar([c.replace('(','(').replace(' Sensitive)','') for c in cats],
            [cat_counts[c] for c in cats], color=cols, edgecolor='white')
axes[1].set_xlabel('Category'); axes[1].set_ylabel('Count')
axes[1].set_title('AQI Category Distribution')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('output/fig1_aqi_distribution.png', bbox_inches='tight')
plt.close()
print("\n[Saved] output/fig1_aqi_distribution.png")

# ── Figure 2: Correlation heatmap ─────────────────────────
numeric_cols = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','AQI']
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig2_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig2_correlation_heatmap.png")

# ── Figure 3: Pollutant box plots ─────────────────────────
pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
fig, axes  = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Pollutant Concentration Distribution', fontsize=14, fontweight='bold')
for ax, col, color in zip(axes.flatten(), pollutants, COLORS*2):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(col); ax.set_ylabel('Concentration')
plt.tight_layout()
plt.savefig('output/fig3_pollutant_boxplots.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig3_pollutant_boxplots.png")

# ── Figure 4: AQI by city and season ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('AQI by City and Season', fontsize=14, fontweight='bold')
city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
axes[0].bar(city_aqi.index, city_aqi.values, color=COLORS[0], alpha=0.85, edgecolor='white')
axes[0].set_xlabel('City'); axes[0].set_ylabel('Average AQI')
axes[0].set_title('Average AQI by City'); plt.sca(axes[0]); plt.xticks(rotation=30, ha='right')

season_aqi = df.groupby('Season')['AQI'].mean().sort_values(ascending=False)
axes[1].bar(season_aqi.index, season_aqi.values, color=COLORS[1], alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Season'); axes[1].set_ylabel('Average AQI')
axes[1].set_title('Average AQI by Season')
plt.tight_layout()
plt.savefig('output/fig4_aqi_by_city_season.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig4_aqi_by_city_season.png")

# ============================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("  STEP 3: PREPROCESSING")
print("="*60)

le_city   = LabelEncoder()
le_season = LabelEncoder()
df['City_enc']   = le_city.fit_transform(df['City'])
df['Season_enc'] = le_season.fit_transform(df['Season'])

feature_cols = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','City_enc','Season_enc']
X = df[feature_cols]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "="*60)
print("  STEP 4: TRAINING MODELS")
print("="*60)

models = {
    'Linear Regression'     : LinearRegression(),
    'Decision Tree'         : DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest'         : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'     : GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR'                   : SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
}

results = {}
print(f"\n{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("-"*60)

for name, model in models.items():
    # SVR needs scaled features
    if name == 'SVR':
        model.fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

    results[name] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape,
                     'model': model, 'y_pred': y_pred}
    print(f"{name:<25} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f} {mape:>8.2f}")

# Best model
best_name = max(results, key=lambda k: results[k]['r2'])
print(f"\nBest Model: {best_name}  (R² = {results[best_name]['r2']:.4f})")

# ── Figure 5: Model comparison bar chart ──────────────────
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R²':    [results[m]['r2']   for m in results],
    'MAE':   [results[m]['mae']  for m in results],
    'RMSE':  [results[m]['rmse'] for m in results],
}).set_index('Model')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
for ax, col, color, title in zip(axes,
        ['R²','MAE','RMSE'],
        ['#4CAF50','#FF9800','#E91E63'],
        ['R² Score (higher = better)','MAE (lower = better)','RMSE (lower = better)']):
    bars = ax.bar(metrics_df.index, metrics_df[col], color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title); ax.set_ylabel(col)
    ax.set_xticklabels(metrics_df.index, rotation=30, ha='right', fontsize=9)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('output/fig5_model_comparison.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig5_model_comparison.png")

# ── Figure 6: Actual vs Predicted (best model) ────────────
best_pred = results[best_name]['y_pred']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{best_name}: Actual vs Predicted AQI', fontsize=14, fontweight='bold')

axes[0].scatter(y_test, best_pred, alpha=0.5, color='#2196F3', s=20)
mn, mx = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
axes[0].plot([mn,mx],[mn,mx],'r--', label='Perfect prediction')
axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI')
axes[0].set_title('Scatter: Actual vs Predicted'); axes[0].legend()

residuals = y_test.values - best_pred
axes[1].hist(residuals, bins=30, color='#9C27B0', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--', label='Zero error')
axes[1].set_xlabel('Residual (Actual − Predicted)'); axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution'); axes[1].legend()
plt.tight_layout()
plt.savefig('output/fig6_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig6_actual_vs_predicted.png")

# ── Figure 7: Feature importance (Random Forest) ──────────
rf_model = results['Random Forest']['model']
feat_imp  = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
colors_feat = ['#E91E63' if v > feat_imp.median() else '#90CAF9' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax, color=colors_feat, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, v in enumerate(feat_imp.values):
    ax.text(v+0.001, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/fig7_feature_importance.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig7_feature_importance.png")

# ── Figure 8: Cross-validation scores ─────────────────────
cv_models = {k: v['model'] for k, v in results.items() if k != 'SVR'}
cv_scores  = {}
print("\nCross-Validation (5-fold R²):")
for name, model in cv_models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:<25} mean={scores.mean():.4f}  std={scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
positions = range(len(cv_scores))
for i, (name, scores) in enumerate(cv_scores.items()):
    ax.boxplot(scores, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=COLORS[i % len(COLORS)], alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
ax.set_xticks(list(positions))
ax.set_xticklabels(list(cv_scores.keys()), rotation=20, ha='right')
ax.set_ylabel('R² Score'); ax.set_title('5-Fold Cross-Validation R² Scores', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig8_cross_validation.png', bbox_inches='tight')
plt.close()
print("[Saved] output/fig8_cross_validation.png")

# ============================================================
# 5. SAVE RESULTS CSV
# ============================================================
results_df = pd.DataFrame([
    {'Model': name, 'R2': round(v['r2'],4), 'MAE': round(v['mae'],2),
     'RMSE': round(v['rmse'],2), 'MAPE%': round(v['mape'],2)}
    for name, v in results.items()
])
results_df.to_csv('output/model_results.csv', index=False)
print("\n[Saved] output/model_results.csv")

# ============================================================
# 6. PREDICT NEW SAMPLE
# ============================================================
print("\n" + "="*60)
print("  STEP 5: PREDICT A NEW SAMPLE")
print("="*60)

sample = pd.DataFrame([{
    'PM2.5': 85, 'PM10': 150, 'NO2': 60, 'SO2': 20,
    'CO': 2.5, 'O3': 55, 'Temperature': 32, 'Humidity': 70,
    'City_enc': 0, 'Season_enc': 2
}])
rf = results['Random Forest']['model']
pred_aqi = rf.predict(sample)[0]
print(f"\nInput: PM2.5=85, PM10=150, NO2=60, SO2=20, CO=2.5, O3=55, Temp=32, Humidity=70")
print(f"Predicted AQI : {pred_aqi:.1f}  →  {aqi_category(pred_aqi)}")

# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print("  PROJECT COMPLETE!")
print("  All figures saved in the 'output/' folder.")
print("="*60 + "\n")
