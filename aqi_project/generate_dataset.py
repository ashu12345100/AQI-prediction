"""
generate_dataset.py
Creates a realistic synthetic AQI dataset and saves it as aqi_dataset.csv
Run this FIRST before running main.py
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

n = 1000  # number of records

# Simulate pollutant readings (realistic ranges)
PM25 = np.random.exponential(scale=40, size=n).clip(1, 300)
PM10 = PM25 * np.random.uniform(1.5, 2.5, n) + np.random.normal(10, 5, n)
PM10 = PM10.clip(5, 500)
NO2  = np.random.exponential(scale=25, size=n).clip(1, 200)
SO2  = np.random.exponential(scale=10, size=n).clip(0.5, 100)
CO   = np.random.exponential(scale=1.0, size=n).clip(0.1, 10)
O3   = np.random.normal(50, 25, size=n).clip(5, 200)

# Temperature and Humidity (affect AQI)
Temperature = np.random.normal(28, 8, size=n).clip(5, 48)
Humidity    = np.random.normal(60, 20, size=n).clip(10, 100)

# Compute a realistic AQI from pollutants (EPA-style formula simplified)
def sub_index(C, breakpoints):
    """Linear interpolation between AQI breakpoints."""
    for (C_low, C_high, I_low, I_high) in breakpoints:
        if C_low <= C <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    return 500

pm25_bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)]
pm10_bp = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
           (255,354,151,200),(355,424,201,300),(425,604,301,500)]
no2_bp  = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
           (361,649,151,200),(650,1249,201,300),(1250,2049,301,500)]

AQI = []
for i in range(n):
    si_pm25 = sub_index(PM25[i], pm25_bp)
    si_pm10 = sub_index(PM10[i], pm10_bp)
    si_no2  = sub_index(NO2[i],  no2_bp)
    # Weighted combination + small noise
    aqi_val = max(si_pm25, si_pm10) * 0.6 + si_no2 * 0.2 + \
              (CO[i]/10)*50*0.1 + (O3[i]/200)*150*0.1
    aqi_val += np.random.normal(0, 5)
    AQI.append(round(min(max(aqi_val, 0), 500), 1))

# City and Season labels for variety
cities  = ['Delhi','Mumbai','Pune','Chennai','Kolkata','Hyderabad','Ahmedabad','Bengaluru']
seasons = ['Winter','Summer','Monsoon','Post-Monsoon']

df = pd.DataFrame({
    'City':        [random.choice(cities)  for _ in range(n)],
    'Season':      [random.choice(seasons) for _ in range(n)],
    'PM2.5':       PM25.round(2),
    'PM10':        PM10.round(2),
    'NO2':         NO2.round(2),
    'SO2':         SO2.round(2),
    'CO':          CO.round(3),
    'O3':          O3.round(2),
    'Temperature': Temperature.round(1),
    'Humidity':    Humidity.round(1),
    'AQI':         AQI
})

df.to_csv('aqi_dataset.csv', index=False)
print(f"Dataset created: aqi_dataset.csv  ({len(df)} rows, {len(df.columns)} columns)")
print(df.head())
