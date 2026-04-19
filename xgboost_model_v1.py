"""
XGBoost Model - Prediksi Okupansi Hotel
======================================
Pipeline:
1. Load & Preprocessing Data
2. Agregasi Time Series Bulanan (Check-In)
3. Feature Engineering (Lags, Time features)
4. Split Data Train/Test
5. Training Model XGBoost
6. Evaluasi Model (MAE, RMSE, MAPE)
7. Forecasting 12 Bulan ke Depan
8. Simpan Hasil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# Buat folder output
OUTPUT_DIR = "output_xgboost_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & PREPROCESSING DATA
# ============================================================
print("=" * 60)
print("1. LOAD & PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv("data-okupansi-hotel-new.csv")
print(f"Total records: {len(df)}")

# drop column
print('Droping column yang tidak digunakan')
df = df.drop(columns=['No', 'Jenis Kelamin', 'Warga Negara', 'Pekerjaan',
 'Nomor Kamar', 'Tipe Kamar', 'Telp.',
 'Alamat', 'Pax', 'Jlh Malam', 'Nama'])

# Convert Checkin dan Checkout ke datetime
print('Convert Checkin dan Checkout ke datetime')
df['Check In'] = pd.to_datetime(df['Check In'], format='%d-%m-%Y')
df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d-%m-%Y')

# Potong data dari awal hingga 2025-12-31 (berdasarkan Check In DAN Check Out)
df = df[(df['Check In'] <= '2025-12-31') & (df['Check Out'] <= '2025-12-31')]

# ============================================================
# 2. AGREGASI TIME SERIES BULANAN
# ============================================================
print("\n" + "=" * 60)
print("2. AGREGASI TIME SERIES BULANAN")
print("=" * 60)

df['Bulan'] = df['Check In'].dt.to_period('M')
monthly_checkin = df.groupby('Bulan').size()
monthly_checkin.index = monthly_checkin.index.to_timestamp()
monthly_checkin = monthly_checkin.asfreq('MS')
monthly_checkin.name = 'Jumlah_CheckIn'

df_ts = pd.DataFrame(monthly_checkin)
print(f"Jumlah data bulanan: {len(df_ts)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING (Lags & Time Features)")
print("=" * 60)

def create_features(data, target_col):
    df_feat = data.copy()
    # Membuat lag features
    for i in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    
    # Time features
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['quarter'] = df_feat.index.quarter
    
    # Hapus row dengan NaN akibat proses shift/lag
    df_feat.dropna(inplace=True)
    return df_feat

df_features = create_features(df_ts, 'Jumlah_CheckIn')
print(f"Sisa data setelah feature engineering: {len(df_features)} bulan")

# ============================================================
# 4. SPLIT DATA: TRAIN & TEST
# ============================================================
print("\n" + "=" * 60)
print("4. SPLIT DATA: TRAIN & TEST")
print("=" * 60)

n_test = 6  # 6 bulan terakhir untuk test
train = df_features.iloc[:-n_test]
test = df_features.iloc[-n_test:]

print(f"Train set: {len(train)} bulan")
print(f"Test set: {len(test)} bulan")

X_train = train.drop('Jumlah_CheckIn', axis=1)
y_train = train['Jumlah_CheckIn']
X_test = test.drop('Jumlah_CheckIn', axis=1)
y_test = test['Jumlah_CheckIn']

# ============================================================
# 5. TRAINING MODEL XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("5. TRAINING MODEL XGBOOST")
print("=" * 60)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)
print("Model XGBoost berhasil dilatih!")

# Membuktikan pentingnya setiap fitur (Feature Importance)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ============================================================
# 6. EVALUASI MODEL
# ============================================================
print("\n" + "=" * 60)
print("6. EVALUASI MODEL")
print("=" * 60)

predictions = model.predict(X_test)
predictions = np.round(predictions) # Jumlah check-in adalah bilangan bulat

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"Metrik Evaluasi (Test Set - {n_test} bulan):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAPE : {mape:.2f}%")

comparison = pd.DataFrame({
    'Bulan': y_test.index.strftime('%B %Y'),
    'Aktual': y_test.values,
    'Prediksi_XGBoost': predictions,
    'Error': (y_test.values - predictions).round(0),
    'Error%': ((np.abs(y_test.values - predictions) / y_test.values) * 100).round(2)
})
print("\nPerbandingan Aktual vs Prediksi:")
print(comparison.to_string(index=False))

# Plot Aktual vs Prediksi pada Data Test
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ts.index, df_ts['Jumlah_CheckIn'], marker='o', label='Aktual', color='#2196F3')
ax.plot(y_test.index, predictions, marker='^', label='Prediksi (Test)', color='#F44336', linestyle='--', linewidth=2)
ax.axvline(x=y_test.index[0], color='gray', linestyle='--', alpha=0.5, label='Split Point')

textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=props)

ax.set_title('XGBoost - Aktual vs Prediksi', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/01_aktual_vs_prediksi.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. FORECASTING 12 BULAN KE DEPAN
# ============================================================
print("\n" + "=" * 60)
print("7. FORECASTING 12 BULAN KE DEPAN")
print("=" * 60)

# Re-train model menggunakan keseluruhan fitur
all_X = df_features.drop('Jumlah_CheckIn', axis=1)
all_y = df_features['Jumlah_CheckIn']

final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
final_model.fit(all_X, all_y)

future_months = 12
last_date = df_ts.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

forecasts = []
current_history = df_ts['Jumlah_CheckIn'].copy()

# Forecasting secara iteratif (Rolling window karena butuh lag dari model ke-t untuk input ke-t+1)
for date in future_dates:
    features = {
        'lag_1': current_history.iloc[-1] if len(current_history) >= 1 else 0,
        'lag_2': current_history.iloc[-2] if len(current_history) >= 2 else 0,
        'lag_3': current_history.iloc[-3] if len(current_history) >= 3 else 0,
        'lag_6': current_history.iloc[-6] if len(current_history) >= 6 else 0,
        'lag_12': current_history.iloc[-12] if len(current_history) >= 12 else 0,
        'month': date.month,
        'year': date.year,
        'quarter': date.quarter
    }
    
    # Pastikan urutan kolom sesuai dengan all_X
    X_fut = pd.DataFrame([features], index=[date])[all_X.columns]
    y_pred = final_model.predict(X_fut)[0]
    y_pred = max(0, round(y_pred)) # Hasil tidak boleh kurang dari 0
    
    forecasts.append(y_pred)
    current_history.loc[date] = y_pred

forecast_df = pd.DataFrame({
    'Bulan': future_dates.strftime('%B %Y'),
    'Prediksi_CheckIn': forecasts
})

print(f"\nPrediksi Okupansi Hotel (Jumlah Check-In) - {future_months} Bulan ke Depan:")
print("-" * 55)
print(forecast_df.to_string(index=False))

# Visualisasi Forecast
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ts.index, df_ts['Jumlah_CheckIn'], marker='o', label='Data Historis', color='#2196F3', linewidth=1.5)
ax.plot(future_dates, forecasts, marker='D', label='Prediksi XGBoost', color='#FF5722', linestyle='--', linewidth=2)
ax.axvline(x=df_ts.index[-1], color='gray', linestyle='--', alpha=0.5)

ax.set_title('XGBoost - Forecasting Okupansi Hotel 12 Bulan Kedepan', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.savefig(f"{OUTPUT_DIR}/02_forecast_future.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 8. SIMPAN HASIL
# ============================================================
print("\n" + "=" * 60)
print("8. SIMPAN HASIL")
print("=" * 60)

comparison.to_csv(f"{OUTPUT_DIR}/aktual_vs_prediksi.csv", index=False)
forecast_df.to_csv(f"{OUTPUT_DIR}/forecast.csv", index=False)

# Metrik Evaluasi
eval_df = pd.DataFrame({
    'Model': ['XGBoost'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'Train_Size': [len(train)],
    'Test_Size': [len(test)]
})
eval_df.to_csv(f"{OUTPUT_DIR}/evaluasi_model.csv", index=False)

print(f"✓ Semua hasil evaluasi dan plot tersimpan di dalam direktori '{OUTPUT_DIR}/'")
print("Misi Selesai!")
