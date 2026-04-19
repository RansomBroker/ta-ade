"""
Perbandingan Model: SARIMA vs XGBoost (V1 & V2)
=====================================
Kode ini melatih dan mengevaluasi 4 model: SARIMA v1, SARIMA v2, XGBoost v1, dan XGBoost v2,
kemudian menampilkan metrik evaluasi dan grafik perbandingan keempatnya secara komprehensif.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import calendar
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# Buat folder output khusus perbandingan
OUTPUT_DIR = "output_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================
df = pd.read_csv("data-okupansi-hotel-new.csv")

# drop column
df = df.drop(columns=['No', 'Jenis Kelamin', 'Warga Negara', 'Pekerjaan',
 'Nomor Kamar', 'Tipe Kamar', 'Telp.',
 'Alamat', 'Pax', 'Jlh Malam', 'Nama'])

# Convert Checkin dan Checkout ke datetime
df['Check In'] = pd.to_datetime(df['Check In'], format='%d-%m-%Y')
df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d-%m-%Y')

# Potong data dari awal hingga 2025-12-31 (berdasarkan Check In DAN Check Out)
df = df[(df['Check In'] <= '2025-12-31') & (df['Check Out'] <= '2025-12-31')]

df['Bulan'] = df['Check In'].dt.to_period('M')
monthly_checkin = df.groupby('Bulan').size()
monthly_checkin.index = monthly_checkin.index.to_timestamp()
monthly_checkin = monthly_checkin.asfreq('MS')
monthly_checkin.name = 'Jumlah_CheckIn'

print("Data berhasil dimuat. Total bulan historis:", len(monthly_checkin))

# 6 bulan terakhir sebagai test set
n_test = 6
train_ts = monthly_checkin[:-n_test]
test_ts = monthly_checkin[-n_test:]

# ============================================================
# 2. TRAINING SARIMA MODEL (V1 & V2)
# ============================================================
print("\n" + "="*50)
print("1. TRAINING MODEL SARIMA V1 & V2")
print("="*50)

# SARIMA V1 (Basic, No Forced Seasonality)
print("Mencari parameter SARIMA v1 (D=None)...")
auto_v1 = auto_arima(
    train_ts, seasonal=True, m=12, d=None, D=None,
    start_p=0, max_p=3, start_q=0, max_q=3, start_P=0, max_P=2, start_Q=0, max_Q=2,
    trace=False, error_action='ignore', suppress_warnings=True, stepwise=True,
    information_criterion='aic'
)
print(f"Optimal Parameter v1: {auto_v1.order} x {auto_v1.seasonal_order}")
sarima_v1 = SARIMAX(train_ts, order=auto_v1.order, seasonal_order=auto_v1.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
res_v1 = sarima_v1.fit(disp=False)
sarima_v1_pred = np.round(res_v1.get_forecast(steps=n_test).predicted_mean)

# SARIMA V2 (Forced Seasonality D=1)
print("\nMencari parameter SARIMA v2 (D=1)...")
auto_v2 = auto_arima(
    train_ts, seasonal=True, m=12, d=None, D=1,
    start_p=0, max_p=3, start_q=0, max_q=3, start_P=0, max_P=2, start_Q=0, max_Q=2,
    trace=False, error_action='ignore', suppress_warnings=True, stepwise=True,
    information_criterion='aic'
)
print(f"Optimal Parameter v2: {auto_v2.order} x {auto_v2.seasonal_order}")
sarima_v2 = SARIMAX(train_ts, order=auto_v2.order, seasonal_order=auto_v2.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
res_v2 = sarima_v2.fit(disp=False)
sarima_v2_pred = np.round(res_v2.get_forecast(steps=n_test).predicted_mean)


# ============================================================
# 3. TRAINING XGBOOST MODEL (V1 & V2)
# ============================================================
print("\n" + "="*50)
print("2. TRAINING MODEL XGBOOST V1 & V2")
print("="*50)
df_ts = pd.DataFrame(monthly_checkin)

def create_features_v1(data, target_col):
    df_feat = data.copy()
    for i in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['quarter'] = df_feat.index.quarter
    df_feat.dropna(inplace=True)
    return df_feat

def create_features_v2(data, target_col):
    df_feat = data.copy()
    for i in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
        
    df_feat['rolling_mean_3'] = df_feat[target_col].shift(1).rolling(window=3).mean()
    df_feat['rolling_std_3']  = df_feat[target_col].shift(1).rolling(window=3).std()
    
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['quarter'] = df_feat.index.quarter
    
    df_feat['is_high_season'] = df_feat['month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11, 12] else 0)
    df_feat['is_low_season']  = df_feat['month'].apply(lambda x: 1 if x in [4, 5] else 0)
    df_feat['days_in_month'] = df_feat.index.map(lambda d: calendar.monthrange(d.year, d.month)[1])
    
    def count_weekends(year, month):
        days = calendar.monthcalendar(year, month)
        return sum(1 for week in days if week[4] != 0) + sum(1 for week in days if week[5] != 0)
    df_feat['total_weekend_days'] = [count_weekends(d.year, d.month) for d in df_feat.index]
    
    df_feat.dropna(inplace=True)
    return df_feat

# == XGBoost V1 ==
df_features_v1 = create_features_v1(df_ts, 'Jumlah_CheckIn')
test_xgb_v1 = df_features_v1.loc[test_ts.index]
train_xgb_v1 = df_features_v1.loc[df_features_v1.index < test_ts.index[0]]

model_xgb_v1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model_xgb_v1.fit(train_xgb_v1.drop('Jumlah_CheckIn', axis=1), train_xgb_v1['Jumlah_CheckIn'])
xgb_v1_pred = np.round(model_xgb_v1.predict(test_xgb_v1.drop('Jumlah_CheckIn', axis=1)))
xgb_v1_pred = pd.Series(xgb_v1_pred, index=test_xgb_v1.index)
print("Model XGBoost v1 berhasil dilatih.")

# == XGBoost V2 (Log Transform & Advanced Features) ==
df_ts_v2 = df_ts.copy()
df_ts_v2['Jumlah_CheckIn_Log'] = np.log1p(df_ts_v2['Jumlah_CheckIn'])

df_features_v2 = create_features_v2(df_ts_v2, 'Jumlah_CheckIn_Log')
test_xgb_v2 = df_features_v2.loc[test_ts.index]
train_xgb_v2 = df_features_v2.loc[df_features_v2.index < test_ts.index[0]]

model_xgb_v2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
X_train_v2 = train_xgb_v2.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore')
y_train_v2 = train_xgb_v2['Jumlah_CheckIn_Log']
model_xgb_v2.fit(X_train_v2, y_train_v2)

X_test_v2 = test_xgb_v2.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore')
xgb_v2_pred_log = model_xgb_v2.predict(X_test_v2)
xgb_v2_pred = np.round(np.expm1(xgb_v2_pred_log))
xgb_v2_pred = pd.Series(xgb_v2_pred, index=test_xgb_v2.index)
print("Model XGBoost v2 berhasil dilatih.")

# ============================================================
# 4. EVALUASI DAN PERBANDINGAN
# ============================================================
print("\n" + "="*50)
print("3. EVALUASI DAN PERBANDINGAN")
print("="*50)

def evaluate(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

metrics = []
for name, preds in [("SARIMA v1", sarima_v1_pred), ("SARIMA v2", sarima_v2_pred), 
                    ("XGBoost v1", xgb_v1_pred), ("XGBoost v2", xgb_v2_pred)]:
    mae, rmse, mape = evaluate(test_ts.values, preds.values)
    metrics.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape})

comparison_metrics = pd.DataFrame(metrics)
print("\nRingkasan Metrik Evaluasi:")
print("-" * 50)
print(comparison_metrics.to_string(index=False))
print("-" * 50)

comparison_preds = pd.DataFrame({
    'Bulan': test_ts.index.strftime('%B %Y'),
    'Aktual': test_ts.values,
    'Pred_SARIMA_v1': sarima_v1_pred.values.astype(int),
    'Pred_SARIMA_v2': sarima_v2_pred.values.astype(int),
    'Pred_XGBoost_v1': xgb_v1_pred.values.astype(int),
    'Pred_XGBoost_v2': xgb_v2_pred.values.astype(int)
})

print("\nDetail Prediksi Test Set (6 Bulan Terakhir):")
print(comparison_preds.to_string(index=False))

# Plot Perbandingan
fig, ax = plt.subplots(figsize=(16, 7))

# Train / Test History
train_subset = train_ts[-12:]
ax.plot(train_subset.index, train_subset.values, marker='o', label='Aktual (Train)', color='grey', alpha=0.5)
ax.plot(test_ts.index, test_ts.values, marker='o', label='Aktual (Test)', color='black', linewidth=2, markersize=8)

# Prediksi V1
ax.plot(test_ts.index, sarima_v1_pred.values, marker='s', label='SARIMA v1', color='#2196F3', linestyle=':', alpha=0.6)
ax.plot(test_ts.index, xgb_v1_pred.values, marker='^', label='XGBoost v1', color='#FF9800', linestyle=':', alpha=0.6)

# Prediksi V2
ax.plot(test_ts.index, sarima_v2_pred.values, marker='D', label='SARIMA v2', color='#3F51B5', linestyle='-', linewidth=2)
ax.plot(test_ts.index, xgb_v2_pred.values, marker='*', label='XGBoost v2', color='#F44336', linestyle='-', linewidth=2, markersize=10)

ax.axvline(x=test_ts.index[0], color='black', linestyle='--', alpha=0.3)

# Title & Labels
ax.set_title('Perbandingan Kinerja Ekstrim: SARIMA (v1/v2) vs XGBoost (v1/v2)', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/perbandingan_semua_model.png", dpi=150, bbox_inches='tight')
plt.close()

# Simpan ke CSV
comparison_metrics.to_csv(f"{OUTPUT_DIR}/metrik_kombinasi_v1_v2.csv", index=False)
comparison_preds.to_csv(f"{OUTPUT_DIR}/prediksi_kombinasi_v1_v2.csv", index=False)

print(f"\n✓ Hasil komparasi 4 model telah tersimpan di direktori '{OUTPUT_DIR}/'")
print("Misi Selesai!")
