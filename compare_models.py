"""
Perbandingan Model: SARIMA vs CatBoost
=============================================
Kode ini melatih dan mengevaluasi 2 model: SARIMA dan CatBoost,
kemudian menampilkan metrik evaluasi dan grafik perbandingan keduanya.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from catboost import CatBoostRegressor
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
# 2. TRAINING SARIMA
# ============================================================
print("\n" + "="*50)
print("1. TRAINING MODEL SARIMA")
print("="*50)

# SARIMA (Forced Seasonality D=1)
print("Mencari parameter SARIMA (D=1)...")
auto_model = auto_arima(
    train_ts, seasonal=True, m=12, d=None, D=1,
    start_p=0, max_p=3, start_q=0, max_q=3, start_P=0, max_P=2, start_Q=0, max_Q=2,
    trace=False, error_action='ignore', suppress_warnings=True, stepwise=True,
    information_criterion='aic'
)
print(f"Optimal Parameter: {auto_model.order} x {auto_model.seasonal_order}")
sarima_model = SARIMAX(train_ts, order=auto_model.order, seasonal_order=auto_model.seasonal_order,
                    enforce_stationarity=True, enforce_invertibility=True)
res = sarima_model.fit(disp=False)
sarima_pred = np.round(res.get_forecast(steps=n_test).predicted_mean)
print("Model SARIMA berhasil dilatih.")

# ============================================================
# 3. TRAINING CATBOOST
# ============================================================
print("\n" + "="*50)
print("2. TRAINING MODEL CATBOOST")
print("="*50)

df_ts = pd.DataFrame(monthly_checkin)

def create_features(data, target_col):
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

# == CatBoost (Log Transform & Advanced Features) ==
df_ts_cb = df_ts.copy()
df_ts_cb['Jumlah_CheckIn_Log'] = np.log1p(df_ts_cb['Jumlah_CheckIn'])

df_features = create_features(df_ts_cb, 'Jumlah_CheckIn_Log')
test_cb = df_features.loc[test_ts.index]
train_cb = df_features.loc[df_features.index < test_ts.index[0]]

model_cb = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    random_seed=42,
    verbose=0
)
X_train_cb = train_cb.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore')
y_train_cb = train_cb['Jumlah_CheckIn_Log']
model_cb.fit(X_train_cb, y_train_cb)

X_test_cb = test_cb.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore')
cb_pred_log = model_cb.predict(X_test_cb)
cb_pred = np.round(np.expm1(cb_pred_log))
cb_pred = pd.Series(cb_pred, index=test_cb.index)
print("Model CatBoost berhasil dilatih.")

# ============================================================
# 4. EVALUASI DAN PERBANDINGAN
# ============================================================
print("\n" + "="*50)
print("3. EVALUASI DAN PERBANDINGAN")
print("="*50)

def evaluate(actual, pred):
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return round(mape, 2)

metrics = []
for name, preds in [("SARIMA", sarima_pred), ("CatBoost", cb_pred)]:
    mape = evaluate(test_ts.values, preds.values)
    metrics.append({'Model': name, 'MAPE (%)': mape})

comparison_metrics = pd.DataFrame(metrics)
print("\nRingkasan Metrik Evaluasi:")
print("-" * 50)
print(comparison_metrics.to_string(index=False))
print("-" * 50)

comparison_preds = pd.DataFrame({
    'Bulan': test_ts.index.strftime('%B %Y'),
    'Aktual': test_ts.values,
    'Pred_SARIMA': sarima_pred.values.astype(int),
    'Pred_CatBoost': cb_pred.values.astype(int)
})

print("\nDetail Prediksi Test Set (6 Bulan Terakhir):")
print(comparison_preds.to_string(index=False))

# Plot Perbandingan
fig, ax = plt.subplots(figsize=(16, 7))

# Train / Test History
train_subset = train_ts[-12:]
ax.plot(train_subset.index, train_subset.values, marker='o', label='Aktual (Train)', color='grey', alpha=0.5)
ax.plot(test_ts.index, test_ts.values, marker='o', label='Aktual (Test)', color='black', linewidth=2, markersize=8)

# Prediksi
ax.plot(test_ts.index, sarima_pred.values, marker='D', label='SARIMA', color='#3F51B5', linestyle='-', linewidth=2)
ax.plot(test_ts.index, cb_pred.values, marker='*', label='CatBoost', color='#F44336', linestyle='-', linewidth=2, markersize=10)

ax.axvline(x=test_ts.index[0], color='black', linestyle='--', alpha=0.3)

# Title & Labels
ax.set_title('Perbandingan Kinerja: SARIMA vs CatBoost', fontsize=14, fontweight='bold')
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
comparison_metrics.to_csv(f"{OUTPUT_DIR}/metrik_perbandingan.csv", index=False)
comparison_preds.to_csv(f"{OUTPUT_DIR}/prediksi_perbandingan.csv", index=False)

print(f"\n✓ Hasil komparasi 2 model telah tersimpan di direktori '{OUTPUT_DIR}/'")
print("Misi Selesai!")
