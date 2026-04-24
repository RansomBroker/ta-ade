"""
CatBoost Model - Prediksi Okupansi Hotel
======================================
Pipeline:
1. Load & Preprocessing Data
2. Agregasi Time Series Bulanan (Check-In)
3. Feature Engineering (Lags, Time features)
4. Split Data Train/Test
5. Training Model CatBoost
6. Evaluasi Model (MAPE)
7. Forecasting 12 Bulan ke Depan
8. Simpan Hasil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from catboost import CatBoostRegressor

import warnings
import os

warnings.filterwarnings('ignore')

# Buat folder output
OUTPUT_DIR = "output_catboost"
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
df_ts['Jumlah_CheckIn_Asli'] = df_ts['Jumlah_CheckIn'].copy()
df_ts['Jumlah_CheckIn'] = np.log1p(df_ts['Jumlah_CheckIn'])
print(f"Jumlah data bulanan: {len(df_ts)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING (Lags & Time Features)")
print("=" * 60)

import calendar

def create_features(data, target_col):
    df_feat = data.copy()
    # Membuat lag features
    for i in [1, 2, 3, 6, 12]:
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
        
    # Rolling features berjalan
    df_feat['rolling_mean_3'] = df_feat[target_col].shift(1).rolling(window=3).mean()
    df_feat['rolling_std_3']  = df_feat[target_col].shift(1).rolling(window=3).std()
    
    # Time features
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['quarter'] = df_feat.index.quarter
    
    # Kategori Musim (High / Low Season)
    df_feat['is_high_season'] = df_feat['month'].apply(lambda x: 1 if x in [6, 7, 8, 9, 10, 11, 12] else 0)
    df_feat['is_low_season']  = df_feat['month'].apply(lambda x: 1 if x in [4, 5] else 0)
    
    # Calendar Info
    df_feat['days_in_month'] = df_feat.index.map(lambda d: calendar.monthrange(d.year, d.month)[1])
    
    def count_weekends(year, month):
        days = calendar.monthcalendar(year, month)
        return sum(1 for week in days if week[4] != 0) + sum(1 for week in days if week[5] != 0)
        
    df_feat['total_weekend_days'] = [count_weekends(d.year, d.month) for d in df_feat.index]
    
    # Hapus row dengan NaN akibat proses shift/lag/rolling
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

X_train = train.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Asli'], axis=1, errors='ignore')
y_train = train['Jumlah_CheckIn']
X_test = test.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Asli'], axis=1, errors='ignore')
y_test = test['Jumlah_CheckIn']
y_test_asli = test['Jumlah_CheckIn_Asli']

# ============================================================
# 5. TRAINING MODEL CATBOOST
# ============================================================
print("\n" + "=" * 60)
print("5. TRAINING MODEL CATBOOST")
print("=" * 60)

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    random_seed=42,
    verbose=0
)
model.fit(X_train, y_train)
print("Model CatBoost berhasil dilatih!")

# Membuktikan pentingnya setiap fitur (Feature Importance)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.get_feature_importance()
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
predictions = np.expm1(predictions)  # Inverse log transform
predictions = np.round(predictions)  # Jumlah check-in adalah bilangan bulat

mape = np.mean(np.abs((y_test_asli - predictions) / y_test_asli)) * 100

print(f"Metrik Evaluasi (Test Set - {n_test} bulan):")
print(f"  MAPE : {mape:.2f}%")

comparison = pd.DataFrame({
    'Bulan': y_test_asli.index.strftime('%B %Y'),
    'Aktual': y_test_asli.values,
    'Prediksi_CatBoost': predictions,
    'Error': (y_test_asli.values - predictions).round(0),
    'Error%': ((np.abs(y_test_asli.values - predictions) / y_test_asli.values) * 100).round(2)
})
print("\nPerbandingan Aktual vs Prediksi:")
print(comparison.to_string(index=False))

# Plot Aktual vs Prediksi pada Data Test
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ts.index, df_ts['Jumlah_CheckIn_Asli'], marker='o', label='Aktual', color='#2196F3')
ax.plot(y_test.index, predictions, marker='^', label='Prediksi (Test)', color='#F44336', linestyle='--', linewidth=2)
ax.axvline(x=y_test.index[0], color='gray', linestyle='--', alpha=0.5, label='Split Point')

textstr = f'MAPE: {mape:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=props)

ax.set_title('CatBoost - Aktual vs Prediksi', fontsize=14, fontweight='bold')
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
all_X = df_features.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Asli'], axis=1, errors='ignore')
all_y = df_features['Jumlah_CheckIn']

final_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=4,
    random_seed=42,
    verbose=0
)
final_model.fit(all_X, all_y)

future_months = 12
last_date = df_ts.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

forecasts = []
current_history = df_ts['Jumlah_CheckIn'].copy()

# Forecasting secara iteratif (Rolling window karena butuh lag dari model ke-t untuk input ke-t+1)
for date in future_dates:
    # Helper hitung rolling manual
    rolling_history = current_history.iloc[-3:] if len(current_history) >= 3 else current_history
    roll_mean_3 = rolling_history.mean() if len(rolling_history) > 0 else 0
    roll_std_3  = rolling_history.std() if len(rolling_history) > 1 else 0

    # Helper hitung weekend
    import calendar
    days = calendar.monthcalendar(date.year, date.month)
    weekend_count = sum(1 for week in days if week[4] != 0) + sum(1 for week in days if week[5] != 0)

    features = {
        'lag_1': current_history.iloc[-1] if len(current_history) >= 1 else 0,
        'lag_2': current_history.iloc[-2] if len(current_history) >= 2 else 0,
        'lag_3': current_history.iloc[-3] if len(current_history) >= 3 else 0,
        'lag_6': current_history.iloc[-6] if len(current_history) >= 6 else 0,
        'lag_12': current_history.iloc[-12] if len(current_history) >= 12 else 0,
        'rolling_mean_3': roll_mean_3,
        'rolling_std_3': roll_std_3,
        'month': date.month,
        'year': date.year,
        'quarter': date.quarter,
        'is_high_season': 1 if date.month in [6, 7, 8, 9, 10, 11, 12] else 0,
        'is_low_season': 1 if date.month in [4, 5] else 0,
        'days_in_month': calendar.monthrange(date.year, date.month)[1],
        'total_weekend_days': weekend_count
    }
    
    # Pastikan urutan kolom sesuai dengan all_X
    X_fut = pd.DataFrame([features], index=[date])[all_X.columns]
    y_pred_log = final_model.predict(X_fut)[0]
    y_pred_asli = max(0, round(np.expm1(y_pred_log)))  # Inverse log
    
    forecasts.append(y_pred_asli)
    current_history.loc[date] = y_pred_log

forecast_df = pd.DataFrame({
    'Bulan': future_dates.strftime('%B %Y'),
    'Prediksi_CheckIn': forecasts
})

print(f"\nPrediksi Okupansi Hotel (Jumlah Check-In) - {future_months} Bulan ke Depan:")
print("-" * 55)
print(forecast_df.to_string(index=False))

# Visualisasi Forecast
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ts.index, df_ts['Jumlah_CheckIn_Asli'], marker='o', label='Data Historis', color='#2196F3', linewidth=1.5)
ax.plot(future_dates, forecasts, marker='D', label='Prediksi CatBoost', color='#FF5722', linestyle='--', linewidth=2)
ax.axvline(x=df_ts.index[-1], color='gray', linestyle='--', alpha=0.5)

ax.set_title('CatBoost - Forecasting Okupansi Hotel 12 Bulan Kedepan', fontsize=14, fontweight='bold')
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
    'Model': ['CatBoost'],
    'MAPE': [round(mape, 2)],
    'Train_Size': [len(train)],
    'Test_Size': [len(test)]
})
eval_df.to_csv(f"{OUTPUT_DIR}/evaluasi_model.csv", index=False)

print(f"✓ Semua hasil evaluasi dan plot tersimpan di dalam direktori '{OUTPUT_DIR}/'")
print("Misi Selesai!")
