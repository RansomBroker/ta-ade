"""
SARIMA Model v2 - Prediksi Okupansi Hotel
======================================
Pipeline:
1. Load & Preprocessing Data
2. Agregasi Time Series Bulanan (Check-In)
3. Eksplorasi & Visualisasi (Decomposition)
4. Uji Stasioneritas (ADF Test)
5. Identifikasi Parameter (ACF/PACF + Auto ARIMA)
6. Split Data Train/Test
7. Training Model SARIMA
8. Evaluasi Model (MAE, RMSE, MAPE)
9. Diagnostic Check (Residual Analysis)
10. Forecasting 12 Bulan ke Depan
11. Simpan Hasil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# Buat folder output
OUTPUT_DIR = "output_sarima_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & PREPROCESSING DATA
# ============================================================
print("=" * 60)
print("1. LOAD & PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv("data-okupansi-hotel-new.csv")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

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

print(f"\nRentang data Check In: {df['Check In'].min()} - {df['Check In'].max()}")

# ============================================================
# 2. AGREGASI TIME SERIES BULANAN
# ============================================================
print("\n" + "=" * 60)
print("2. AGREGASI TIME SERIES BULANAN")
print("=" * 60)

print(f"Total transaksi Check-In: {len(df)}")

# Hitung jumlah Check-In per bulan
df['Bulan'] = df['Check In'].dt.to_period('M')
monthly_checkin = df.groupby('Bulan').size()

# Konversi ke DatetimeIndex dengan frekuensi bulanan
monthly_checkin.index = monthly_checkin.index.to_timestamp()
monthly_checkin = monthly_checkin.asfreq('MS')  # Month Start frequency
monthly_checkin.name = 'Jumlah_CheckIn'

print(f"\nJumlah data bulanan: {len(monthly_checkin)}")
print(f"Periode: {monthly_checkin.index[0].strftime('%B %Y')} - {monthly_checkin.index[-1].strftime('%B %Y')}")
print(f"\nStatistik Deskriptif:")
print(monthly_checkin.describe())
print(f"\nData Bulanan:")
print(monthly_checkin)

# Plot Time Series Asli
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly_checkin.index, monthly_checkin.values, marker='o', linewidth=2, 
        color='#2196F3', markersize=5)
ax.set_title('Time Series - Jumlah Check-In Hotel per Bulan', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_time_series_asli.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 3. EKSPLORASI & VISUALISASI (DECOMPOSITION)
# ============================================================
print("\n" + "=" * 60)
print("3. DEKOMPOSISI TIME SERIES")
print("=" * 60)

decomposition = seasonal_decompose(monthly_checkin, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], title='Observed', color='#2196F3')
axes[0].grid(True, alpha=0.3)
decomposition.trend.plot(ax=axes[1], title='Trend', color='#FF9800')
axes[1].grid(True, alpha=0.3)
decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='#4CAF50')
axes[2].grid(True, alpha=0.3)
decomposition.resid.plot(ax=axes[3], title='Residual', color='#F44336')
axes[3].grid(True, alpha=0.3)

plt.suptitle('Dekomposisi Time Series (Additive, Period=12)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_decomposition.png", dpi=150, bbox_inches='tight')
plt.show()

print("Dekomposisi selesai - komponen: Observed, Trend, Seasonal, Residual")

# ============================================================
# 4. UJI STASIONERITAS (ADF TEST)
# ============================================================
print("\n" + "=" * 60)
print("4. UJI STASIONERITAS (ADF TEST)")
print("=" * 60)

def adf_test(series, title=''):
    """Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n--- ADF Test: {title} ---")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Lags Used     : {result[2]}")
    print(f"  Observations  : {result[3]}")
    for key, value in result[4].items():
        print(f"  Critical Value ({key}): {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  >> STASIONER (p-value < 0.05)")
    else:
        print(f"  >> TIDAK STASIONER (p-value >= 0.05)")
    
    return result[1]

# Test data asli
p_original = adf_test(monthly_checkin, 'Data Asli')

# Differencing orde 1
diff_1 = monthly_checkin.diff().dropna()
p_diff1 = adf_test(diff_1, 'Differencing Orde 1 (d=1)')

# Seasonal differencing
diff_seasonal = monthly_checkin.diff(12).dropna()
p_seasonal = adf_test(diff_seasonal, 'Seasonal Differencing (D=1, s=12)')

# Differencing + Seasonal differencing
diff_both = monthly_checkin.diff().diff(12).dropna()
p_both = adf_test(diff_both, 'Differencing d=1 + D=1')

# ============================================================
# 5. IDENTIFIKASI PARAMETER (ACF/PACF + AUTO ARIMA)
# ============================================================
print("\n" + "=" * 60)
print("5. IDENTIFIKASI PARAMETER SARIMA")
print("=" * 60)

# Plot ACF & PACF data yang sudah di-differencing
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ACF & PACF - Differencing biasa (d=1)
lags_d1 = min(24, len(diff_1)//2 - 1)
plot_acf(diff_1, ax=axes[0, 0], lags=lags_d1, title='ACF - Differencing (d=1)')
plot_pacf(diff_1, ax=axes[0, 1], lags=lags_d1, title='PACF - Differencing (d=1)')

# ACF & PACF - Seasonal differencing
lags_D1 = min(24, len(diff_seasonal)//2 - 1)
plot_acf(diff_seasonal, ax=axes[1, 0], lags=lags_D1, title='ACF - Seasonal Diff (D=1)')
plot_pacf(diff_seasonal, ax=axes[1, 1], lags=lags_D1, title='PACF - Seasonal Diff (D=1)')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.suptitle('ACF & PACF Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_acf_pacf.png", dpi=150, bbox_inches='tight')
plt.show()

# Auto ARIMA untuk pencarian parameter optimal (Pencegahan Data Leakage)
print("\nMencari parameter optimal dengan Auto ARIMA pada Train Set...")
auto_model = auto_arima(
    monthly_checkin[:-6],
    seasonal=True,
    m=12,               # periode musiman = 12 bulan
    d=None,             # auto-detect d
    D=1, 
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    trace=True,          # print progress
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='aic'
)

print(f"\nParameter SARIMA terbaik (Auto ARIMA): {auto_model.order} x {auto_model.seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
print(f"BIC: {auto_model.bic():.2f}")
print(auto_model.summary())

# Gunakan parameter dari auto_arima
order = auto_model.order                    # (p, d, q)
seasonal_order = auto_model.seasonal_order  # (P, D, Q, s)

print(f"\n>> Parameter yang digunakan:")
print(f"   Order (p,d,q)       = {order}")
print(f"   Seasonal (P,D,Q,s) = {seasonal_order}")

# ============================================================
# 6. SPLIT DATA: TRAIN & TEST
# ============================================================
print("\n" + "=" * 60)
print("6. SPLIT DATA: TRAIN & TEST")
print("=" * 60)

n_test = 6  # 6 bulan terakhir untuk test
train = monthly_checkin[:-n_test]
test = monthly_checkin[-n_test:]

print(f"Train: {train.index[0].strftime('%B %Y')} - {train.index[-1].strftime('%B %Y')} ({len(train)} bulan)")
print(f"Test : {test.index[0].strftime('%B %Y')} - {test.index[-1].strftime('%B %Y')} ({len(test)} bulan)")

# Visualisasi split
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train.index, train.values, marker='o', label='Train', color='#2196F3', markersize=5)
ax.plot(test.index, test.values, marker='s', label='Test', color='#F44336', markersize=7)
ax.axvline(x=test.index[0], color='gray', linestyle='--', alpha=0.7, label='Split Point')
ax.set_title('Train / Test Split', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_train_test_split.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. TRAINING MODEL SARIMA
# ============================================================
print("\n" + "=" * 60)
print("7. TRAINING MODEL SARIMA")
print("=" * 60)

model = SARIMAX(
    train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
print(results.summary())

# ============================================================
# 8. EVALUASI MODEL
# ============================================================
print("\n" + "=" * 60)
print("8. EVALUASI MODEL")
print("=" * 60)

# Prediksi pada periode test
forecast_test = results.get_forecast(steps=n_test)
forecast_test_mean = forecast_test.predicted_mean
forecast_test_ci = forecast_test.conf_int()

# Hitung metrik error
mae = mean_absolute_error(test, forecast_test_mean)
rmse = np.sqrt(mean_squared_error(test, forecast_test_mean))
mape = np.mean(np.abs((test - forecast_test_mean) / test)) * 100

print(f"\nMetrik Evaluasi (Test Set - {n_test} bulan):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAPE : {mape:.2f}%")

# Tabel perbandingan aktual vs prediksi
comparison = pd.DataFrame({
    'Bulan': test.index.strftime('%B %Y'),
    'Aktual': test.values,
    'Prediksi': forecast_test_mean.values.round(0),
    'Error': (test.values - forecast_test_mean.values).round(0),
    'Error%': ((np.abs(test.values - forecast_test_mean.values) / test.values) * 100).round(2)
})
print(f"\nPerbandingan Aktual vs Prediksi:")
print(comparison.to_string(index=False))

# Visualisasi Aktual vs Prediksi
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train.index, train.values, marker='o', label='Train', color='#2196F3', 
        markersize=4, alpha=0.7)
ax.plot(test.index, test.values, marker='s', label='Aktual (Test)', color='#4CAF50', 
        markersize=7, linewidth=2)
ax.plot(forecast_test_mean.index, forecast_test_mean.values, marker='^', 
        label='Prediksi (Test)', color='#F44336', markersize=7, linewidth=2, linestyle='--')
ax.fill_between(forecast_test_ci.index, 
                forecast_test_ci.iloc[:, 0], 
                forecast_test_ci.iloc[:, 1], 
                color='#F44336', alpha=0.15, label='95% Confidence Interval')
ax.axvline(x=test.index[0], color='gray', linestyle='--', alpha=0.5)

# Anotasi metrik
textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_title(f'SARIMA{order}x{seasonal_order} - Aktual vs Prediksi', fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_aktual_vs_prediksi.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. DIAGNOSTIC CHECK (RESIDUAL ANALYSIS)
# ============================================================
print("\n" + "=" * 60)
print("9. DIAGNOSTIC CHECK")
print("=" * 60)

# Plot diagnostik bawaan statsmodels
try:
    fig = results.plot_diagnostics(figsize=(14, 10))
    plt.suptitle('Model Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Plot diagnostics dilewati: jumlah observasi tidak mencukupi untuk model (Error: {e})")

# Ljung-Box Test
residuals = results.resid
max_lb_lag = len(residuals) // 2 - 1
lb_lags = [l for l in [6, 12, 18, 24] if l <= max_lb_lag]
if not lb_lags:
    lb_lags = [max(1, max_lb_lag)]

if lb_lags and lb_lags[0] > 0:
    lb_test = acorr_ljungbox(residuals, lags=lb_lags, return_df=True)
    print("\nLjung-Box Test (H0: Residual = White Noise):")
    print(lb_test)
    print("\nInterpretasi:")
    for lag, row in lb_test.iterrows():
        status = "White Noise ✓" if row['lb_pvalue'] > 0.05 else "Ada autokorelasi ✗"
        print(f"  Lag {lag}: p-value = {row['lb_pvalue']:.4f} -> {status}")
else:
    print("\nLjung-Box Test tidak dapat dihitung karena ukuran sampel terlalu kecil.")

# Statistik Residual
print(f"\nStatistik Residual:")
print(f"  Mean     : {residuals.mean():.4f}")
print(f"  Std Dev  : {residuals.std():.4f}")
print(f"  Skewness : {residuals.skew():.4f}")
print(f"  Kurtosis : {residuals.kurtosis():.4f}")

# ============================================================
# 10. FORECASTING 12 BULAN KE DEPAN
# ============================================================
print("\n" + "=" * 60)
print("10. FORECASTING 12 BULAN KE DEPAN (2026)")
print("=" * 60)

# Re-train model dengan seluruh data
model_full = SARIMAX(
    monthly_checkin,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
results_full = model_full.fit(disp=False)

# Forecast 12 bulan ke depan
n_forecast = 12
forecast_future = results_full.get_forecast(steps=n_forecast)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()

print(f"\nPrediksi Okupansi Hotel (Jumlah Check-In) - 12 Bulan ke Depan:")
print("-" * 55)
forecast_df = pd.DataFrame({
    'Bulan': forecast_future_mean.index.strftime('%B %Y'),
    'Prediksi_CheckIn': forecast_future_mean.values.round(0).astype(int),
    'CI_Lower': forecast_future_ci.iloc[:, 0].values.round(0).astype(int),
    'CI_Upper': forecast_future_ci.iloc[:, 1].values.round(0).astype(int)
})
print(forecast_df.to_string(index=False))

# Visualisasi Forecast
fig, ax = plt.subplots(figsize=(16, 6))

# Data historis
ax.plot(monthly_checkin.index, monthly_checkin.values, marker='o', 
        label='Data Historis', color='#2196F3', markersize=4, linewidth=1.5)

# Forecast
ax.plot(forecast_future_mean.index, forecast_future_mean.values, marker='D', 
        label='Prediksi 2025', color='#FF5722', markersize=7, linewidth=2, linestyle='--')
ax.fill_between(forecast_future_ci.index, 
                forecast_future_ci.iloc[:, 0], 
                forecast_future_ci.iloc[:, 1], 
                color='#FF5722', alpha=0.15, label='95% CI')

# Garis pemisah historis & prediksi
ax.axvline(x=monthly_checkin.index[-1], color='gray', linestyle='--', alpha=0.5)
ax.text(monthly_checkin.index[-1], ax.get_ylim()[1] * 0.95, ' Forecast →', 
        fontsize=10, color='gray', ha='left')

ax.set_title(f'SARIMA{order}x{seasonal_order} - Forecasting Okupansi Hotel 2026', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Check-In')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_forecast_2026.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 11. SIMPAN HASIL
# ============================================================
print("\n" + "=" * 60)
print("11. SIMPAN HASIL")
print("=" * 60)

# Simpan data historis bulanan
monthly_checkin.to_csv(f"{OUTPUT_DIR}/data_bulanan_checkin.csv", header=True)
print(f"✓ Data bulanan disimpan: {OUTPUT_DIR}/data_bulanan_checkin.csv")

# Simpan hasil evaluasi
eval_results = {
    'Model': [f'SARIMA{order}x{seasonal_order}'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'AIC': [results.aic],
    'BIC': [results.bic],
    'Train_Size': [len(train)],
    'Test_Size': [len(test)]
}
eval_df = pd.DataFrame(eval_results)
eval_df.to_csv(f"{OUTPUT_DIR}/evaluasi_model.csv", index=False)
print(f"✓ Hasil evaluasi disimpan: {OUTPUT_DIR}/evaluasi_model.csv")

# Simpan perbandingan aktual vs prediksi
comparison.to_csv(f"{OUTPUT_DIR}/aktual_vs_prediksi.csv", index=False)
print(f"✓ Perbandingan aktual vs prediksi disimpan: {OUTPUT_DIR}/aktual_vs_prediksi.csv")

# Simpan forecast 2026
forecast_df.to_csv(f"{OUTPUT_DIR}/forecast_2026.csv", index=False)
print(f"✓ Forecast 2026 disimpan: {OUTPUT_DIR}/forecast_2026.csv")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN MODEL SARIMA")
print("=" * 60)
print(f"  Model            : SARIMA{order}x{seasonal_order}")
print(f"  Data              : Jumlah Check-In per Bulan")
print(f"  Periode Data     : {monthly_checkin.index[0].strftime('%B %Y')} - {monthly_checkin.index[-1].strftime('%B %Y')}")
print(f"  Jumlah Data      : {len(monthly_checkin)} bulan")
print(f"  Train Set        : {len(train)} bulan")
print(f"  Test Set         : {len(test)} bulan")
print(f"  MAE              : {mae:.2f}")
print(f"  RMSE             : {rmse:.2f}")
print(f"  MAPE             : {mape:.2f}%")
print(f"  AIC              : {results.aic:.2f}")
print(f"  BIC              : {results.bic:.2f}")
print(f"  Forecast Horizon : 12 bulan (Jan - Des 2026)")
print(f"  Output Folder    : {OUTPUT_DIR}/")
print("=" * 60)
