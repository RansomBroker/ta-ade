import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import calendar
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecasting Okupansi", layout="wide")

st.title("🏨 Dashboard Perbandingan Model Prediksi")
st.markdown("Aplikasi pintar untuk memantau, melatih, dan membandingkan performa SARIMA & XGBoost secara *real-time*.")

@st.cache_data
def load_data():
    df = pd.read_csv("data-okupansi-hotel-new.csv")
    df['Check In'] = pd.to_datetime(df['Check In'], format='%d-%m-%Y')
    df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d-%m-%Y')
    df = df[(df['Check In'] <= '2025-12-31') & (df['Check Out'] <= '2025-12-31')]
    df['Bulan'] = df['Check In'].dt.to_period('M')
    monthly = df.groupby('Bulan').size()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.asfreq('MS')
    monthly.name = 'Jumlah_CheckIn'
    return monthly

monthly_checkin = load_data()
n_test = 6
train_ts = monthly_checkin[:-n_test]
test_ts = monthly_checkin[-n_test:]

with st.expander("Tampilkan Data Historis Bulanan (Dataset Asli)"):
    fig_hist, ax_hist = plt.subplots(figsize=(14, 4))
    ax_hist.plot(monthly_checkin.index, monthly_checkin.values, marker='o', color='teal')
    ax_hist.set_title("Total Check-In per Bulan")
    ax_hist.grid(alpha=0.3)
    st.pyplot(fig_hist)
    
@st.cache_resource
def train_models(train_data, test_data):
    # Parameter statis yang sudah kita temukan (supaya loading app super cepat tanpa auto_arima berjam-jam)
    sarima_v1 = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    pred_sarima_v1 = np.round(sarima_v1.fit(disp=False).get_forecast(steps=n_test).predicted_mean)

    sarima_v2 = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    pred_sarima_v2 = np.round(sarima_v2.fit(disp=False).get_forecast(steps=n_test).predicted_mean)

    df_ts = pd.DataFrame(monthly_checkin)
    
    # helper funct
    def create_features_v1(data, target_col):
        df_feat = data.copy()
        for i in [1, 2, 3, 6, 12]:
            df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
        df_feat['month'] = df_feat.index.month
        df_feat['year'] = df_feat.index.year
        df_feat['quarter'] = df_feat.index.quarter
        return df_feat.dropna()

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
        return df_feat.dropna()

    # XGBoost V1
    df_feat_v1 = create_features_v1(df_ts, 'Jumlah_CheckIn')
    train_x1, test_x1 = df_feat_v1.iloc[:-n_test], df_feat_v1.iloc[-n_test:]
    xgb_v1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb_v1.fit(train_x1.drop('Jumlah_CheckIn', axis=1), train_x1['Jumlah_CheckIn'])
    pred_xgb_v1 = pd.Series(np.round(xgb_v1.predict(test_x1.drop('Jumlah_CheckIn', axis=1))), index=test_data.index)

    # XGBoost V2
    df_ts_v2 = df_ts.copy()
    df_ts_v2['Jumlah_CheckIn_Log'] = np.log1p(df_ts_v2['Jumlah_CheckIn'])
    df_feat_v2 = create_features_v2(df_ts_v2, 'Jumlah_CheckIn_Log')
    train_x2, test_x2 = df_feat_v2.iloc[:-n_test], df_feat_v2.iloc[-n_test:]
    
    xgb_v2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb_v2.fit(train_x2.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore'), train_x2['Jumlah_CheckIn_Log'])
    pred_xgb_v2 = np.round(np.expm1(xgb_v2.predict(test_x2.drop(['Jumlah_CheckIn', 'Jumlah_CheckIn_Log'], axis=1, errors='ignore'))))
    pred_xgb_v2 = pd.Series(pred_xgb_v2, index=test_data.index)

    return pred_sarima_v1, pred_sarima_v2, pred_xgb_v1, pred_xgb_v2

with st.spinner("Sedang memproses dan melatih ke-4 Model..."):
    p_s1, p_s2, p_x1, p_x2 = train_models(train_ts, test_ts)

def get_metrics(actual, pred, name):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

metrics_list = [
    get_metrics(test_ts.values, p_s1.values, "SARIMA v1"),
    get_metrics(test_ts.values, p_s2.values, "SARIMA v2"),
    get_metrics(test_ts.values, p_x1.values, "XGBoost v1"),
    get_metrics(test_ts.values, p_x2.values, "XGBoost v2")
]
df_metrics = pd.DataFrame(metrics_list)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Metrik Akurasi")
    # Tampilkan dengan highlight nilai terkecil (terbaik)
    st.dataframe(df_metrics.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen', axis=0), use_container_width=True)

with col2:
    st.subheader("📈 Komparasi 6 Bulan Terakhir")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    
    # Train context
    ax2.plot(train_ts[-12:].index, train_ts[-12:].values, marker='o', color='gray', alpha=0.3, label='Historical Train')
    ax2.plot(test_ts.index, test_ts.values, marker='o', color='black', linewidth=3, label='Aktual (Test)')
    
    # Predictions
    ax2.plot(test_ts.index, p_s1.values, marker='s', linestyle=':', label='SARIMA v1 (d=0)', alpha=0.7)
    ax2.plot(test_ts.index, p_s2.values, marker='s', linestyle='--', label='SARIMA v2 (D=1)')
    ax2.plot(test_ts.index, p_x1.values, marker='^', linestyle=':', label='XGBoost v1 (Basic)', alpha=0.7)
    ax2.plot(test_ts.index, p_x2.values, marker='*', linestyle='-', color='red', linewidth=2, markersize=8, label='XGBoost v2 (Advanced)')
    
    ax2.axvline(x=test_ts.index[0], color='black', linestyle='--', alpha=0.2)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

st.subheader("📋 Tabel Detail Prediksi Test Set")
df_preds = pd.DataFrame({
    'Bulan': test_ts.index.strftime('%B %Y'),
    'Set Aktual': test_ts.values,
    'S-v1': p_s1.values.astype(int),
    'S-v2': p_s2.values.astype(int),
    'X-v1': p_x1.values.astype(int),
    'X-v2': p_x2.values.astype(int)
})
st.dataframe(df_preds, use_container_width=True)
