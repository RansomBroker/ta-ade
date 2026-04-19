import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.dates as mdates
import calendar
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecasting Okupansi Hotel Dangau", layout="wide")

st.title("Dashboard Prediksi Tingkat Okupansi Hotel Dangau Kubu Raya dengan SARIMA dan XGBoost")

# --------------------------------------------------
# 1. LOAD DATA & PREPROCESSING
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data-okupansi-hotel-new.csv")
    df['Check In'] = pd.to_datetime(df['Check In'], format='%d-%m-%Y')
    df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d-%m-%Y')
    df = df[(df['Check In'] <= '2025-12-31') & (df['Check Out'] <= '2025-12-31')]
    
    df['Company'] = df['Company'].replace('TRAVELOKA', 'WALK IN').replace('RESERVASI', 'WALK IN')
    
    GOVT_KEYWORDS = ['DIKNAS', 'DINKES', 'BKD', 'KPU', 'BAWASLU', 'BAPEDAS', 'PEMDES', 'KEMENDIKBUD', 'POLTEKKES', 'POLRI', 'DINAS', 'PEMKAB', 'PEMKOT', 'DPRD', 'BPKAD', 'KECAMATAN', 'KELURAHAN', 'CAMAT', 'DISHUB', 'DISKOMINFO', 'KEMENAG', 'KODAM', 'KODIM', 'POLDA', 'POLRES', 'KEMENTERIAN', 'PEMERINTAH', 'DISPORA', 'BPBD', 'BNPB', 'KEMENKUMHAM']
    CORP_KEYWORDS = ['PT', 'CV', 'UD', 'HOTEL', 'COMPANY', 'CORP', 'GROUP', 'YAYASAN', 'WAHANA', 'VISI', 'LEMBAGA', 'FOUNDATION', 'BAKERY', 'BACKERY', 'INDONESIA', 'PERSERO', 'TBK', 'ASOSIASI', 'BANK', 'INSURANCE', 'HMI', 'GOT TALENT', 'RUNNING', 'WEDDING', 'TOUR', 'TRAVEL', 'RESERVASI', 'EUNIKE']

    def classify_segment(company):
        c = str(company).upper()
        if c in ('WALK IN',): return 'Individual'
        for kw in GOVT_KEYWORDS:
            if kw in c: return 'Government'
        for kw in CORP_KEYWORDS:
            if kw in c: return 'Corporate'
        return 'Individual'

    df['Segment'] = df['Company'].apply(classify_segment)
    
    df_raw = df.copy()
    df_raw.set_index('Check In', inplace=True)
    
    df['Bulan'] = df['Check In'].dt.to_period('M')
    monthly = df.groupby('Bulan').size()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.asfreq('MS')
    monthly.name = 'Jumlah_CheckIn'
    
    return df_raw, monthly

df_raw, monthly_checkin = load_data()
n_test = 6
train_ts = monthly_checkin[:-n_test]
test_ts = monthly_checkin[-n_test:]

tab_eda, tab_forecast = st.tabs(["📊 Analisis Data Klasik (EDA)", "🤖 Forecasting & Evaluasi AI/ML"])

# ==================================================
# TAB 1: EDA (Statistik Deskriptif dari main.py)
# ==================================================
with tab_eda:
    st.header("1. Trend Okupansi Historis")
    
    daily_occ   = df_raw.resample('D').size()
    weekly_occ  = df_raw.resample('W').size()
    monthly_occ = df_raw.resample('M').size()
    
    monthly_growth = monthly_occ.pct_change() * 100
    mean_occ = monthly_occ.mean()
    std_occ  = monthly_occ.std()
    high_threshold = mean_occ + std_occ
    low_threshold  = mean_occ - std_occ
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Rata-Rata Bulanan", f"{mean_occ:.0f} Tamu per bulan")
    k2.metric("Ambang Batas High Season", f"{high_threshold:.0f} Tamu")
    k3.metric("Ambang Batas Low Season", f"{low_threshold:.0f} Tamu")
    
    fig_tr, axes_tr = plt.subplots(3, 1, figsize=(14, 14))
    sns.set_theme(style='darkgrid')
    
    axes_tr[0].plot(daily_occ.index, daily_occ.values, color='steelblue', linewidth=0.8)
    axes_tr[0].set_title('Okupansi Harian')
    axes_tr[0].set_ylabel('Jumlah Tamu')

    axes_tr[1].plot(weekly_occ.index, weekly_occ.values, color='darkorange', linewidth=1.2)
    axes_tr[1].set_title('Okupansi Mingguan')
    axes_tr[1].set_ylabel('Jumlah Tamu')

    bar_colors = ['#e74c3c' if v >= high_threshold else '#3498db' if v <= low_threshold else '#2ecc71' for v in monthly_occ.values]
    axes_tr[2].bar(monthly_occ.index, monthly_occ.values, width=20, color=bar_colors, alpha=0.85)
    axes_tr[2].axhline(high_threshold, color='#e74c3c', linestyle='--', label='High Season')
    axes_tr[2].axhline(low_threshold,  color='#3498db', linestyle='--', label='Low Season')
    axes_tr[2].axhline(mean_occ, color='gray', linestyle=':', label='Rata-rata')
    
    ax2_tr = axes_tr[2].twinx()
    ax2_tr.plot(monthly_growth.index, monthly_growth.values, color='purple', marker='o', alpha=0.7, label='Growth MoM (%)')
    ax2_tr.axhline(0, color='purple', linestyle=':', alpha=0.5)
    axes_tr[2].set_title('Okupansi Bulanan — High / Low Season & Growth Rate')
    
    h1, l1 = axes_tr[2].get_legend_handles_labels()
    h2, l2 = ax2_tr.get_legend_handles_labels()
    axes_tr[2].legend(h1+h2, l1+l2, loc='upper left')
    
    st.pyplot(fig_tr)

    st.divider()
    
    st.header("2. Segmentasi Customer")
    seg_counts = df_raw['Segment'].value_counts()
    seg_monthly = df_raw.groupby('Segment').resample('M').size().unstack(level=0).fillna(0)
    seg_colors = {'Government': '#e74c3c', 'Corporate': '#3498db', 'Individual': '#2ecc71'}
    
    fig_seg, axes_seg = plt.subplots(2, 2, figsize=(16, 10))
    axes_seg[0, 0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
                       colors=[seg_colors[s] for s in seg_counts.index], wedgeprops=dict(edgecolor='white'))
    axes_seg[0, 0].set_title('Proporsi Segmen Pemasukan Hotel')

    for seg in seg_monthly.columns:
        axes_seg[0, 1].plot(seg_monthly.index, seg_monthly[seg], label=seg, color=seg_colors[seg])
    axes_seg[0, 1].set_title('Tren Bulanan per Segmen')
    axes_seg[0, 1].legend()

    box_data = [seg_monthly[seg].values for seg in seg_monthly.columns]
    bp = axes_seg[1, 0].boxplot(box_data, labels=seg_monthly.columns, patch_artist=True)
    for patch, seg in zip(bp['boxes'], seg_monthly.columns):
        patch.set_facecolor(seg_colors[seg])
    axes_seg[1, 0].set_title('Variabilitas Bulanan per Segmen \n(Kotak sempit = Lebih stabil, Kotak tinggi = Musiman kuat)')
    
    top_company = df_raw[df_raw['Segment'] != 'Individual']['Company'].value_counts().head(10)
    # Using proper mapping safely
    def classify_segment_stateless(c):
        GOVT_KEYWORDS = ['DIKNAS', 'DINKES', 'BKD', 'KPU', 'BAWASLU', 'BAPEDAS', 'PEMDES', 'KEMENDIKBUD', 'POLTEKKES', 'POLRI', 'DINAS', 'PEMKAB', 'PEMKOT', 'DPRD', 'BPKAD', 'KECAMATAN', 'KELURAHAN', 'CAMAT', 'DISHUB', 'DISKOMINFO', 'KEMENAG', 'KODAM', 'KODIM', 'POLDA', 'POLRES', 'KEMENTERIAN', 'PEMERINTAH', 'DISPORA', 'BPBD', 'BNPB', 'KEMENKUMHAM']
        CORP_KEYWORDS = ['PT', 'CV', 'UD', 'HOTEL', 'COMPANY', 'CORP', 'GROUP', 'YAYASAN', 'WAHANA', 'VISI', 'LEMBAGA', 'FOUNDATION', 'BAKERY', 'BACKERY', 'INDONESIA', 'PERSERO', 'TBK', 'ASOSIASI', 'BANK', 'INSURANCE', 'HMI', 'GOT TALENT', 'RUNNING', 'WEDDING', 'TOUR', 'TRAVEL', 'RESERVASI', 'EUNIKE']
        c = str(c).upper()
        if c in ('WALK IN',): return 'Individual'
        for kw in GOVT_KEYWORDS:
             if kw in c: return 'Government'
        for kw in CORP_KEYWORDS:
             if kw in c: return 'Corporate'
        return 'Individual'

    axes_seg[1, 1].barh(top_company.index[::-1], top_company.values[::-1], color=[seg_colors[classify_segment_stateless(c)] for c in top_company.index[::-1]])
    axes_seg[1, 1].set_title('Top 10 Company & Instansi Teratas')
    
    plt.tight_layout()
    st.pyplot(fig_seg)
    
    st.divider()
    
    st.header("3. Pola Kedatangan Harian (Daily Pattern)")
    df_raw['DayOfWeek'] = df_raw.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_id = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    day_counts = df_raw['DayOfWeek'].value_counts().reindex(day_order)
    
    df_raw['Month'] = df_raw.index.month
    pivot_day = df_raw.groupby(['DayOfWeek', 'Month']).size().unstack(fill_value=0).reindex(day_order)
    pivot_day.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    fig_day, axes_day = plt.subplots(1, 2, figsize=(16, 6))
    bar_c = ['#e74c3c' if d in ('Saturday','Sunday') else '#3498db' for d in day_order]
    axes_day[0].bar(day_id, day_counts.values, color=bar_c)
    axes_day[0].set_title('Akumulasi Check-In per Hari')
    
    sns.heatmap(pivot_day, ax=axes_day[1], cmap='YlOrRd', yticklabels=day_id)
    axes_day[1].set_title('Heatmap: Hari vs Bulan Kepadatan')
    
    st.pyplot(fig_day)


# ==================================================
# TAB 2: FORECASTING MODEL
# ==================================================
with tab_forecast:
    st.markdown("### Pencarian & Evaluasi Performa Algoritma")
    
    @st.cache_resource
    def train_models(train_data, test_data):
        # SARIMA Setup
        sv1 = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        p_sv1 = np.round(sv1.fit(disp=False).get_forecast(steps=n_test).predicted_mean)

        sv2 = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        p_sv2 = np.round(sv2.fit(disp=False).get_forecast(steps=n_test).predicted_mean)

        # XGBoost Setup
        df_ts = pd.DataFrame(monthly_checkin)
        
        def feat_v1(d, col):
            tmp = d.copy()
            for i in [1,2,3,6,12]: tmp[f'lag_{i}'] = tmp[col].shift(i)
            tmp['month']   = tmp.index.month
            tmp['year']    = tmp.index.year
            tmp['quarter'] = tmp.index.quarter
            return tmp.dropna()

        def feat_v2(d, col):
            tmp = d.copy()
            for i in [1,2,3,6,12]: tmp[f'lag_{i}'] = tmp[col].shift(i)
            tmp['rolling_mean_3'] = tmp[col].shift(1).rolling(window=3).mean()
            tmp['rolling_std_3']  = tmp[col].shift(1).rolling(window=3).std()
            tmp['month']   = tmp.index.month
            tmp['year']    = tmp.index.year
            tmp['quarter'] = tmp.index.quarter
            tmp['is_high_season'] = tmp['month'].apply(lambda x: 1 if x in [6,7,8,9,10,11,12] else 0)
            tmp['is_low_season']  = tmp['month'].apply(lambda x: 1 if x in [4,5] else 0)
            tmp['days_in_month']  = tmp.index.map(lambda x: calendar.monthrange(x.year, x.month)[1])
            tmp['t_weekends']     = [sum(1 for week in calendar.monthcalendar(d.year, d.month) if week[4]!=0)+sum(1 for week in calendar.monthcalendar(d.year, d.month) if week[5]!=0) for d in tmp.index]
            return tmp.dropna()

        # Fit XGB v1
        fv1 = feat_v1(df_ts, 'Jumlah_CheckIn')
        tr_x1, ts_x1 = fv1.iloc[:-n_test], fv1.iloc[-n_test:]
        xgb1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        xgb1.fit(tr_x1.drop('Jumlah_CheckIn', axis=1), tr_x1['Jumlah_CheckIn'])
        p_xv1 = pd.Series(np.round(xgb1.predict(ts_x1.drop('Jumlah_CheckIn', axis=1))), index=test_data.index)

        # Fit XGB v2
        d2 = df_ts.copy()
        d2['Log_Y'] = np.log1p(d2['Jumlah_CheckIn'])
        fv2 = feat_v2(d2, 'Log_Y')
        tr_x2, ts_x2 = fv2.iloc[:-n_test], fv2.iloc[-n_test:]
        xgb2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        xgb2.fit(tr_x2.drop(['Jumlah_CheckIn', 'Log_Y'], axis=1, errors='ignore'), tr_x2['Log_Y'])
        p_xv2 = pd.Series(np.round(np.expm1(xgb2.predict(ts_x2.drop(['Jumlah_CheckIn', 'Log_Y'], axis=1, errors='ignore')))), index=test_data.index)
        
        return p_sv1, p_sv2, p_xv1, p_xv2

    with st.spinner("Sedang memproses dan melatih ke-4 Algoritma AI..."):
        p_s1, p_s2, p_x1, p_x2 = train_models(train_ts, test_ts)

    def get_metrics(actual, pred, name):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

    metrics_list = [
        get_metrics(test_ts.values, p_s1.values, "SARIMA v1 (Statis)"),
        get_metrics(test_ts.values, p_s2.values, "SARIMA v2 (Tuning Musiman)"),
        get_metrics(test_ts.values, p_x1.values, "XGBoost v1 (Lags Basic)"),
        get_metrics(test_ts.values, p_x2.values, "XGBoost v2 (Analisis Kalender + Log)")
    ]
    df_metrics = pd.DataFrame(metrics_list)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Metrik Akurasi")
        st.dataframe(df_metrics.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen', axis=0), use_container_width=True)

    with col2:
        st.subheader("📈 Komparasi 6 Bulan Terakhir")
        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
        ax_cmp.plot(train_ts[-12:].index, train_ts[-12:].values, marker='o', color='gray', alpha=0.3, label='Train (Past Year)')
        ax_cmp.plot(test_ts.index, test_ts.values, marker='o', color='black', linewidth=3, label='Aktual Data')
        ax_cmp.plot(test_ts.index, p_s1.values, marker='s', linestyle=':', label='SARIMA v1', alpha=0.5)
        ax_cmp.plot(test_ts.index, p_s2.values, marker='s', linestyle='--', label='SARIMA v2')
        ax_cmp.plot(test_ts.index, p_x1.values, marker='^', linestyle=':', label='XGB v1', alpha=0.5)
        ax_cmp.plot(test_ts.index, p_x2.values, marker='*', linestyle='-', color='red', linewidth=2, label='XGB v2')
        ax_cmp.axvline(x=test_ts.index[0], color='black', linestyle='--', alpha=0.2)
        ax_cmp.legend(fontsize=8)
        st.pyplot(fig_cmp)

    st.subheader("📋 Tabel Detail Prediksi")
    df_preds = pd.DataFrame({
        'Bulan': test_ts.index.strftime('%B %Y'),
        'Aktual': test_ts.values,
        'SARIMA v1': p_s1.values.astype(int),
        'SARIMA v2': p_s2.values.astype(int),
        'XGBoost v1': p_x1.values.astype(int),
        'XGBoost v2': p_x2.values.astype(int)
    })
    st.dataframe(df_preds, use_container_width=True)
