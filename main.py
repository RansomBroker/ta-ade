import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns

# Load Data
df = pd.read_csv("data-okupansi-hotel-new.csv")
print("Initial Data:")
print(df.head())

# drop column
print('Droping column yang tidak digunakan')
df = df.drop(columns=['No', 'Jenis Kelamin', 'Warga Negara', 'Pekerjaan',
 'Nomor Kamar', 'Tipe Kamar', 'Telp.',
 'Alamat', 'Pax', 'Jlh Malam', 'Nama'])
print(df.head())

# Convert Checkin dan Checkout ke datetime
print('Convert Checkin dan Checkout ke datetime')
df['Check In'] = pd.to_datetime(df['Check In'], format='%d-%m-%Y')
df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d-%m-%Y')

# Potong data dari awal hingga 2025-12-31 (berdasarkan Check In DAN Check Out)
df = df[(df['Check In'] <= '2025-12-31') & (df['Check Out'] <= '2025-12-31')]

# Membuat Indeks baru
df.set_index('Check In', inplace=True)

# Informasi Data
print("Data Information:")
print(df.info())

# Statistik Deskriptif
print("Data Shape:")
print(df.shape)

# Lihat distribusi Company
print("\nTop 10 Company:")
print(df['Company'].value_counts().head(10))

# Convert company TRAVELOKA == WALK IN && RESERVASI == WALK IN 
df['Company'] = df['Company'].replace('TRAVELOKA', 'WALK IN')
df['Company'] = df['Company'].replace('RESERVASI', 'WALK IN')
print("\nTop 10 Company:")
print(df['Company'].value_counts().head(10))

# ============================================================
# BAGIAN 1: TREND OKUPANSI (TIME SERIES)
# ============================================================

# --- 1.1 Agregasi Okupansi ---
daily_occ   = df.resample('D').size().rename('Jumlah Tamu')
weekly_occ  = df.resample('W').size().rename('Jumlah Tamu')
monthly_occ = df.resample('M').size().rename('Jumlah Tamu')

print("\n--- Okupansi Harian (5 pertama) ---")
print(daily_occ.head())

print("\n--- Okupansi Mingguan (5 pertama) ---")
print(weekly_occ.head())

print("\n--- Okupansi Bulanan (5 pertama) ---")
print(monthly_occ.head())

# --- 1.2 Growth Rate Bulanan (Month-over-Month) ---
monthly_growth = monthly_occ.pct_change() * 100
monthly_growth.name = 'Growth (%)'

print("\n--- Growth Rate Bulanan (%) ---")
print(monthly_growth.dropna().round(2))

# --- 1.3 Identifikasi High / Low Season ---
mean_occ = monthly_occ.mean()
std_occ  = monthly_occ.std()

high_threshold = mean_occ + std_occ
low_threshold  = mean_occ - std_occ

high_season = monthly_occ[monthly_occ >= high_threshold]
low_season  = monthly_occ[monthly_occ <= low_threshold]

print(f"\nRata-rata bulanan : {mean_occ:.0f} tamu")
print(f"High Season (≥ {high_threshold:.0f}): {high_season.index.strftime('%b %Y').tolist()}")
print(f"Low Season  (≤ {low_threshold:.0f}): {low_season.index.strftime('%b %Y').tolist()}")

# ============================================================
# VISUALISASI
# ============================================================
sns.set_theme(style='darkgrid')
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle('Trend Okupansi Hotel', fontsize=16, fontweight='bold', y=0.98)

# ---- Plot 1: Harian ----
axes[0].plot(daily_occ.index, daily_occ.values, color='steelblue', linewidth=0.8, alpha=0.7)
axes[0].set_title('Okupansi Harian')
axes[0].set_ylabel('Jumlah Tamu')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# ---- Plot 2: Mingguan ----
axes[1].plot(weekly_occ.index, weekly_occ.values, color='darkorange', linewidth=1.2)
axes[1].set_title('Okupansi Mingguan')
axes[1].set_ylabel('Jumlah Tamu')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# ---- Plot 3: Bulanan + High/Low Season + Growth ----
bar_colors = []
for val in monthly_occ.values:
    if val >= high_threshold:
        bar_colors.append('#e74c3c')   # merah = high season
    elif val <= low_threshold:
        bar_colors.append('#3498db')   # biru  = low season
    else:
        bar_colors.append('#2ecc71')   # hijau = normal

axes[2].bar(monthly_occ.index, monthly_occ.values, width=20,
            color=bar_colors, edgecolor='none', alpha=0.85)
axes[2].axhline(high_threshold, color='#e74c3c', linestyle='--', linewidth=1.2, label=f'High Season (≥{high_threshold:.0f})')
axes[2].axhline(low_threshold,  color='#3498db', linestyle='--', linewidth=1.2, label=f'Low Season (≤{low_threshold:.0f})')
axes[2].axhline(mean_occ, color='gray', linestyle=':', linewidth=1.0, label=f'Rata-rata ({mean_occ:.0f})')

# Twin axis untuk growth rate
ax2 = axes[2].twinx()
ax2.plot(monthly_growth.index, monthly_growth.values, color='purple',
         linewidth=1.2, linestyle='-', marker='o', markersize=3, alpha=0.7, label='Growth MoM (%)')
ax2.axhline(0, color='purple', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_ylabel('Growth MoM (%)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

axes[2].set_title('Okupansi Bulanan — High / Low Season & Growth Rate')
axes[2].set_ylabel('Jumlah Tamu')
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Gabung legend
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
output_path = 'trend_okupansi.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGrafik disimpan: {output_path}")
plt.show()

# ============================================================
# BAGIAN 2: SEGMENTASI CUSTOMER
# Government vs Corporate vs Individual
# ============================================================

# --- 2.1 Keyword list per segmen ---
GOVT_KEYWORDS = [
    'DIKNAS', 'DINKES', 'BKD', 'KPU', 'BAWASLU', 'BAPEDAS', 'PEMDES',
    'KEMENDIKBUD', 'POLTEKKES', 'POLRI', 'DINAS', 'PEMKAB', 'PEMKOT',
    'DPRD', 'BPKAD', 'KECAMATAN', 'KELURAHAN', 'CAMAT', 'DISHUB',
    'DISKOMINFO', 'KEMENAG', 'KODAM', 'KODIM', 'POLDA', 'POLRES',
    'KEMENTERIAN', 'PEMERINTAH', 'DISPORA', 'BPBD', 'BNPB', 'KEMENKUMHAM'
]
CORP_KEYWORDS = [
    'PT', 'CV', 'UD', 'HOTEL', 'COMPANY', 'CORP', 'GROUP', 'YAYASAN',
    'WAHANA', 'VISI', 'LEMBAGA', 'FOUNDATION', 'BAKERY', 'BACKERY',
    'INDONESIA', 'PERSERO', 'TBK', 'ASOSIASI', 'BANK', 'INSURANCE',
    'HMI', 'GOT TALENT', 'RUNNING', 'WEDDING', 'TOUR', 'TRAVEL',
    'RESERVASI', 'EUNIKE',
]

def classify_segment(company: str) -> str:
    c = str(company).upper()
    if c in ('WALK IN',):
        return 'Individual'
    for kw in GOVT_KEYWORDS:
        if kw in c:
            return 'Government'
    for kw in CORP_KEYWORDS:
        if kw in c:
            return 'Corporate'
    return 'Individual'

df['Segment'] = df['Company'].apply(classify_segment)

seg_counts = df['Segment'].value_counts()
print("\n--- Distribusi Segmen ---")
print(seg_counts)
print(f"\nProporsi (%):\n{(seg_counts / seg_counts.sum() * 100).round(1)}")

# --- 2.2 Tren bulanan per segmen ---
seg_monthly = df.groupby('Segment').resample('M').size().unstack(level=0).fillna(0)

# --- 2.3 Stabilitas (Coefficient of Variation) ---
print("\n--- Stabilitas per Segmen (CV = std/mean, lebih kecil = lebih stabil) ---")
for seg in seg_monthly.columns:
    s = seg_monthly[seg]
    cv = s.std() / s.mean() * 100
    print(f"  {seg:15s}: mean={s.mean():.0f}/bln, CV={cv:.1f}%")

# --- 2.4 Visualisasi Segmentasi ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Segmentasi Customer: Government vs Corporate vs Individual',
             fontsize=15, fontweight='bold')

seg_colors = {'Government': '#e74c3c', 'Corporate': '#3498db', 'Individual': '#2ecc71'}

# Plot A: Pie chart proporsi
axes[0, 0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
               colors=[seg_colors[s] for s in seg_counts.index],
               startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0, 0].set_title('Proporsi Segmen Keseluruhan')

# Plot B: Tren bulanan per segmen (line)
for seg in seg_monthly.columns:
    axes[0, 1].plot(seg_monthly.index, seg_monthly[seg],
                    label=seg, color=seg_colors[seg], linewidth=1.5)
axes[0, 1].set_title('Tren Bulanan per Segmen')
axes[0, 1].set_ylabel('Jumlah Tamu')
axes[0, 1].legend()
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot C: Box plot distribusi bulanan per segmen (seasonality spread)
box_data = [seg_monthly[seg].values for seg in seg_monthly.columns]
bp = axes[1, 0].boxplot(box_data, labels=seg_monthly.columns, patch_artist=True,
                         medianprops=dict(color='white', linewidth=2))
for patch, seg in zip(bp['boxes'], seg_monthly.columns):
    patch.set_facecolor(seg_colors[seg])
    patch.set_alpha(0.8)
axes[1, 0].set_title('Variabilitas Bulanan per Segmen\n(box lebih pendek = lebih stabil)')
axes[1, 0].set_ylabel('Jumlah Tamu / Bulan')

# Plot D: Top 10 Company non-individual
top_non_indiv = (df[df['Segment'] != 'Individual']['Company']
                 .value_counts().head(10))
colors_bar = [seg_colors[classify_segment(c)] for c in top_non_indiv.index]
axes[1, 1].barh(top_non_indiv.index[::-1], top_non_indiv.values[::-1],
                color=colors_bar[::-1], edgecolor='none')
axes[1, 1].set_title('Top 10 Company (Government & Corporate)')
axes[1, 1].set_xlabel('Jumlah Tamu')
for i, (v, label) in enumerate(zip(top_non_indiv.values[::-1],
                                    top_non_indiv.index[::-1])):
    axes[1, 1].text(v + 5, i, str(v), va='center', fontsize=8)

plt.tight_layout()
plt.savefig('segmentasi_customer.png', dpi=150, bbox_inches='tight')
print("\nGrafik disimpan: segmentasi_customer.png")
plt.show()

# ============================================================
# BAGIAN 3: DAILY PATTERN ANALYSIS
# ============================================================

# --- 3.1 Distribusi Hari dalam Seminggu ---
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_id     = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

df['DayOfWeek'] = df.index.day_name()
day_counts = df['DayOfWeek'].value_counts().reindex(day_order)

print("\n--- Check-in per Hari (rata-rata) ---")
for d, di, val in zip(day_order, day_id, day_counts.values):
    bar = '█' * int(val / day_counts.max() * 30)
    print(f"  {di:8s}: {val:5d}  {bar}")

# --- 3.2 Heatmap: Hari vs Bulan ---
df['Month']    = df.index.month
df['MonthName']= df.index.month_name()
pivot_day_month = df.groupby(['DayOfWeek', 'Month']).size().unstack(fill_value=0)
pivot_day_month = pivot_day_month.reindex(day_order)
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot_day_month.columns = [month_names[m-1] for m in pivot_day_month.columns]

# --- 3.3 Visualisasi Daily Pattern ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Daily Pattern Analysis', fontsize=15, fontweight='bold')

# Plot A: Bar chart per hari
bar_colors_day = ['#e74c3c' if d in ('Saturday','Sunday') else '#3498db' for d in day_order]
bars = axes[0].bar(day_id, day_counts.values, color=bar_colors_day,
                   edgecolor='none', alpha=0.85)
axes[0].set_title('Total Check-in per Hari dalam Seminggu\n(Weekend | Weekday)')
axes[0].set_ylabel('Jumlah Check-in')
for bar_, val in zip(bars, day_counts.values):
    axes[0].text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + 30,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_ylim(0, day_counts.max() * 1.15)

# Plot B: Heatmap hari vs bulan
sns.heatmap(pivot_day_month, ax=axes[1], cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            yticklabels=day_id, annot=False, fmt='d', cbar_kws={'label': 'Jumlah'})
axes[1].set_title('Heatmap Check-in per Hari × Bulan')
axes[1].set_xlabel('Bulan')
axes[1].set_ylabel('Hari')

plt.tight_layout()
plt.savefig('daily_pattern.png', dpi=150, bbox_inches='tight')
print("Grafik disimpan: daily_pattern.png")
plt.show()
