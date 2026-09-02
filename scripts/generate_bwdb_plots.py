import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ['MPLCONFIGDIR'] = '/tmp'
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

out_dir = 'output/bwdb_analysis'
os.makedirs(out_dir, exist_ok=True)

# 1. Load Station Metadata
bmd_st = pd.read_csv('data/stations/data_2020_2025/Stations.csv').dropna(subset=['Latitude', 'Longitude'])
excel_path = 'data/stations/BWDB_Rainfall_2000_2025.xlsx'
bwdb_st = pd.read_excel(excel_path, sheet_name='StationList')
failed_st = pd.read_excel(excel_path, sheet_name='FailedStations')

# Load Rainfall data
bwdb_rf = pd.read_excel(excel_path, sheet_name='RainfallData')
bwdb_rf['Date'] = pd.to_datetime(bwdb_rf['Date'])
bwdb_rf_idx = bwdb_rf.set_index('Date')

# Load BMD time series
bmd_dir = 'data/stations/data_2020_2025'
bmd_data = {}
for idx, r in bmd_st.iterrows():
    st_name = r['Station']
    path = os.path.join(bmd_dir, f"{st_name}.csv")
    if not os.path.exists(path):
        m = glob.glob(os.path.join(bmd_dir, f"*{st_name}*.csv"))
        if m:
            path = m[0]
        else:
            m = glob.glob(os.path.join(bmd_dir, f"*{st_name.lower()}*.csv"))
            if m:
                path = m[0]
    if os.path.exists(path):
        df_temp = pd.read_csv(path)
        df_temp['Date'] = pd.to_datetime(df_temp['Datetime'])
        bmd_data[st_name] = df_temp.set_index('Date')['Rainfall']

# Load precomputed metrics
metrics_df = pd.read_csv(os.path.join(out_dir, 'paired_station_metrics.csv'))
bwdb_st_dist = pd.read_csv(os.path.join(out_dir, 'bwdb_stations_with_distance.csv'))

print("Plotting Figure 1: Spatial Map of Stations...")
# FIG 1: Spatial Map
fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

# Filter valid lat/lon for BWDB (mark CL312 separately)
bwdb_valid = bwdb_st[(bwdb_st['Latitude'] >= 20.0) & (bwdb_st['Latitude'] <= 27.0)]
bwdb_typo = bwdb_st[bwdb_st['Latitude'] < 20.0]

# Scatter BWDB
sc_bwdb = ax.scatter(
    bwdb_valid['Longitude'], bwdb_valid['Latitude'],
    c='#1f77b4', s=35, alpha=0.75, edgecolors='none', label=f'BWDB Stations (N={len(bwdb_valid)})', zorder=3
)

# Scatter BMD
sc_bmd = ax.scatter(
    bmd_st['Longitude'], bmd_st['Latitude'],
    c='#d62728', s=70, marker='s', edgecolors='black', linewidth=1.0,
    label=f'BMD Stations (N={len(bmd_st)})', zorder=5
)

# Highlight close co-located pairs (<5 km)
close_pairs = metrics_df[metrics_df['Distance_km'] <= 5.0]
for idx, r in close_pairs.iterrows():
    bmd_match = bmd_st[bmd_st['Station'] == r['BMD_Station']]
    if len(bmd_match) > 0:
        lat = bmd_match['Latitude'].iloc[0]
        lon = bmd_match['Longitude'].iloc[0]
        circle = plt.Circle((lon, lat), 0.12, color='#2ca02c', fill=False, linestyle='--', linewidth=1.2, zorder=4)
        ax.add_patch(circle)

# Label selected prominent cities
key_cities = ['Dhaka', 'Sylhet', 'Chittagong', 'Rajshahi', 'Bogra', 'Khulna', 'Barisal', 'Rangpur', 'CoxsBazar']
for city in key_cities:
    m = bmd_st[bmd_st['Station'].str.contains(city, case=False, na=False)]
    if len(m) > 0:
        ax.annotate(
            city, (m['Longitude'].iloc[0], m['Latitude'].iloc[0]),
            xytext=(6, 6), textcoords='offset points', fontsize=9, fontweight='bold',
            color='#222222', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='#cccccc', linewidth=0.5),
            zorder=6
        )

# Mark Teknaf typo if visible or annotate inset
if len(bwdb_typo) > 0:
    for _, tr in bwdb_typo.iterrows():
        ax.annotate(
            f"Metadata Typo: {tr['Station ID']} ({tr['Station']})\nLat={tr['Latitude']:.2f} (in sea!)",
            xy=(tr['Longitude'], 20.3), xytext=(tr['Longitude'] - 1.2, 20.45),
            arrowprops=dict(facecolor='magenta', shrink=0.05, width=1.5, headwidth=6),
            fontsize=8, fontweight='bold', color='magenta',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6f0', edgecolor='magenta', linewidth=1)
        )

# Decorative grid and aesthetics
ax.set_xlim(87.8, 92.9)
ax.set_ylim(20.3, 26.8)
ax.set_xlabel('Longitude (°E)', fontsize=11, labelpad=8)
ax.set_ylabel('Latitude (°N)', fontsize=11, labelpad=8)
ax.grid(True, linestyle=':', alpha=0.6, color='gray')
ax.set_title('Spatial Distribution: BMD (Original) vs BWDB (New) Rainfall Stations\nBangladesh Observational Network Expansion', fontsize=12, fontweight='bold', pad=12)

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label=f'BMD Synoptic Stations (N={len(bmd_st)})', markerfacecolor='#d62728', markeredgecolor='black', markersize=9),
    Line2D([0], [0], marker='o', color='w', label=f'BWDB Hydrological Stations (N={len(bwdb_valid)})', markerfacecolor='#1f77b4', markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Co-located Pairs (<5 km, N=27)', markerfacecolor='none', markeredgecolor='#2ca02c', linestyle='--', markersize=11, markeredgewidth=1.5)
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.92, facecolor='#fafafa', edgecolor='#cccccc', fontsize=9.5)

# Annotation text box
text_box = (
    "Network Expansion Summary:\n"
    "• BMD Network: 42 synoptic/climate stations nationwide.\n"
    "• BWDB Network: 275 stations (266 active in data sheet).\n"
    "• 181 BWDB stations (66%) are >20 km from any BMD gauge.\n"
    "• Dense coverage in floodplains, river basins, haors, & coast."
)
ax.text(
    0.025, 0.03, text_box, transform=ax.transAxes, fontsize=8.5,
    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.92, edgecolor='#adb5bd', linewidth=0.8)
)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig1_spatial_station_distribution.png'), dpi=300)
plt.close(fig)

print("Plotting Figure 2: Distance Distribution & Correlation Decay...")
# FIG 2: Distance & Correlation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Distance histogram
ax1.hist(bwdb_st_dist['Dist_to_BMD_km'], bins=np.arange(0, 75, 5), color='#3a86ff', edgecolor='white', alpha=0.85, rwidth=0.9)
ax1.axvline(20.0, color='#e63946', linestyle='--', linewidth=1.5, label='20 km threshold (66% > 20 km)')
ax1.set_xlabel('Distance to Nearest BMD Station (km)', fontsize=10, labelpad=6)
ax1.set_ylabel('Number of BWDB Stations', fontsize=10, labelpad=6)
ax1.set_title('(a) Distance of BWDB Stations from Nearest BMD Station', fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right', fontsize=9)

# Correlation vs Distance
ax2.scatter(metrics_df['Distance_km'], metrics_df['r_lag0'], c='#0077b6', s=45, alpha=0.8, edgecolors='black', linewidth=0.5, label='Daily Lag-0 Correlation')
# Highlight Dhaka with lag+1
dhaka_row = metrics_df[metrics_df['BMD_Station'] == 'Dhaka']
if len(dhaka_row) > 0:
    ax2.scatter(dhaka_row['Distance_km'], dhaka_row['r_lag_p1'], c='#e63946', s=70, marker='^', zorder=5, label='Dhaka (Lag +1 day, r=0.82)')
    ax2.annotate('Dhaka (Lag 0: r=0.35\nLag +1: r=0.82)', (dhaka_row['Distance_km'].iloc[0], dhaka_row['r_lag0'].iloc[0]),
                 xytext=(15, -15), textcoords='offset points', fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1))

ax2.set_xlabel('Pairwise Distance (km)', fontsize=10, labelpad=6)
ax2.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=10, labelpad=6)
ax2.set_title('(b) Daily Rainfall Correlation vs Pair Distance', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.05, 1.0)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='lower left', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig2_distance_and_correlation.png'), dpi=300)
plt.close(fig)

print("Plotting Figure 3: Scatter Comparison for Close Pairs...")
# FIG 3: 6-Panel Scatter
selected_pairs = [
    ('Sylhet', 'CL128', 0.91, 'Sylhet (0.9 km)'),
    ('Srimangal', 'CL126', 1.72, 'Srimangal (1.7 km)'),
    ('Khepupara', 'CL269', 1.29, 'Khepupara (1.3 km)'),
    ('Bogra', 'CL6', 2.37, 'Bogra (2.4 km)'),
    ('Ambagan(Ctg)', 'CL306', 1.82, 'Ambagan Ctg (1.8 km)'),
    ('Dhaka', 'CL9', 1.56, 'Dhaka (1.6 km) - 1-Day Lag Shifted')
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=300)
axes = axes.flatten()

for i, (bmd_name, bwdb_id, dist, title) in enumerate(selected_pairs):
    ax = axes[i]
    if bmd_name not in bmd_data or bwdb_id not in bwdb_rf_idx.columns:
        continue
    
    s_bmd = bmd_data[bmd_name]
    s_bwdb = bwdb_rf_idx[bwdb_id]
    
    if bmd_name == 'Dhaka':
        # Apply the 1-day shift for Dhaka to demonstrate the exact match
        s_bwdb_plot = s_bwdb.shift(1)
        df_p = pd.DataFrame({'BMD': s_bmd, 'BWDB': s_bwdb_plot}).dropna()
        sub_title_extra = " [BWDB(t-1) vs BMD(t)]"
    else:
        df_p = pd.DataFrame({'BMD': s_bmd, 'BWDB': s_bwdb}).dropna()
        sub_title_extra = ""

    r = df_p['BMD'].corr(df_p['BWDB'])
    spearman = df_p['BMD'].corr(df_p['BWDB'], method='spearman')
    rmse = np.sqrt(((df_p['BMD'] - df_p['BWDB'])**2).mean())
    bias = (df_p['BWDB'] - df_p['BMD']).mean()
    
    max_val = max(df_p['BMD'].max(), df_p['BWDB'].max(), 100) * 1.05
    
    ax.scatter(df_p['BMD'], df_p['BWDB'], color='#0077b6', alpha=0.45, s=18, edgecolors='none')
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.2, label='1:1 Line')
    
    ax.set_xlim(-2, max_val)
    ax.set_ylim(-2, max_val)
    ax.set_xlabel('BMD Daily Rainfall (mm)', fontsize=9)
    ax.set_ylabel('BWDB Daily Rainfall (mm)', fontsize=9)
    ax.set_title(f"{title}{sub_title_extra}", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    info_str = f"r = {r:.3f}\nSpearman = {spearman:.3f}\nRMSE = {rmse:.1f} mm\nBias = {bias:+.2f} mm/d"
    ax.text(0.05, 0.93, info_str, transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc', linewidth=0.6))

plt.suptitle('Daily Rainfall Consistency for Closely Paired BMD vs BWDB Stations (2020–2024)', fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig3_close_pairs_scatter.png'), dpi=300)
plt.close(fig)

print("Plotting Figure 4: Time Series & Cumulative Curves...")
# FIG 4: Time series & Cumulative Rainfall
fig, axes = plt.subplots(4, 2, figsize=(14, 12), dpi=300, sharex='col')

demo_stations = [
    ('Sylhet', 'CL128', 'Sylhet (0.91 km) - Northeast Haor / Extreme Rain'),
    ('Srimangal', 'CL126', 'Srimangal (1.72 km) - Eastern Hills'),
    ('Bogra', 'CL6', 'Bogra (2.37 km) - Northern Floodplains'),
    ('Khepupara', 'CL269', 'Khepupara (1.29 km) - Coastal South')
]

for row_idx, (bmd_name, bwdb_id, title) in enumerate(demo_stations):
    ax_ts = axes[row_idx, 0]
    ax_cum = axes[row_idx, 1]
    
    s_bmd = bmd_data[bmd_name].loc['2022-05-01':'2022-09-30']
    s_bwdb = bwdb_rf_idx[bwdb_id].loc['2022-05-01':'2022-09-30']
    
    # Time series (Monsoon 2022 - landmark flood)
    ax_ts.plot(s_bmd.index, s_bmd.values, label='BMD', color='#d62728', linewidth=1.2, alpha=0.85)
    ax_ts.plot(s_bwdb.index, s_bwdb.values, label='BWDB', color='#1f77b4', linewidth=1.2, linestyle='--', alpha=0.85)
    ax_ts.set_ylabel('Rainfall (mm/d)', fontsize=8.5)
    ax_ts.set_title(f"{title}: Monsoon 2022 Daily Time Series", fontsize=9.5, fontweight='bold')
    ax_ts.grid(True, linestyle=':', alpha=0.4)
    if row_idx == 0:
        ax_ts.legend(loc='upper right', fontsize=8)
    
    # Cumulative curves (Full period 2020-2024)
    full_bmd = bmd_data[bmd_name].dropna()
    full_bwdb = bwdb_rf_idx[bwdb_id].reindex(full_bmd.index).dropna()
    common_idx = full_bmd.index.intersection(full_bwdb.index)
    
    cum_bmd = full_bmd.loc[common_idx].cumsum()
    cum_bwdb = full_bwdb.loc[common_idx].cumsum()
    
    ax_cum.plot(common_idx, cum_bmd.values / 1000.0, label='BMD Cumul.', color='#d62728', linewidth=1.5)
    ax_cum.plot(common_idx, cum_bwdb.values / 1000.0, label='BWDB Cumul.', color='#1f77b4', linewidth=1.5, linestyle='--')
    ax_cum.set_ylabel('Total Rain (meters)', fontsize=8.5)
    ax_cum.set_title(f"{title}: Cumulative Rainfall (2020–2024)", fontsize=9.5, fontweight='bold')
    ax_cum.grid(True, linestyle=':', alpha=0.4)
    if row_idx == 0:
        ax_cum.legend(loc='upper left', fontsize=8)

axes[3, 0].set_xlabel('Date (Monsoon 2022)', fontsize=9.5)
axes[3, 1].set_xlabel('Date (2020–2024)', fontsize=9.5)

plt.suptitle('Temporal Tracking & Cumulative Alignment: BMD vs BWDB at Colocated Sites', fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_timeseries_and_cumulative.png'), dpi=300)
plt.close(fig)

print("Plotting Figure 5: BWDB Data Completeness and Quality Audit...")
# FIG 5: Quality Audit & Yearly Availability
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Annual completeness
station_cols = [c for c in bwdb_rf.columns if c != 'Date']
bwdb_rf['Year'] = bwdb_rf['Date'].dt.year
annual_comp = []
for yr, grp in bwdb_rf.groupby('Year'):
    sub = grp[station_cols]
    valid_pct = (sub.notna().sum().sum() / sub.size) * 100
    rain_pct = ((sub > 0).sum().sum() / sub.notna().sum().sum()) * 100 if sub.notna().sum().sum() > 0 else 0
    annual_comp.append({'Year': yr, 'Valid_Pct': valid_pct, 'Rainy_Pct': rain_pct})

annual_df = pd.DataFrame(annual_comp)

ax1.bar(annual_df['Year'], annual_df['Valid_Pct'], color='#457b9d', edgecolor='white', width=0.8)
ax1.set_ylim(0, 105)
ax1.set_xlabel('Year', fontsize=10, labelpad=6)
ax1.set_ylabel('Data Completeness (%)', fontsize=10, labelpad=6)
ax1.set_title('(a) BWDB Annual Data Completeness (2000–2025)', fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')
ax1.axhline(90, color='orange', linestyle='--', linewidth=1, label='90% Completeness Benchmark')
ax1.legend(loc='lower left', fontsize=8.5)

# Monthly rainy days vs dry days
bwdb_rf['Month'] = bwdb_rf['Date'].dt.month
monthly_rain_prob = []
for mo, grp in bwdb_rf.groupby('Month'):
    sub = grp[station_cols]
    valid_cells = sub.notna().sum().sum()
    rainy_cells = (sub >= 1.0).sum().sum()
    heavy_cells = (sub >= 25.0).sum().sum()
    monthly_rain_prob.append({
        'Month': mo,
        'Rainy_Day_Prob': (rainy_cells / valid_cells) * 100,
        'Heavy_Rain_Prob': (heavy_cells / valid_cells) * 100
    })
mo_df = pd.DataFrame(monthly_rain_prob)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax2.plot(mo_df['Month'], mo_df['Rainy_Day_Prob'], marker='o', color='#1d3557', linewidth=2, label='Rain Day (≥1 mm)')
ax2.plot(mo_df['Month'], mo_df['Heavy_Rain_Prob'], marker='s', color='#e63946', linewidth=2, label='Heavy Rain (≥25 mm)')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names)
ax2.set_xlabel('Month of Year', fontsize=10, labelpad=6)
ax2.set_ylabel('Occurrence Frequency (%)', fontsize=10, labelpad=6)
ax2.set_title('(b) Climatological Seasonality (Monsoon vs Dry Season)', fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8.5)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_data_completeness_and_seasonality.png'), dpi=300)
plt.close(fig)

print("All 5 figures generated successfully in output/bwdb_analysis!")
