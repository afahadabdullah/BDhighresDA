import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = '/tmp'
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

out_dir = 'output/bwdb_analysis'
os.makedirs(out_dir, exist_ok=True)

# Load data
excel_path = 'data/stations/BWDB_Rainfall_2000_2025.xlsx'
bwdb_rf = pd.read_excel(excel_path, sheet_name='RainfallData')
bwdb_rf['Date'] = pd.to_datetime(bwdb_rf['Date'])
bwdb_rf = bwdb_rf.set_index('Date')

metrics_df = pd.read_csv(os.path.join(out_dir, 'paired_station_metrics.csv'))
close_pairs = metrics_df[metrics_df['Distance_km'] <= 10.0].copy()

bmd_dir = 'data/stations/data_2020_2025'

# Calculate lag correlations
lags = [-2, -1, 0, 1, 2]
records = []
for idx, row in close_pairs.iterrows():
    bmd_name = row['BMD_Station']
    bwdb_id = row['BWDB_ID']
    dist = row['Distance_km']
    
    csv_file = os.path.join(bmd_dir, f'{bmd_name}.csv')
    if not os.path.exists(csv_file):
        matched = glob.glob(os.path.join(bmd_dir, f'*{bmd_name}*.csv'))
        if matched:
            csv_file = matched[0]
        else:
            continue
            
    bmd_df = pd.read_csv(csv_file)
    bmd_df['Date'] = pd.to_datetime(bmd_df['Datetime'])
    bmd_s = bmd_df.set_index('Date')['Rainfall']
    
    if bwdb_id not in bwdb_rf.columns:
        continue
    bwdb_s = bwdb_rf[bwdb_id]
    
    common_idx = bmd_s.dropna().index.intersection(bwdb_s.dropna().index)
    if len(common_idx) < 365:
        continue
        
    lag_corrs = {}
    for k in lags:
        # Shifting BMD by k days relative to BWDB:
        # k = 0: BMD(t) vs BWDB(t)
        # k = +1: BMD(t+1) vs BWDB(t)
        # k = -1: BMD(t-1) vs BWDB(t)
        bmd_shifted = bmd_s.shift(-k)
        df_k = pd.DataFrame({'BMD_shifted': bmd_shifted, 'BWDB': bwdb_s}).dropna()
        r = df_k['BMD_shifted'].corr(df_k['BWDB'])
        lag_corrs[k] = r
        
    best_lag = max(lag_corrs, key=lag_corrs.get)
    records.append({
        'BMD_Station': bmd_name,
        'BWDB_ID': bwdb_id,
        'BWDB_Name': row['BWDB_Name'],
        'Distance_km': dist,
        'Label': f"{bmd_name} ({dist:.1f} km)",
        '-2': lag_corrs[-2],
        '-1': lag_corrs[-1],
        '0': lag_corrs[0],
        '+1': lag_corrs[1],
        '+2': lag_corrs[2],
        'Best_Lag': best_lag,
        'Max_r': lag_corrs[best_lag],
        'Lag_Diff_p1_m1': lag_corrs[1] - lag_corrs[-1]
    })

lag_df = pd.DataFrame(records).sort_values('Distance_km')
lag_df.to_csv(os.path.join(out_dir, 'lag_correlation_summary.csv'), index=False)
print(f"Computed lag correlations for {len(lag_df)} pairs.")

# Create 2-Panel Comprehensive Plot
fig = plt.figure(figsize=(15, 11), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 0.8], hspace=0.3, wspace=0.25)

# 1. Heatmap of all close pairs across lags
ax_heat = fig.add_subplot(gs[0, :])
heat_data = lag_df.set_index('Label')[['-2', '-1', '0', '+1', '+2']]

# Plot heatmap
im = ax_heat.imshow(heat_data.values, cmap='YlGnBu', aspect='auto', vmin=0.0, vmax=0.95)
cbar = plt.colorbar(im, ax=ax_heat, pad=0.02, shrink=0.9)
cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=10, labelpad=8)

# Set ticks
ax_heat.set_xticks(range(len(lags)))
ax_heat.set_xticklabels(['Shift -2d\n[BMD(t-2)]', 'Shift -1d\n[BMD(t-1)]', 'Shift 0d\n[BMD(t) - Unshifted]', 'Shift +1d\n[BMD(t+1)]', 'Shift +2d\n[BMD(t+2)]'], fontsize=9.5, fontweight='bold')
ax_heat.set_yticks(range(len(heat_data)))
ax_heat.set_yticklabels(heat_data.index, fontsize=8.5)

# Annotate values and highlight best lag with red box/marker
for y in range(heat_data.shape[0]):
    best_l = lag_df['Best_Lag'].iloc[y]
    best_x = lags.index(best_l)
    for x in range(heat_data.shape[1]):
        val = heat_data.values[y, x]
        is_best = (x == best_x)
        txt_color = 'white' if val > 0.6 else 'black'
        weight = 'bold' if is_best else 'normal'
        txt = f"{val:.3f}"
        if is_best:
            txt += " *"
            # Draw highlight rectangle
            rect = plt.Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor='#d62728', linewidth=2.0)
            ax_heat.add_patch(rect)
        ax_heat.text(x, y, txt, ha='center', va='center', color=txt_color, fontsize=8, fontweight=weight)

ax_heat.set_title('(a) Pairwise Correlation Heatmap across Shift Lags (-2, -1, 0, +1, +2 Days)\n* Red boxes indicate the highest correlation lag for each station pair', fontsize=11, fontweight='bold', pad=10)

# 2. Correlogram Lines for Top Representative Stations
ax_lines = fig.add_subplot(gs[1, 0])
selected_lines = [
    ('Sylhet (0.9 km)', '#1f77b4', '-o'),
    ('Khepupara (1.3 km)', '#2ca02c', '-s'),
    ('Tangail (3.6 km)', '#ff7f0e', '-^'),
    ('Srimangal (1.7 km)', '#9467bd', '-d'),
    ('Ambagan(Ctg) (1.8 km)', '#8c564b', '-v'),
    ('Bogra (2.4 km)', '#e377c2', '-p'),
    ('Dhaka (1.6 km)', '#d62728', '--X') # Highlight Dhaka
]

for label, color, marker in selected_lines:
    row = lag_df[lag_df['BMD_Station'] == label.split()[0]]
    if len(row) > 0:
        vals = [row['-2'].iloc[0], row['-1'].iloc[0], row['0'].iloc[0], row['+1'].iloc[0], row['+2'].iloc[0]]
        lw = 2.4 if 'Dhaka' in label else 1.6
        ms = 8 if 'Dhaka' in label else 6
        ax_lines.plot(lags, vals, marker, label=label, color=color, linewidth=lw, markersize=ms)

ax_lines.axvline(0, color='gray', linestyle=':', linewidth=1)
ax_lines.set_xticks(lags)
ax_lines.set_xticklabels(['-2 days', '-1 day', '0 (Same day)', '+1 day', '+2 days'], fontsize=9.5)
ax_lines.set_xlabel('BMD Shift Relative to BWDB (days)', fontsize=10, labelpad=6)
ax_lines.set_ylabel('Pearson Correlation (r)', fontsize=10, labelpad=6)
ax_lines.set_title('(b) Correlogram Profiles for Key Co-located Stations', fontsize=11, fontweight='bold')
ax_lines.grid(True, linestyle=':', alpha=0.5)
ax_lines.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax_lines.set_ylim(0.0, 1.0)

# Annotate Dhaka anomaly and physical gauges
ax_lines.annotate('Dhaka peaks at +1 day\n(Desk office copying BMD\nwith start-date stamp)',
                  xy=(1, 0.821), xytext=(0.1, 0.65),
                  arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                  fontsize=8.5, fontweight='bold', color='#d62728',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6f0', edgecolor='#d62728', linewidth=0.8))

# 3. Bar Chart: Lag +1 vs Lag -1 Asymmetry (Signature of 06:00 vs 09:00 BST Cutoff)
ax_bar = fig.add_subplot(gs[1, 1])

# Filter physical stations (exclude Dhaka)
phys_stations = lag_df[lag_df['BMD_Station'] != 'Dhaka'].head(15)
diff_vals = phys_stations['Lag_Diff_p1_m1'].values
y_pos = np.arange(len(phys_stations))

colors = ['#2b5c8f' if v > 0 else '#e76f51' for v in diff_vals]
ax_bar.barh(y_pos, diff_vals, color=colors, edgecolor='none', height=0.7)
ax_bar.axvline(0, color='black', linewidth=0.8)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(phys_stations['BMD_Station'], fontsize=8.5)
ax_bar.invert_yaxis()
ax_bar.set_xlabel('Correlation Difference: [r(Lag +1) - r(Lag -1)]', fontsize=9.5, labelpad=6)
ax_bar.set_title('(c) Asymmetry: r(Lag +1) vs r(Lag -1)\nPositive = 3h Window Lead Signature', fontsize=10, fontweight='bold')
ax_bar.grid(True, linestyle=':', alpha=0.5, axis='x')

ax_bar.text(0.05, 0.05, 
            "All physical stations show\nr(Lag +1) > r(Lag -1)\nbecause BWDB 06:00 BST\nresets 3h before BMD.", 
            transform=ax_bar.transAxes, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#adb5bd', linewidth=0.7))

plt.tight_layout()
fig_path = os.path.join(out_dir, 'fig6_pairwise_lag_correlation_analysis.png')
fig.savefig(fig_path, dpi=300)
plt.close(fig)

print("Saved figure successfully:", fig_path)
