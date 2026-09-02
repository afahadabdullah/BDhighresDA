import os
import glob
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = '/tmp'

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0 # km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

print("--- Step 1: Load and Audit BWDB metadata and stations ---")
excel_path = 'data/stations/BWDB_Rainfall_2000_2025.xlsx'
xls = pd.ExcelFile(excel_path)
print("Sheets:", xls.sheet_names)

bwdb_st = pd.read_excel(xls, sheet_name='StationList')
failed_st = pd.read_excel(xls, sheet_name='FailedStations')

print(f"Total BWDB stations in StationList: {len(bwdb_st)}")
print(f"Total FailedStations: {len(failed_st)}")

# Lat/lon bounding box audit
print("BWDB Latitude range:", bwdb_st['Latitude'].min(), "to", bwdb_st['Latitude'].max())
print("BWDB Longitude range:", bwdb_st['Longitude'].min(), "to", bwdb_st['Longitude'].max())
out_of_bounds = bwdb_st[(bwdb_st['Latitude'] < 20.0) | (bwdb_st['Latitude'] > 27.0) | 
                        (bwdb_st['Longitude'] < 88.0) | (bwdb_st['Longitude'] > 93.0)]
print("Out of bounds stations count:", len(out_of_bounds))
if len(out_of_bounds) > 0:
    print(out_of_bounds[['Station ID', 'Station', 'District', 'Latitude', 'Longitude']])

print("\n--- Step 2: Load BWDB Rainfall Data ---")
bwdb_rf = pd.read_excel(xls, sheet_name='RainfallData')
bwdb_rf['Date'] = pd.to_datetime(bwdb_rf['Date'])
date_min = bwdb_rf['Date'].min()
date_max = bwdb_rf['Date'].max()
print(f"BWDB Date range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')} ({len(bwdb_rf)} days)")

station_cols = [c for c in bwdb_rf.columns if c != 'Date']
print(f"Active station columns in RainfallData: {len(station_cols)}")

sub_rf = bwdb_rf[station_cols]
neg_count = (sub_rf < 0).sum().sum()
extreme_500 = (sub_rf > 500).sum().sum()
extreme_1000 = (sub_rf > 1000).sum().sum()
max_val = sub_rf.max().max()
print(f"Negative values: {neg_count}")
print(f"Values > 500 mm/day: {extreme_500}")
print(f"Values > 1000 mm/day: {extreme_1000}")
print(f"Maximum value recorded: {max_val:.1f} mm")

# Top 10 highest rainfall events in BWDB
top_events = []
for c in station_cols:
    s = bwdb_rf[['Date', c]].dropna()
    top = s.nlargest(3, c)
    for _, r in top.iterrows():
        top_events.append({'Date': r['Date'].strftime('%Y-%m-%d'), 'Station_ID': c, 'Rainfall_mm': r[c]})
top_events_df = pd.DataFrame(top_events).sort_values('Rainfall_mm', ascending=False).drop_duplicates(['Date', 'Station_ID'])
top_events_df = top_events_df.merge(bwdb_st[['Station ID', 'Station', 'District']], left_on='Station_ID', right_on='Station ID', how='left')
print("\nTop 10 daily rainfall events in BWDB:")
print(top_events_df[['Date', 'Station_ID', 'Station', 'District', 'Rainfall_mm']].head(10).to_string(index=False))

print("\n--- Step 3: Load BMD Stations and CSVs ---")
bmd_stations = pd.read_csv('data/stations/data_2020_2025/Stations.csv').dropna(subset=['Latitude', 'Longitude'])
print(f"BMD stations in catalogue: {len(bmd_stations)}")

bmd_dir = 'data/stations/data_2020_2025'
bmd_data = {}
for idx, r in bmd_stations.iterrows():
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

print(f"Loaded BMD station time series: {len(bmd_data)}")

print("\n--- Step 4: Spatial Distance Matching ---")
nearest_bmd_dist = []
for idx, row in bwdb_st.iterrows():
    dists = haversine_np(row['Latitude'], row['Longitude'], bmd_stations['Latitude'].values, bmd_stations['Longitude'].values)
    nearest_bmd_dist.append(dists.min())
bwdb_st['Dist_to_BMD_km'] = nearest_bmd_dist

print("BWDB distance to nearest BMD station summary:")
print(bwdb_st['Dist_to_BMD_km'].describe())
print(f"BWDB stations within 5 km of a BMD station: {(bwdb_st['Dist_to_BMD_km'] <= 5).sum()}")
print(f"BWDB stations within 10 km of a BMD station: {(bwdb_st['Dist_to_BMD_km'] <= 10).sum()}")
print(f"BWDB stations within 20 km of a BMD station: {(bwdb_st['Dist_to_BMD_km'] <= 20).sum()}")
print(f"BWDB stations > 20 km away (genuinely new locations): {(bwdb_st['Dist_to_BMD_km'] > 20).sum()}")
print(f"BWDB stations > 40 km away (remote regional locations): {(bwdb_st['Dist_to_BMD_km'] > 40).sum()}")

bmd_pairs = []
for idx, row in bmd_stations.iterrows():
    dists = haversine_np(row['Latitude'], row['Longitude'], bwdb_st['Latitude'].values, bwdb_st['Longitude'].values)
    min_i = np.argmin(dists)
    min_d = dists[min_i]
    bmd_pairs.append({
        'BMD_Station': row['Station'],
        'BMD_Lat': row['Latitude'],
        'BMD_Lon': row['Longitude'],
        'BWDB_ID': bwdb_st.iloc[min_i]['Station ID'],
        'BWDB_Name': bwdb_st.iloc[min_i]['Station'],
        'BWDB_District': bwdb_st.iloc[min_i]['District'],
        'BWDB_Lat': bwdb_st.iloc[min_i]['Latitude'],
        'BWDB_Lon': bwdb_st.iloc[min_i]['Longitude'],
        'Distance_km': round(min_d, 2)
    })
bmd_pairs_df = pd.DataFrame(bmd_pairs).sort_values('Distance_km')

print("\n--- Step 5: Consistency Evaluation for Close Pairs ---")
bwdb_rf_indexed = bwdb_rf.set_index('Date')
metrics_list = []
for idx, p in bmd_pairs_df.iterrows():
    bmd_st_name = p['BMD_Station']
    bwdb_id = p['BWDB_ID']
    dist = p['Distance_km']
    
    if bmd_st_name not in bmd_data or bwdb_id not in bwdb_rf_indexed.columns:
        continue
    
    s_bmd = bmd_data[bmd_st_name]
    s_bwdb = bwdb_rf_indexed[bwdb_id]
    
    df_c = pd.DataFrame({'BMD': s_bmd, 'BWDB': s_bwdb}).dropna()
    if len(df_c) < 365:
        continue
    
    n_days = len(df_c)
    
    r0 = df_c['BMD'].corr(df_c['BWDB'])
    r_p1 = df_c['BMD'].iloc[1:].corr(pd.Series(df_c['BWDB'].iloc[:-1].values, index=df_c.index[1:]))
    r_m1 = df_c['BMD'].iloc[:-1].corr(pd.Series(df_c['BWDB'].iloc[1:].values, index=df_c.index[:-1]))
    
    opt_lag = 0
    max_r = r0
    if r_p1 > max_r and r_p1 > 0.7:
        opt_lag = 1
        max_r = r_p1
    elif r_m1 > max_r and r_m1 > 0.7:
        opt_lag = -1
        max_r = r_m1
    
    mae0 = (df_c['BMD'] - df_c['BWDB']).abs().mean()
    rmse0 = np.sqrt(((df_c['BMD'] - df_c['BWDB'])**2).mean())
    bias0 = (df_c['BWDB'] - df_c['BMD']).mean()
    sum_bmd = df_c['BMD'].sum()
    sum_bwdb = df_c['BWDB'].sum()
    
    bmd_rain = df_c['BMD'] >= 1.0
    bwdb_rain = df_c['BWDB'] >= 1.0
    concordance = (bmd_rain == bwdb_rain).mean()
    spearman = df_c['BMD'].corr(df_c['BWDB'], method='spearman')
    
    metrics_list.append({
        'BMD_Station': bmd_st_name,
        'BWDB_ID': bwdb_id,
        'BWDB_Name': p['BWDB_Name'],
        'District': p['BWDB_District'],
        'Distance_km': dist,
        'N_days': n_days,
        'r_lag0': round(r0, 3) if not np.isnan(r0) else 0.0,
        'r_lag_p1': round(r_p1, 3) if not np.isnan(r_p1) else 0.0,
        'r_lag_m1': round(r_m1, 3) if not np.isnan(r_m1) else 0.0,
        'Opt_Lag': opt_lag,
        'Max_r': round(max_r, 3) if not np.isnan(max_r) else 0.0,
        'Spearman_r': round(spearman, 3) if not np.isnan(spearman) else 0.0,
        'Concordance_pct': round(concordance * 100, 1),
        'BMD_Sum_mm': round(sum_bmd, 1),
        'BWDB_Sum_mm': round(sum_bwdb, 1),
        'Sum_Ratio': round(sum_bwdb / sum_bmd, 3) if sum_bmd > 0 else np.nan,
        'Bias_mm_day': round(bias0, 2),
        'MAE_mm_day': round(mae0, 2),
        'RMSE_mm_day': round(rmse0, 2)
    })

metrics_df = pd.DataFrame(metrics_list).sort_values('Distance_km')
print(f"\nAnalyzed {len(metrics_df)} matched pairs. Top 20 by distance:")
print(metrics_df.head(20).to_string(index=False))

os.makedirs('output/bwdb_analysis', exist_ok=True)
metrics_df.to_csv('output/bwdb_analysis/paired_station_metrics.csv', index=False)
bwdb_st.to_csv('output/bwdb_analysis/bwdb_stations_with_distance.csv', index=False)
print("\nSaved output/bwdb_analysis/paired_station_metrics.csv and bwdb_stations_with_distance.csv successfully!")
