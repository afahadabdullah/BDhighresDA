import os
import pandas as pd
import datetime

print("Loading original BWDB_Rainfall_2000_2025.xlsx...")
src_file = 'data/stations/BWDB_Rainfall_2000_2025.xlsx'
target_file = 'data/stations/BWDB_Rainfall_2000_2025_correct.xlsx'

xls = pd.ExcelFile(src_file)
df_rf = pd.read_excel(xls, sheet_name='RainfallData')
df_st = pd.read_excel(xls, sheet_name='StationList')
df_failed = pd.read_excel(xls, sheet_name='FailedStations')

print(f"Loaded RainfallData shape: {df_rf.shape}")
print(f"Loaded StationList shape: {df_st.shape}")
print(f"Loaded FailedStations shape: {df_failed.shape}")

# Shift CL9 (Dhaka) forward by 1 day
# CL9 at day t receives the value from day t-1
df_rf_correct = df_rf.copy()
df_rf_correct['CL9'] = df_rf_correct['CL9'].shift(1)

# Verify shift
print("\nVerifying shift on sample dates in May 2020:")
sample_orig = df_rf.loc[df_rf['Date'].astype(str).str.startswith('2020-05-0'), ['Date', 'CL9']].head(6)
sample_corr = df_rf_correct.loc[df_rf_correct['Date'].astype(str).str.startswith('2020-05-0'), ['Date', 'CL9']].head(6)
print("Original CL9 (Dhaka):")
print(sample_orig.to_string(index=False))
print("Shifted CL9 (Dhaka):")
print(sample_corr.to_string(index=False))

# Write to Excel with all 3 sheets preserved
print(f"\nWriting corrected dataset to {target_file}...")
with pd.ExcelWriter(target_file, engine='openpyxl', date_format='YYYY-MM-DD', datetime_format='YYYY-MM-DD') as writer:
    df_rf_correct.to_excel(writer, sheet_name='RainfallData', index=False)
    df_st.to_excel(writer, sheet_name='StationList', index=False)
    df_failed.to_excel(writer, sheet_name='FailedStations', index=False)

print(f"Successfully created {target_file} ({os.path.getsize(target_file)} bytes)")

# Also create BWDB_Rainfall_2000_2025_corrected.xlsx alias
alias_file = 'data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx'
import shutil
shutil.copyfile(target_file, alias_file)
print(f"Also created alias at {alias_file}")
