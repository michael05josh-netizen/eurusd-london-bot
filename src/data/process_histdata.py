import pandas as pd
import os

PAIR    = "EURUSD"
IN_DIR  = "data/raw/histdata"
OUT_DIR = "data/raw"

os.makedirs(OUT_DIR, exist_ok=True)

def load_histdata_csv(path):
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"]
    )
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%Y.%m.%d %H:%M'
    )
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time'], inplace=True)
    df.index = df.index.tz_localize('UTC')
    return df[['open', 'high', 'low', 'close', 'volume']]

def resample_to_m5(df):
    return df.resample('5min').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }).dropna()

# --- Build file list ---
files = [
    f"{IN_DIR}/EURUSD_2022.csv",
    f"{IN_DIR}/EURUSD_2023.csv",
    f"{IN_DIR}/EURUSD_2024.csv",
    f"{IN_DIR}/EURUSD_2025.csv",
    f"{IN_DIR}/EURUSD_2026_01.csv",
    f"{IN_DIR}/EURUSD_2026_02.csv",
    f"{IN_DIR}/EURUSD_2026_03.csv",
    f"{IN_DIR}/EURUSD_2026_04.csv",
]

# --- Load and merge ---
print(f"Processing {PAIR}...")
frames = []

for path in files:
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        continue
    df = load_histdata_csv(path)
    print(f"  Loaded {os.path.basename(path)}: {len(df):,} M1 candles")
    frames.append(df)

if not frames:
    print("No data loaded. Check your file names and paths.")
    exit()

# --- Combine ---
df_all = pd.concat(frames).sort_index()
df_all = df_all[~df_all.index.duplicated(keep='first')]
print(f"\nTotal M1 candles: {len(df_all):,}")
print(f"Range: {df_all.index[0]} → {df_all.index[-1]}")

# --- Resample to M5 ---
df_m5 = resample_to_m5(df_all)
print(f"Resampled to M5: {len(df_m5):,} candles")

# --- Save ---
out_path = f"{OUT_DIR}/EURUSD_M5.csv"
df_m5.to_csv(out_path)
print(f"Saved → {out_path}")