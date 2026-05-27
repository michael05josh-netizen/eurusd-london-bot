import pandas as pd
import numpy as np
import zipfile
import os
import glob

TICK_DIR = "data/raw/ticks"
OUT_PATH = "data/raw/EURUSD_tick_features_2023_2025.csv"

os.makedirs("data/raw", exist_ok=True)

def extract_tick_features(df_ticks, freq='5min'):
    """
    Convert raw ticks to M5 bars with microstructure features.
    df_ticks columns: datetime, bid, ask, bid_vol, ask_vol (or similar)
    """
    df_ticks = df_ticks.sort_index()

    # Mid price
    df_ticks['mid']    = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['spread'] = df_ticks['ask'] - df_ticks['bid']

    # Tick direction: +1 if mid went up, -1 if down, 0 if same
    df_ticks['tick_dir'] = np.sign(df_ticks['mid'].diff()).fillna(0)

    # Up tick vs down tick
    df_ticks['is_uptick']   = (df_ticks['tick_dir'] > 0).astype(int)
    df_ticks['is_downtick'] = (df_ticks['tick_dir'] < 0).astype(int)

    # Resample to M5
    ohlc = df_ticks['mid'].resample(freq).ohlc()
    ohlc.columns = ['open', 'high', 'low', 'close']

    features = pd.DataFrame(index=ohlc.index)
    features['open']  = ohlc['open']
    features['high']  = ohlc['high']
    features['low']   = ohlc['low']
    features['close'] = ohlc['close']

    # --- Spread features ---
    features['spread_mean'] = df_ticks['spread'].resample(freq).mean()
    features['spread_std']  = df_ticks['spread'].resample(freq).std()
    features['spread_max']  = df_ticks['spread'].resample(freq).max()
    features['spread_min']  = df_ticks['spread'].resample(freq).min()

    # --- Volume / tick count (real activity proxy) ---
    features['tick_count']  = df_ticks['mid'].resample(freq).count()

    # Tick count rolling mean for normalization
    features['tick_count_ma'] = features['tick_count'].rolling(20).mean()
    features['tick_ratio']    = features['tick_count'] / features['tick_count_ma'].replace(0, np.nan)

    # --- Tick direction features ---
    features['up_tick_count']   = df_ticks['is_uptick'].resample(freq).sum()
    features['down_tick_count'] = df_ticks['is_downtick'].resample(freq).sum()
    features['tick_imbalance']  = (
        (features['up_tick_count'] - features['down_tick_count']) /
        features['tick_count'].replace(0, np.nan)
    )  # +1 = all upticks, -1 = all downticks

    # --- Tick velocity (ticks per second = urgency) ---
    # Compute time span of each 5min bar
    tick_times = df_ticks.index.to_series().resample(freq)
    features['bar_duration_s'] = (
        tick_times.last() - tick_times.first()
    ).dt.total_seconds().replace(0, np.nan)
    features['tick_velocity']  = (
        features['tick_count'] / features['bar_duration_s']
    )  # ticks per second

    # --- Price acceleration (momentum within bar) ---
    # Split bar into first/second half tick direction
    def half_imbalance(x, first_half=True):
        if len(x) < 2:
            return 0
        half = len(x) // 2
        subset = x[:half] if first_half else x[half:]
        return subset.mean() if len(subset) > 0 else 0

    features['tick_dir_first'] = df_ticks['tick_dir'].resample(freq).apply(
        lambda x: half_imbalance(x.values, True)
    )
    features['tick_dir_last']  = df_ticks['tick_dir'].resample(freq).apply(
        lambda x: half_imbalance(x.values, False)
    )
    features['tick_accel'] = features['tick_dir_last'] - features['tick_dir_first']

    # --- Spread spike (widens before big moves) ---
    spread_ma = features['spread_mean'].rolling(20).mean()
    features['spread_ratio'] = features['spread_mean'] / spread_ma.replace(0, np.nan)
    features['spread_spike'] = (features['spread_ratio'] > 1.5).astype(int)

    # --- Large tick bars (institutional activity) ---
    tick_ma = features['tick_count'].rolling(20).mean()
    features['is_high_activity'] = (
        features['tick_count'] > tick_ma * 1.5
    ).astype(int)

    # --- Buy/sell pressure from spread position ---
    # If close is near ask → buying pressure; near bid → selling
    features['buy_pressure'] = (
        (df_ticks['mid'].resample(freq).last() - df_ticks['bid'].resample(freq).last()) /
        df_ticks['spread'].resample(freq).last().replace(0, np.nan)
    )

    return features.dropna(how='all')


# ================================================================
# PROCESS ALL MONTHS
# ================================================================
all_features = []

zip_files = sorted(glob.glob(f"{TICK_DIR}/DAT_ASCII_EURUSD_T_202*.zip"))
print(f"Found {len(zip_files)} ZIP files\n")

for zip_path in zip_files:
    month_label = zip_path[-10:-4]  # e.g. 202501
    print(f"Processing {month_label}...")

    try:
        # --- Extract and read CSV from ZIP ---
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df_raw = pd.read_csv(
                    f,
                    header=None,
                    names=['datetime', 'bid', 'ask', 'bid_vol', 'ask_vol'],
                    dtype={'bid': np.float64, 'ask': np.float64,
                           'bid_vol': np.float64, 'ask_vol': np.float64}
                )

        print(f"  Raw ticks: {len(df_raw):,}")

        # --- Parse datetime ---
        df_raw['datetime'] = pd.to_datetime(
            df_raw['datetime'], format='%Y%m%d %H%M%S%f'
        )
        df_raw.set_index('datetime', inplace=True)
        df_raw.index = df_raw.index.tz_localize('UTC')

        print(f"  Range: {df_raw.index[0]} → {df_raw.index[-1]}")
        print(f"  Spread sample (pips): {(df_raw['ask'] - df_raw['bid']).mean()*10000:.2f}")

        # --- Extract M5 features ---
        features = extract_tick_features(df_raw, freq='5min')
        print(f"  M5 bars extracted: {len(features):,}")

        all_features.append(features)

        # Free memory
        del df_raw

    except Exception as e:
        print(f"  [ERROR] {month_label}: {e}")
        import traceback
        traceback.print_exc()

# ================================================================
# COMBINE AND SAVE
# ================================================================
print("\nCombining all months...")
df_final = pd.concat(all_features).sort_index()
df_final = df_final[~df_final.index.duplicated(keep='first')]

print(f"Total M5 bars: {len(df_final):,}")
print(f"Range: {df_final.index[0]} → {df_final.index[-1]}")
print(f"Features: {df_final.shape[1]} columns")
print(f"\nFeature list:")
print(list(df_final.columns))

# Check key features
print(f"\nSample stats:")
print(f"  Avg spread (pips): {df_final['spread_mean'].mean()*10000:.2f}")
print(f"  Avg tick count/bar: {df_final['tick_count'].mean():.0f}")
print(f"  Avg tick velocity:  {df_final['tick_velocity'].mean():.1f} ticks/sec")
print(f"  Tick imbalance mean: {df_final['tick_imbalance'].mean():.4f}")

df_final.to_csv(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")