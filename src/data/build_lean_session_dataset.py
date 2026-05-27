import pandas as pd
import numpy as np
import os

TICK_PATH = "data/raw/EURUSD_tick_features_2023_2025.csv"
M5_PATH   = "data/raw/EURUSD_M5_clean.csv"
H1_PATH   = "data/raw/EURUSD_H1.csv"
H4_PATH   = "data/raw/EURUSD_H4.csv"
OUT_PATH  = "data/featured/EURUSD_lean_session.csv"

os.makedirs("data/featured", exist_ok=True)

# ================================================================
# LOAD
# ================================================================
print("Loading data...")
df_tick = pd.read_csv(TICK_PATH, index_col=0, parse_dates=True)
df_m5   = pd.read_csv(M5_PATH,   index_col=0, parse_dates=True)
df_h1   = pd.read_csv(H1_PATH,   index_col=0, parse_dates=True)
df_h4   = pd.read_csv(H4_PATH,   index_col=0, parse_dates=True)

start = df_tick.index[0]
end   = df_tick.index[-1]
print(f"Range: {start} → {end}")

df_m5 = df_m5[(df_m5.index >= start) & (df_m5.index <= end)].copy()
df_h1 = df_h1[(df_h1.index >= start) & (df_h1.index <= end)].copy()
df_h4 = df_h4[(df_h4.index >= start) & (df_h4.index <= end)].copy()

print(f"M5: {len(df_m5):,}  H1: {len(df_h1):,}  "
      f"H4: {len(df_h4):,}  Tick: {len(df_tick):,}")

# ================================================================
# RESAMPLE TICK TO H1
# ================================================================
print("\nResampling tick to H1...")
tick_h1 = df_tick.resample('1h').agg({
    'spread_mean':    'mean',
    'tick_count':     'sum',
    'tick_imbalance': 'mean',
    'tick_velocity':  'mean',
    'buy_pressure':   'mean',
    'up_tick_count':  'sum',
    'down_tick_count':'sum',
    'spread_ratio':   'mean',
}).dropna()

tick_h1['h1_tick_imbal'] = (
    (tick_h1['up_tick_count'] - tick_h1['down_tick_count']) /
    tick_h1['tick_count'].replace(0, np.nan)
)
# Rolling 3h context — top features from importance analysis
tick_h1['tick_imbal_3h'] = tick_h1['h1_tick_imbal'].rolling(3).mean()
tick_h1['spread_3h']     = tick_h1['spread_mean'].rolling(3).mean()
tick_h1['tick_vol_3h']   = tick_h1['tick_count'].rolling(3).mean()
tick_h1['buy_pres_3h']   = tick_h1['buy_pressure'].rolling(3).mean()

# ================================================================
# SELECT LEAN H1 FEATURES
# Only columns that ranked in top importance
# ================================================================
H1_KEEP = [
    'open', 'high', 'low', 'close',
    'H1_rsi_14', 'H1_rsi_slope', 'H1_atr_14', 'H1_atr_ratio',
    'H1_atr_vs_mean', 'H1_macd', 'H1_macd_diff', 'H1_macd_signal',
    'H1_bb_position', 'H1_bb_width', 'H1_ema_cross_8_21',
    'H1_range_sma_20', 'H1_range_ratio', 'H1_body_lag_1',
    'H1_body_lag_2', 'H1_dist_high_20', 'H1_dist_low_20',
]

H4_KEEP = [
    'H4_rsi_14', 'H4_rsi_slope', 'H4_atr_ratio', 'H4_atr_vs_mean',
    'H4_bb_width', 'H4_bb_position', 'H4_macd_diff', 'H4_macd_signal',
    'H4_roc_10', 'H4_range_ratio', 'H4_body_lag_3', 'H4_lower_wick',
    'H4_ema_cross_8_21', 'H4_trend_aligned',
]

TICK_KEEP = [
    'tick_imbal_3h', 'spread_3h', 'tick_vol_3h',
    'buy_pres_3h', 'tick_velocity', 'buy_pressure',
]

# Filter to existing columns
h1_cols  = [c for c in H1_KEEP  if c in df_h1.columns]
h4_cols  = [c for c in H4_KEEP  if c in df_h4.columns]
tick_cols = [c for c in TICK_KEEP if c in tick_h1.columns]

print(f"\nSelected features — H1: {len(h1_cols)}  "
      f"H4: {len(h4_cols)}  Tick: {len(tick_cols)}")

# ================================================================
# ALIGN TO H1 INDEX
# ================================================================
h1 = df_h1[h1_cols].copy()

h4_sub = df_h4[[c for c in h4_cols if c in df_h4.columns]].copy()
h4_sub = h4_sub.shift(1)
h4_aligned = h4_sub.reindex(h1.index, method='ffill')

tick_sub    = tick_h1[[c for c in tick_cols if c in tick_h1.columns]].copy()
tick_aligned = tick_sub.reindex(h1.index, method='ffill')

# Time features
time_df = pd.DataFrame(index=h1.index)
time_df['hour']        = h1.index.hour
time_df['day_of_week'] = h1.index.dayofweek
time_df['sin_hour']    = np.sin(2 * np.pi * time_df['hour'] / 24)
time_df['cos_hour']    = np.cos(2 * np.pi * time_df['hour'] / 24)
time_df['sin_dow']     = np.sin(2 * np.pi * time_df['day_of_week'] / 5)
time_df['cos_dow']     = np.cos(2 * np.pi * time_df['day_of_week'] / 5)
time_df['is_monday']   = (time_df['day_of_week'] == 0).astype(int)
time_df['is_thursday'] = (time_df['day_of_week'] == 3).astype(int)

# Concat
df_full = pd.concat([h1, h4_aligned, tick_aligned, time_df], axis=1)
df_full = df_full.loc[:, ~df_full.columns.duplicated(keep='first')]
df_full = df_full.copy()

# Drop bad cols
nan_counts = df_full.isnull().sum()
thresh     = int(len(df_full) * 0.05)
all_nan    = nan_counts[nan_counts == len(df_full)].index.tolist()
df_full.drop(columns=all_nan, inplace=True)
nan_counts = df_full.isnull().sum()
high_nan   = nan_counts[nan_counts > thresh].index.tolist()
df_full.drop(columns=high_nan, inplace=True)

before = len(df_full)
df_full.dropna(inplace=True)
print(f"After dropna: {len(df_full):,} rows (dropped {before-len(df_full):,})")
print(f"Feature cols: {df_full.shape[1]}")

# ================================================================
# OVERNIGHT M5 CONTEXT — lean version
# Only the features that ranked highest
# ================================================================
print("\nBuilding overnight M5 context...")
m5_clean = df_m5.dropna(axis=1, how='all')

h1_london_idx = df_full[
    (df_full.index.hour.isin([7, 8])) &
    (df_full.index.dayofweek != 4)
].index

overnight_records = []

for ts in h1_london_idx:
    day_start = ts.replace(hour=0, minute=0, second=0)
    overnight = m5_clean[
        (m5_clean.index >= day_start) &
        (m5_clean.index < ts)
    ]
    if len(overnight) < 5 or 'close' not in overnight.columns:
        continue

    closes = overnight['close'].values
    highs  = overnight['high'].values
    lows   = overnight['low'].values

    rec = {'datetime': ts}

    # Core overnight features (top importance)
    rec['overnight_ret']   = float((closes[-1] - closes[0]) / closes[0])
    rec['overnight_range'] = float(highs.max() - lows.min())
    rec['last_hour_ret']   = 0.0
    rec['last_hour_range'] = 0.0

    last_hour = overnight[overnight.index.hour == 6]
    if len(last_hour) > 0:
        lh = last_hour['close'].values
        rec['last_hour_ret']   = float((lh[-1] - lh[0]) / lh[0])
        rec['last_hour_range'] = float(
            last_hour['high'].max() - last_hour['low'].min()
        )

    if 'M5_macd_diff' in overnight.columns:
        rec['overnight_macd']   = float(overnight['M5_macd_diff'].iloc[-1])
    if 'M5_bb_position' in overnight.columns:
        rec['overnight_bb_pos'] = float(overnight['M5_bb_position'].iloc[-1])
    if 'M5_rsi_14' in overnight.columns:
        rec['overnight_rsi']    = float(overnight['M5_rsi_14'].iloc[-1])
    if 'M5_atr_14' in overnight.columns:
        rec['overnight_atr']    = float(overnight['M5_atr_14'].iloc[-1])

    overnight_records.append(rec)

df_ovn = pd.DataFrame(overnight_records).set_index('datetime')
df_ovn.index = pd.DatetimeIndex(df_ovn.index)
if df_ovn.index.tz is None and df_full.index.tz is not None:
    df_ovn.index = df_ovn.index.tz_localize('UTC')

print(f"Overnight context: {len(df_ovn):,} bars, {df_ovn.shape[1]} features")

# ================================================================
# LABELING
# ================================================================
print("\nLabeling session trades...")

atr_col = next((c for c in df_full.columns if 'atr_14' in c.lower()), None)
print(f"Using ATR: {atr_col}")

closes = df_full['close'].values
highs  = df_full['high'].values
lows   = df_full['low'].values
atrs   = df_full[atr_col].values
hours  = df_full.index.hour
days   = df_full.index.dayofweek

labels       = np.full(len(df_full), -1, dtype=int)
directions   = np.full(len(df_full),  0, dtype=int)
long_results = np.full(len(df_full), -1, dtype=int)
shrt_results = np.full(len(df_full), -1, dtype=int)

MAX_HOLD  = 13
TP_FACTOR = 1.5
SL_FACTOR = 1.0

for i in range(len(df_full) - MAX_HOLD):
    if hours[i] not in [7, 8]:
        continue
    if days[i] == 4:
        continue

    atr = atrs[i]
    if np.isnan(atr) or atr == 0:
        continue

    entry   = closes[i]
    long_tp = entry + atr * TP_FACTOR
    long_sl = entry - atr * SL_FACTOR
    shrt_tp = entry - atr * TP_FACTOR
    shrt_sl = entry + atr * SL_FACTOR

    long_won = long_done = False
    shrt_won = shrt_done = False

    for j in range(1, MAX_HOLD + 1):
        if i + j >= len(df_full):
            break
        if df_full.index[i+j].hour >= 21:
            break
        if df_full.index[i+j].dayofweek != days[i]:
            break

        h = highs[i+j]
        l = lows[i+j]

        if not long_done:
            if l <= long_sl:   long_done = True; long_won = False
            elif h >= long_tp: long_done = True; long_won = True

        if not shrt_done:
            if h >= shrt_sl:   shrt_done = True; shrt_won = False
            elif l <= shrt_tp: shrt_done = True; shrt_won = True

        if long_done and shrt_done:
            break

    if long_done or shrt_done:
        long_results[i] = 1 if long_won else 0
        shrt_results[i] = 1 if shrt_won else 0

        if long_won and not shrt_won:
            labels[i] = 1;  directions[i] =  1
        elif shrt_won and not long_won:
            labels[i] = 0;  directions[i] = -1
        elif long_won and shrt_won:
            labels[i] = 1;  directions[i] =  1
        else:
            labels[i] = -1; directions[i] =  0

df_full['label']       = labels
df_full['direction']   = directions
df_full['long_result'] = long_results
df_full['shrt_result'] = shrt_results

# ================================================================
# MERGE OVERNIGHT + SAVE
# ================================================================
df_full = df_full.join(df_ovn, how='left', rsuffix='_ovn')
df_full = df_full.loc[:, ~df_full.columns.duplicated(keep='first')]

df_setups = df_full[df_full['long_result'] >= 0].copy()
df_setups.dropna(subset=['overnight_range'], inplace=True)

total  = len(df_setups)
long_w = int(df_setups['long_result'].sum())
shrt_w = int(df_setups['shrt_result'].sum())
lbl_1  = int((df_setups['label'] ==  1).sum())
lbl_0  = int((df_setups['label'] ==  0).sum())
lbl_n  = int((df_setups['label'] == -1).sum())

print(f"\n--- Lean Session Summary ---")
print(f"Total resolved:   {total:,}")
print(f"Long win rate:    {long_w/total*100:.1f}%")
print(f"Short win rate:   {shrt_w/total*100:.1f}%")
print(f"Label=1 (long):   {lbl_1:,}  ({lbl_1/total*100:.1f}%)")
print(f"Label=0 (short):  {lbl_0:,}  ({lbl_0/total*100:.1f}%)")
print(f"Label=-1 (skip):  {lbl_n:,}  ({lbl_n/total*100:.1f}%)")
print(f"Features:         {df_setups.shape[1]} columns")

df_trade = df_setups[df_setups['label'] >= 0].copy()
print(f"Tradeable:        {len(df_trade):,}")

df_trade.to_csv(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")