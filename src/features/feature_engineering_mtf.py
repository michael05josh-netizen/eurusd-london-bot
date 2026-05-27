import pandas as pd
import numpy as np
import os

M5_PATH  = "data/raw/EURUSD_M5_clean.csv"
H1_PATH  = "data/raw/EURUSD_H1.csv"
H4_PATH  = "data/raw/EURUSD_H4.csv"
OUT_PATH = "data/featured/EURUSD_MTF_features.csv"

os.makedirs("data/featured", exist_ok=True)

# --- Load ---
print("Loading timeframes...")
df_m5 = pd.read_csv(M5_PATH, index_col=0, parse_dates=True)
df_h1 = pd.read_csv(H1_PATH, index_col=0, parse_dates=True)
df_h4 = pd.read_csv(H4_PATH, index_col=0, parse_dates=True)

print(f"M5: {len(df_m5):,}  H1: {len(df_h1):,}  H4: {len(df_h4):,}")

# --- Drop raw OHLCV from H1/H4 ---
drop_cols  = ['open', 'high', 'low', 'close', 'volume']
df_h1_feat = df_h1.drop(columns=[c for c in drop_cols if c in df_h1.columns])
df_h4_feat = df_h4.drop(columns=[c for c in drop_cols if c in df_h4.columns])

# --- Drop all-NaN cols before alignment ---
df_h1_feat = df_h1_feat.dropna(axis=1, how='all')
df_h4_feat = df_h4_feat.dropna(axis=1, how='all')
print(f"H1 feature cols: {df_h1_feat.shape[1]}  H4 feature cols: {df_h4_feat.shape[1]}")

# --- Shift by 1 to prevent lookahead bias ---
df_h1_shifted = df_h1_feat.shift(1)
df_h4_shifted = df_h4_feat.shift(1)

# --- Align to M5 index via forward fill ---
print("Aligning to M5 index...")
df_h1_aligned = df_h1_shifted.reindex(df_m5.index, method='ffill')
df_h4_aligned = df_h4_shifted.reindex(df_m5.index, method='ffill')

print(f"H1 fully clean rows: {df_h1_aligned.notna().all(axis=1).sum():,} / {len(df_h1_aligned):,}")
print(f"H4 fully clean rows: {df_h4_aligned.notna().all(axis=1).sum():,} / {len(df_h4_aligned):,}")

# --- Time features ---
time_df = pd.DataFrame(index=df_m5.index)
time_df['hour']        = df_m5.index.hour
time_df['day_of_week'] = df_m5.index.dayofweek
time_df['is_london']   = ((time_df['hour'] >= 7)  & (time_df['hour'] < 16)).astype(int)
time_df['is_ny']       = ((time_df['hour'] >= 13) & (time_df['hour'] < 21)).astype(int)
time_df['is_overlap']  = ((time_df['hour'] >= 13) & (time_df['hour'] < 16)).astype(int)
time_df['is_asian']    = ((time_df['hour'] >= 0)  & (time_df['hour'] < 7)).astype(int)
time_df['is_active']   = ((time_df['hour'] >= 7)  & (time_df['hour'] < 21)).astype(int)

# --- Concat all at once ---
print("Concatenating all features...")
df_full = pd.concat([df_m5, df_h1_aligned, df_h4_aligned, time_df], axis=1)
df_full = df_full.copy()
print(f"Combined shape: {df_full.shape}")

# --- Drop all-NaN cols (volume-derived, etc) ---
nan_counts   = df_full.isnull().sum()
all_nan_cols = nan_counts[nan_counts == len(df_full)].index.tolist()
print(f"Dropping {len(all_nan_cols)} all-NaN cols: {all_nan_cols}")
df_full.drop(columns=all_nan_cols, inplace=True)

# --- Drop high-NaN cols (>5% NaN = deep warmup artifacts) ---
nan_counts = df_full.isnull().sum()
thresh     = int(len(df_full) * 0.05)
drop_high  = nan_counts[nan_counts > thresh].index.tolist()
print(f"Dropping {len(drop_high)} high-NaN cols: {drop_high}")
df_full.drop(columns=drop_high, inplace=True)

# --- Cross-TF features (only using columns confirmed to exist) ---
print("Building cross-TF features...")
cross = pd.DataFrame(index=df_full.index)

def safe_col(df, col):
    """Return column if exists, else zeros series."""
    if col in df.columns:
        return df[col]
    print(f"  [WARN] Column not found: {col}")
    return pd.Series(0, index=df.index)

# MTF EMA alignment
cross['mtf_bull'] = (
    (safe_col(df_full, 'M5_ema_cross_8_21') == 1) &
    (safe_col(df_full, 'H1_ema_cross_8_21') == 1) &
    (safe_col(df_full, 'H4_ema_cross_8_21') == 1)
).astype(int)

cross['mtf_bear'] = (
    (safe_col(df_full, 'M5_ema_cross_8_21') == 0) &
    (safe_col(df_full, 'H1_ema_cross_8_21') == 0) &
    (safe_col(df_full, 'H4_ema_cross_8_21') == 0)
).astype(int)

cross['mtf_trend_bull'] = (
    (safe_col(df_full, 'M5_trend_aligned') == 1) &
    (safe_col(df_full, 'H1_trend_aligned') == 1)
).astype(int)

cross['mtf_trend_bear'] = (
    (safe_col(df_full, 'M5_trend_aligned_bear') == 1) &
    (safe_col(df_full, 'H1_trend_aligned_bear') == 1)
).astype(int)

# RSI divergence across TFs
cross['rsi_mtf_diff_h1'] = safe_col(df_full, 'M5_rsi_14') - safe_col(df_full, 'H1_rsi_14')
cross['rsi_mtf_diff_h4'] = safe_col(df_full, 'M5_rsi_14') - safe_col(df_full, 'H4_rsi_14')

# Range activity context from H1/H4
cross['h1_range_ratio']  = safe_col(df_full, 'H1_range_ratio')
cross['h4_range_ratio']  = safe_col(df_full, 'H4_range_ratio')
cross['h1_range_act']    = safe_col(df_full, 'H1_range_activity')
cross['h4_range_act']    = safe_col(df_full, 'H4_range_activity')

# Volatility regime
cross['h1_atr_regime']   = safe_col(df_full, 'H1_atr_vs_mean')
cross['h4_atr_regime']   = safe_col(df_full, 'H4_atr_vs_mean')
cross['h1_atr_ratio']    = safe_col(df_full, 'H1_atr_ratio')
cross['h4_atr_ratio']    = safe_col(df_full, 'H4_atr_ratio')

# BB position spread across TFs
cross['bb_mtf_diff_h1']  = (
    safe_col(df_full, 'M5_bb_position') - safe_col(df_full, 'H1_bb_position')
)
cross['bb_mtf_diff_h4']  = (
    safe_col(df_full, 'M5_bb_position') - safe_col(df_full, 'H4_bb_position')
)

# MACD alignment
cross['macd_mtf_bull'] = (
    (safe_col(df_full, 'M5_macd_cross') == 1) &
    (safe_col(df_full, 'H1_macd_cross') == 1)
).astype(int)

cross['macd_mtf_bear'] = (
    (safe_col(df_full, 'M5_macd_cross') == 0) &
    (safe_col(df_full, 'H1_macd_cross') == 0)
).astype(int)

# Price position vs higher TF EMAs
cross['m5_vs_h1_ema50']  = (df_full['close'] > safe_col(df_full, 'H1_ema_50')).astype(int)
cross['m5_vs_h4_ema50']  = (df_full['close'] > safe_col(df_full, 'H4_ema_50')).astype(int)
cross['m5_vs_h1_ema21']  = (df_full['close'] > safe_col(df_full, 'H1_ema_21')).astype(int)
cross['m5_vs_h4_ema21']  = (df_full['close'] > safe_col(df_full, 'H4_ema_21')).astype(int)

# Trend strength: distance from H1/H4 EMA50 normalized by ATR
cross['h1_trend_strength'] = (
    (df_full['close'] - safe_col(df_full, 'H1_ema_50')) /
    safe_col(df_full, 'H1_atr_14').replace(0, np.nan)
)
cross['h4_trend_strength'] = (
    (df_full['close'] - safe_col(df_full, 'H4_ema_50')) /
    safe_col(df_full, 'H4_atr_14').replace(0, np.nan)
)

# RSI regime on H1/H4
cross['h1_rsi_bull'] = (safe_col(df_full, 'H1_rsi_14') > 50).astype(int)
cross['h4_rsi_bull'] = (safe_col(df_full, 'H4_rsi_14') > 50).astype(int)
cross['h1_rsi_14']   = safe_col(df_full, 'H1_rsi_14')
cross['h4_rsi_14']   = safe_col(df_full, 'H4_rsi_14')

# H1 MACD diff value
cross['h1_macd_diff'] = safe_col(df_full, 'H1_macd_diff')
cross['h4_macd_diff'] = safe_col(df_full, 'H4_macd_diff')

# H1 BB squeeze
cross['h1_bb_squeeze'] = safe_col(df_full, 'H1_bb_squeeze')
cross['h4_bb_squeeze'] = safe_col(df_full, 'H4_bb_squeeze')

# Concat cross features
df_full = pd.concat([df_full, cross], axis=1)
df_full = df_full.copy()
print(f"Shape after cross-TF features: {df_full.shape}")

# --- Final NaN cleanup ---
nan_counts = df_full.isnull().sum()
remaining_bad = nan_counts[nan_counts > thresh].index.tolist()
if remaining_bad:
    print(f"Dropping {len(remaining_bad)} remaining high-NaN cols: {remaining_bad}")
    df_full.drop(columns=remaining_bad, inplace=True)

print(f"Rows before dropna: {len(df_full):,}")
df_full.dropna(inplace=True)
print(f"Rows after  dropna: {len(df_full):,}")

if len(df_full) == 0:
    print("\n[ERROR] Still zero rows.")
    print(df_full.isnull().sum().sort_values(ascending=False).head(10))
    exit()

# ================================================================
# SETUP FILTER
# ================================================================
long_mask = (
    (df_full['is_active']       == 1) &
    (df_full['mtf_bull']        == 1) &
    (df_full['M5_rsi_14']       >= 40) &
    (df_full['M5_rsi_14']       <= 65) &
    (df_full['h1_rsi_14']       >= 45) &
    (df_full['m5_vs_h1_ema50']  == 1) &
    (df_full['m5_vs_h4_ema50']  == 1) &
    (df_full['M5_atr_ratio']    >  0.9) &
    (df_full['M5_near_high_20'] == 0) &
    (df_full['day_of_week']     != 4)
)

short_mask = (
    (df_full['is_active']       == 1) &
    (df_full['mtf_bear']        == 1) &
    (df_full['M5_rsi_14']       >= 35) &
    (df_full['M5_rsi_14']       <= 60) &
    (df_full['h1_rsi_14']       <= 55) &
    (df_full['m5_vs_h1_ema50']  == 0) &
    (df_full['m5_vs_h4_ema50']  == 0) &
    (df_full['M5_atr_ratio']    >  0.9) &
    (df_full['M5_near_low_20']  == 0) &
    (df_full['day_of_week']     != 4)
)

print(f"\nLong  candidates: {long_mask.sum():,} ({long_mask.sum()/len(df_full)*100:.1f}%)")
print(f"Short candidates: {short_mask.sum():,} ({short_mask.sum()/len(df_full)*100:.1f}%)")

# ================================================================
# LABELING — 1:2 R:R, max 12 candles
# ================================================================
WINDOW    = 12
SL_FACTOR = 0.5
TP_FACTOR = 1.0

print("Labeling setups...")

closes    = df_full['close'].values
highs     = df_full['high'].values
lows      = df_full['low'].values
atrs      = df_full['M5_atr_14'].values
l_mask    = long_mask.values
s_mask    = short_mask.values
labels    = np.full(len(df_full), -1, dtype=int)
direction = np.full(len(df_full),  0, dtype=int)

for i in range(len(df_full) - WINDOW):
    atr = atrs[i]
    if np.isnan(atr) or atr == 0:
        continue
    if l_mask[i]:
        entry = closes[i]
        tp    = entry + atr * TP_FACTOR
        sl    = entry - atr * SL_FACTOR
        won   = False
        for j in range(1, WINDOW + 1):
            if lows[i+j]  <= sl: break
            if highs[i+j] >= tp: won = True; break
        labels[i]    = 1 if won else 0
        direction[i] = 1
    elif s_mask[i]:
        entry = closes[i]
        tp    = entry - atr * TP_FACTOR
        sl    = entry + atr * SL_FACTOR
        won   = False
        for j in range(1, WINDOW + 1):
            if highs[i+j] >= sl: break
            if lows[i+j]  <= tp: won = True; break
        labels[i]    = 1 if won else 0
        direction[i] = -1

df_full['label']     = labels
df_full['direction'] = direction

df_setups  = df_full[df_full['label'] >= 0].copy()
total      = len(df_setups)
long_total = (df_setups['direction'] ==  1).sum()
shrt_total = (df_setups['direction'] == -1).sum()
long_wins  = df_setups[df_setups['direction'] ==  1]['label'].sum()
shrt_wins  = df_setups[df_setups['direction'] == -1]['label'].sum()

print(f"\n--- MTF Label Summary ---")
print(f"Total setups:   {total:,}")
print(f"Long  setups:   {long_total:,}  |  wins: {long_wins:,}  ({long_wins/max(long_total,1)*100:.1f}%)")
print(f"Short setups:   {shrt_total:,}  |  wins: {shrt_wins:,}  ({shrt_wins/max(shrt_total,1)*100:.1f}%)")
print(f"Overall win:    {df_setups['label'].mean()*100:.1f}%")
print(f"Total features: {df_setups.shape[1]} columns")

df_setups.to_csv(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")