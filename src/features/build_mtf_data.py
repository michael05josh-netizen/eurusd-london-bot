import pandas as pd
import numpy as np
import ta
import os

IN_PATH  = "data/raw/EURUSD_M5.csv"
OUT_M5   = "data/raw/EURUSD_M5_clean.csv"
OUT_H1   = "data/raw/EURUSD_H1.csv"
OUT_H4   = "data/raw/EURUSD_H4.csv"

os.makedirs("data/raw", exist_ok=True)

# --- Load M5 base ---
print("Loading M5 data...")
df_m5 = pd.read_csv(IN_PATH, index_col=0, parse_dates=True)
df_m5 = df_m5.sort_index()
print(f"M5: {len(df_m5):,} candles  |  {df_m5.index[0]} → {df_m5.index[-1]}")

# --- Resample to H1 ---
print("Resampling to H1...")
df_h1 = df_m5.resample('1h').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum'   # sum of tick counts = H1 volume proxy
}).dropna()
print(f"H1: {len(df_h1):,} candles")

# --- Resample to H4 ---
print("Resampling to H4...")
df_h4 = df_m5.resample('4h').agg({
    'open':   'first',
    'high':   'max',
    'low':    'min',
    'close':  'last',
    'volume': 'sum'
}).dropna()
print(f"H4: {len(df_h4):,} candles")

# ================================================================
# FEATURE BUILDER — runs on any timeframe dataframe
# ================================================================
def build_features(df, tf_label):
    d = df.copy()
    print(f"\nBuilding {tf_label} features...")

    # --- Candle structure ---
    d['body']         = d['close'] - d['open']
    d['body_size']    = d['body'].abs()
    d['upper_wick']   = d['high'] - d[['open','close']].max(axis=1)
    d['lower_wick']   = d[['open','close']].min(axis=1) - d['low']
    d['candle_range'] = d['high'] - d['low']
    d['is_bullish']   = (d['close'] > d['open']).astype(int)

    # --- Trend ---
    d['ema_8']          = ta.trend.ema_indicator(d['close'], window=8)
    d['ema_21']         = ta.trend.ema_indicator(d['close'], window=21)
    d['ema_50']         = ta.trend.ema_indicator(d['close'], window=50)
    d['ema_200']        = ta.trend.ema_indicator(d['close'], window=200)
    d['ema_cross_8_21'] = (d['ema_8']  > d['ema_21']).astype(int)
    d['ema_cross_21_50']= (d['ema_21'] > d['ema_50']).astype(int)
    d['trend_aligned']  = (
        (d['ema_8'] > d['ema_21']) &
        (d['ema_21'] > d['ema_50']) &
        (d['ema_50'] > d['ema_200'])
    ).astype(int)
    d['trend_aligned_bear'] = (
        (d['ema_8'] < d['ema_21']) &
        (d['ema_21'] < d['ema_50']) &
        (d['ema_50'] < d['ema_200'])
    ).astype(int)
    d['price_vs_ema50']  = (d['close'] - d['ema_50'])  / d['ema_50']
    d['price_vs_ema200'] = (d['close'] - d['ema_200']) / d['ema_200']
    d['ema8_slope']      = d['ema_8'].diff(3)  / d['ema_8'].shift(3)
    d['ema21_slope']     = d['ema_21'].diff(3) / d['ema_21'].shift(3)

    # --- Momentum ---
    d['rsi_14']    = ta.momentum.rsi(d['close'], window=14)
    d['rsi_7']     = ta.momentum.rsi(d['close'], window=7)
    d['rsi_slope'] = d['rsi_14'].diff(3)
    d['roc_5']     = ta.momentum.roc(d['close'], window=5)
    d['roc_10']    = ta.momentum.roc(d['close'], window=10)

    macd = ta.trend.MACD(d['close'], window_slow=26, window_fast=12, window_sign=9)
    d['macd']        = macd.macd()
    d['macd_signal'] = macd.macd_signal()
    d['macd_diff']   = macd.macd_diff()
    d['macd_cross']  = (d['macd'] > d['macd_signal']).astype(int)

    # --- Volatility ---
    d['atr_14'] = ta.volatility.average_true_range(
        d['high'], d['low'], d['close'], window=14
    ).replace(0, np.nan)
    d['atr_7']  = ta.volatility.average_true_range(
        d['high'], d['low'], d['close'], window=7
    ).replace(0, np.nan)
    d['atr_ratio']   = d['atr_7'] / d['atr_14']
    d['atr_vs_mean'] = d['atr_14'] / d['atr_14'].rolling(50).mean()

    bb = ta.volatility.BollingerBands(d['close'], window=20, window_dev=2)
    d['bb_width']    = (bb.bollinger_hband() - bb.bollinger_lband()) / d['close']
    d['bb_position'] = (d['close'] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )
    d['bb_squeeze']  = (d['bb_width'] < d['bb_width'].rolling(50).mean()).astype(int)

    # --- Volume proxy (real volume is zero in HistData, use range expansion) ---
    d['vol_sma_20']   = d['volume'].rolling(20).mean().replace(0, np.nan)
    d['vol_ratio']    = d['volume'] / d['vol_sma_20']  # will be NaN for HistData
    # Range-based activity proxy (always valid)
    d['range_activity'] = d['candle_range'] / d['candle_range'].rolling(50).mean()

    # Range activity (M5 supplement)
    d['range_sma_20'] = d['candle_range'].rolling(20).mean()
    d['range_ratio']  = d['candle_range'] / d['range_sma_20'].replace(0, np.nan)

    # --- Market structure ---
    d['high_10']  = d['high'].rolling(10).max()
    d['low_10']   = d['low'].rolling(10).min()
    d['high_20']  = d['high'].rolling(20).max()
    d['low_20']   = d['low'].rolling(20).min()
    d['high_50']  = d['high'].rolling(50).max()
    d['low_50']   = d['low'].rolling(50).min()

    d['dist_high_20'] = (d['high_20'] - d['close']) / d['atr_14']
    d['dist_low_20']  = (d['close']  - d['low_20'])  / d['atr_14']
    d['near_high_20'] = (d['dist_high_20'] < 1.0).astype(int)
    d['near_low_20']  = (d['dist_low_20']  < 1.0).astype(int)

    # --- Lag features ---
    for lag in [1, 2, 3]:
        d[f'close_lag_{lag}']  = d['close'].shift(lag)
        d[f'rsi_lag_{lag}']    = d['rsi_14'].shift(lag)
        d[f'body_lag_{lag}']   = d['body'].shift(lag)
        d[f'macd_lag_{lag}']   = d['macd_diff'].shift(lag)
        d[f'vol_lag_{lag}']    = d['vol_ratio'].shift(lag)

    # Rename all feature cols with timeframe prefix
    feature_cols = [c for c in d.columns
                    if c not in ['open','high','low','close','volume']]
    rename_map   = {c: f"{tf_label}_{c}" for c in feature_cols}
    d.rename(columns=rename_map, inplace=True)

    print(f"  {tf_label} features: {len(feature_cols)}")
    return d

# --- Build features for each TF ---
df_m5_feat = build_features(df_m5, 'M5')
df_h1_feat = build_features(df_h1, 'H1')
df_h4_feat = build_features(df_h4, 'H4')

# --- Save ---
df_m5_feat.to_csv(OUT_M5)
df_h1_feat.to_csv(OUT_H1)
df_h4_feat.to_csv(OUT_H4)

print(f"\nSaved:")
print(f"  M5 → {OUT_M5}  ({len(df_m5_feat):,} rows, {df_m5_feat.shape[1]} cols)")
print(f"  H1 → {OUT_H1}  ({len(df_h1_feat):,} rows, {df_h1_feat.shape[1]} cols)")
print(f"  H4 → {OUT_H4}  ({len(df_h4_feat):,} rows, {df_h4_feat.shape[1]} cols)")