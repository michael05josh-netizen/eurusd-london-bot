import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
import logging
import os
from datetime import datetime, timezone

# ================================================================
# CONFIG
# ================================================================
SYMBOL          = "EURUSD"
RISK_PER_TRADE  = 0.01      # 1% risk per trade
TP_R            = 1.5
SL_R            = 1.0
ML_THRESHOLD    = 0.70
SPREAD_LIMIT    = 2.0       # max allowed spread in pips
SKIP_MONTHS = []  # No hardcoded month filters
ENTRY_HOURS     = [7, 8]    # London open UTC
MAGIC           = 20250101  # unique ID for bot orders
COMMENT         = "LondonSessionBot"

# Paths
MODEL_PATH  = "models/lgbm_session.pkl"
SCALER_PATH = "models/lgbm_session_scaler.pkl"
FEAT_PATH   = "models/lgbm_session_features.pkl"
M5_PATH     = "data/raw/EURUSD_M5_clean.csv"
H4_PATH     = "data/raw/EURUSD_H4.csv"
TICK_PATH   = "data/raw/EURUSD_tick_features_2023_2025.csv"
LOG_DIR     = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
# LOGGING
# ================================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(message)s',
    handlers= [
        logging.FileHandler(f"{LOG_DIR}/bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ================================================================
# LOAD MODEL
# ================================================================
log.info("Loading model...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feats  = joblib.load(FEAT_PATH)
log.info(f"Model loaded. Features: {len(feats)}")

# ================================================================
# MT5 CONNECTION
# ================================================================
def connect_mt5():
    if not mt5.initialize():
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    log.info(f"Connected: {info.company} | Account: {info.login} | "
             f"Balance: ${info.balance:,.2f}")
    return True

def disconnect_mt5():
    mt5.shutdown()
    log.info("MT5 disconnected")

# ================================================================
# DATA FETCHERS
# ================================================================
def get_h1_bars(symbol, n=300):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n)
    if rates is None:
        log.error(f"Failed to get H1 bars: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def get_h4_bars(symbol, n=200):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def get_m5_bars(symbol, n=500):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    return df

def get_current_spread(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 999
    spread = (tick.ask - tick.bid) * 10000
    return spread

# ================================================================
# INDICATOR COMPUTATION
# ================================================================
def compute_indicators(df, prefix=''):
    import ta
    d = df.copy()

    # Trend
    d[f'{prefix}ema_8']           = ta.trend.ema_indicator(d['close'], 8)
    d[f'{prefix}ema_21']          = ta.trend.ema_indicator(d['close'], 21)
    d[f'{prefix}ema_50']          = ta.trend.ema_indicator(d['close'], 50)
    d[f'{prefix}ema_200']         = ta.trend.ema_indicator(d['close'], 200)
    d[f'{prefix}ema_cross_8_21']  = (d[f'{prefix}ema_8'] > d[f'{prefix}ema_21']).astype(int)
    d[f'{prefix}trend_aligned']   = (
        (d[f'{prefix}ema_8']  > d[f'{prefix}ema_21']) &
        (d[f'{prefix}ema_21'] > d[f'{prefix}ema_50']) &
        (d[f'{prefix}ema_50'] > d[f'{prefix}ema_200'])
    ).astype(int)

    # Momentum
    d[f'{prefix}rsi_14']    = ta.momentum.rsi(d['close'], 14)
    d[f'{prefix}rsi_slope'] = d[f'{prefix}rsi_14'].diff(3)

    macd = ta.trend.MACD(d['close'])
    d[f'{prefix}macd']        = macd.macd()
    d[f'{prefix}macd_signal'] = macd.macd_signal()
    d[f'{prefix}macd_diff']   = macd.macd_diff()

    # Volatility
    d[f'{prefix}atr_14'] = ta.volatility.average_true_range(
        d['high'], d['low'], d['close'], 14
    ).replace(0, np.nan)
    d[f'{prefix}atr_7']  = ta.volatility.average_true_range(
        d['high'], d['low'], d['close'], 7
    ).replace(0, np.nan)
    d[f'{prefix}atr_ratio']   = d[f'{prefix}atr_7'] / d[f'{prefix}atr_14']
    d[f'{prefix}atr_vs_mean'] = d[f'{prefix}atr_14'] / d[f'{prefix}atr_14'].rolling(50).mean()

    bb = ta.volatility.BollingerBands(d['close'], 20)
    d[f'{prefix}bb_position'] = (
        (d['close'] - bb.bollinger_lband()) /
        (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
    )
    d[f'{prefix}bb_width'] = (
        (bb.bollinger_hband() - bb.bollinger_lband()) / d['close']
    )

    # Price action
    d[f'{prefix}body']      = d['close'] - d['open']
    d[f'{prefix}body_lag_1']= d[f'{prefix}body'].shift(1)
    d[f'{prefix}body_lag_2']= d[f'{prefix}body'].shift(2)
    d[f'{prefix}body_lag_3']= d[f'{prefix}body'].shift(3)
    d[f'{prefix}lower_wick']= d[['open','close']].min(axis=1) - d['low']

    # Range
    d[f'{prefix}range']         = d['high'] - d['low']
    d[f'{prefix}range_sma_20']  = d[f'{prefix}range'].rolling(20).mean()
    d[f'{prefix}range_ratio']   = d[f'{prefix}range'] / d[f'{prefix}range_sma_20'].replace(0, np.nan)

    # ROC
    d[f'{prefix}roc_10'] = ta.momentum.roc(d['close'], 10)

    # Distance from structure
    d[f'{prefix}high_20']      = d['high'].rolling(20).max()
    d[f'{prefix}low_20']       = d['low'].rolling(20).min()
    d[f'{prefix}dist_high_20'] = (d[f'{prefix}high_20'] - d['close']) / d[f'{prefix}atr_14']
    d[f'{prefix}dist_low_20']  = (d['close'] - d[f'{prefix}low_20'])  / d[f'{prefix}atr_14']

    return d

# ================================================================
# OVERNIGHT CONTEXT
# ================================================================
def get_overnight_context(df_m5, london_ts):
    day_start = london_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Handle timezone
    if hasattr(london_ts, 'tzinfo') and london_ts.tzinfo is not None:
        if df_m5.index.tz is None:
            df_m5 = df_m5.copy()
            df_m5.index = df_m5.index.tz_localize('UTC')
    
    overnight = df_m5[
        (df_m5.index >= day_start) & 
        (df_m5.index < london_ts)
    ]

    if len(overnight) < 5 or 'close' not in overnight.columns:
        return {
            'overnight_ret':    0.0,
            'overnight_range':  0.0,
            'overnight_rsi':    50.0,
            'overnight_macd':   0.0,
            'overnight_bb_pos': 0.5,
            'overnight_atr':    0.0,
            'last_hour_ret':    0.0,
            'last_hour_range':  0.0,
        }

    closes = overnight['close'].values
    highs  = overnight['high'].values
    lows   = overnight['low'].values

    def safe_float(series, default=0.0):
        try:
            val = float(series.iloc[-1])
            return val if not np.isnan(val) else default
        except:
            return default

    ctx = {
        'overnight_ret':   float((closes[-1] - closes[0]) / closes[0]),
        'overnight_range': float(highs.max() - lows.min()),
        'overnight_rsi':   safe_float(overnight['M5_rsi_14'],    50.0)
                           if 'M5_rsi_14'    in overnight.columns else 50.0,
        'overnight_macd':  safe_float(overnight['M5_macd_diff'],  0.0)
                           if 'M5_macd_diff' in overnight.columns else 0.0,
        'overnight_bb_pos':safe_float(overnight['M5_bb_position'],0.5)
                           if 'M5_bb_position' in overnight.columns else 0.5,
        'overnight_atr':   safe_float(overnight['M5_atr_14'],     0.0)
                           if 'M5_atr_14'    in overnight.columns else 0.0,
    }

    last_hour = overnight[overnight.index.hour == 6]
    if len(last_hour) > 0:
        lh = last_hour['close'].values
        ctx['last_hour_ret']   = float((lh[-1] - lh[0]) / lh[0])
        ctx['last_hour_range'] = float(
            last_hour['high'].max() - last_hour['low'].min()
        )
    else:
        ctx['last_hour_ret']   = 0.0
        ctx['last_hour_range'] = 0.0

    return ctx

# ================================================================
# TICK CONTEXT (from historical tick features)
# ================================================================
def get_tick_context(tick_df, ts):
    # Get last 3 hours of tick data before ts
    window = tick_df[tick_df.index <= ts].tail(3)
    if len(window) == 0:
        return {}
    return {
        'tick_imbal_3h': float(window['tick_imbalance'].mean()),
        'spread_3h':     float(window['spread_mean'].mean()),
        'tick_vol_3h':   float(window['tick_count'].mean()),
        'buy_pres_3h':   float(window['buy_pressure'].mean()),
        'tick_velocity': float(window['tick_velocity'].mean()),
        'buy_pressure':  float(window['buy_pressure'].iloc[-1]),
    }

# ================================================================
# BUILD FEATURE VECTOR FOR PREDICTION
# ================================================================
def build_feature_vector(h1_bars, h4_bars, m5_bars, tick_df, ts):
    # Compute H1 indicators
    h1 = compute_indicators(h1_bars, prefix='H1_')
    h4 = compute_indicators(h4_bars, prefix='H4_')

    # Get latest H1 bar (current bar)
    h1_latest = h1.iloc[-1]
    h4_latest = h4.iloc[-1]

    # Time features
    row = {
        'hour':        ts.hour,
        'day_of_week': ts.weekday(),
        'sin_hour':    np.sin(2 * np.pi * ts.hour / 24),
        'cos_hour':    np.cos(2 * np.pi * ts.hour / 24),
        'sin_dow':     np.sin(2 * np.pi * ts.weekday() / 5),
        'cos_dow':     np.cos(2 * np.pi * ts.weekday() / 5),
        'is_monday':   int(ts.weekday() == 0),
        'is_thursday': int(ts.weekday() == 3),
        'month':       ts.month,
        'is_feb':      int(ts.month == 2),
        'is_jun':      int(ts.month == 6),
        'is_mar':      int(ts.month == 3),
        'is_q1':       int(ts.month in [1,2,3]),
        'is_q2':       int(ts.month in [4,5,6]),
    }

    # H1 features
    for col in h1_latest.index:
        if col.startswith('H1_'):
            row[col] = float(h1_latest[col]) if not pd.isna(h1_latest[col]) else 0.0

    # H4 features
    for col in h4_latest.index:
        if col.startswith('H4_'):
            row[col] = float(h4_latest[col]) if not pd.isna(h4_latest[col]) else 0.0

    # Overnight context
    ovn = get_overnight_context(m5_bars, ts)
    row.update(ovn)

    # Tick context
    tick_ctx = get_tick_context(tick_df, ts)
    row.update(tick_ctx)

    # Interaction features
    row['trend_x_overnight'] = row.get('H4_trend_aligned', 0) * row.get('overnight_ret', 0) * 10000
    row['h4rsi_x_lasthour']  = row.get('H4_rsi_slope', 0) * row.get('last_hour_ret', 0) * 10000
    row['tick_x_trend']      = row.get('tick_imbal_3h', 0) * row.get('H4_trend_aligned', 0)
    row['spread_x_range']    = row.get('spread_3h', 0) * row.get('overnight_range', 0) * 10000
    row['atr_x_rsi']         = row.get('H4_atr_ratio', 0) * row.get('H4_rsi_slope', 0)
    row['month_x_trend']     = row.get('month', 0) * row.get('H4_trend_aligned', 0)
    row['ovn_rsi_x_macd']    = (row.get('overnight_rsi', 50) - 50) * np.sign(row.get('overnight_macd', 0))
    row['dist_x_atr']        = row.get('H1_dist_high_20', 0) * row.get('H1_atr_ratio', 0)

    # Build feature vector in correct order
    x = np.array([row.get(f, 0.0) for f in feats], dtype=np.float32)
    return x, row

# ================================================================
# REGIME DETECTION
# ================================================================
def get_regime(row_dict):
    h4_bull = row_dict.get('H4_trend_aligned', 0)
    h4_ema  = row_dict.get('H4_ema_cross_8_21', 0)
    h4_rsi  = row_dict.get('H4_rsi_slope', 0)
    if h4_bull == 1 and h4_ema == 1:
        return 'bull'
    elif h4_ema == 0 and h4_rsi < -0.3:
        return 'bear'
    return 'ranging'

# ================================================================
# POSITION MANAGEMENT
# ================================================================
def has_open_position(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == magic for p in positions)

def place_order(symbol, direction, lot_size, sl_price, tp_price, comment):
    tick    = mt5.symbol_info_tick(symbol)
    info    = mt5.symbol_info(symbol)

    if direction == 'long':
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid

    # Normalize lot size to broker's step
    lot_step = info.volume_step
    lot_size = round(round(lot_size / lot_step) * lot_step, 2)
    lot_size = max(info.volume_min, min(lot_size, info.volume_max))

    # Normalize SL/TP to broker's digits
    digits   = info.digits
    sl_price = round(sl_price, digits)
    tp_price = round(tp_price, digits)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed: {result.retcode} — {result.comment}")
        return None

    log.info(f"Order placed: {direction.upper()} {lot_size} lots @ {price:.5f} | "
             f"SL: {sl_price:.5f} | TP: {tp_price:.5f}")
    return result

# ================================================================
# MAIN BOT LOOP
# ================================================================
def run_bot():
    log.info("="*55)
    log.info("  EURUSD London Session Bot — Starting")
    log.info("="*55)

    if not connect_mt5():
        return

    # Load historical tick features
    log.info("Loading tick context data...")
    try:
        tick_df = pd.read_csv(TICK_PATH, index_col=0, parse_dates=True)
        log.info(f"Tick data loaded: {len(tick_df):,} bars")
    except Exception as e:
        log.warning(f"Tick data not available: {e}. Using zeros.")
        tick_df = pd.DataFrame()

    # Load M5 historical (for overnight context indicators)
    log.info("Loading M5 historical data...")
    try:
        m5_hist = pd.read_csv(M5_PATH, index_col=0, parse_dates=True)
        log.info(f"M5 history loaded: {len(m5_hist):,} bars")
    except Exception as e:
        log.warning(f"M5 history not available: {e}")
        m5_hist = pd.DataFrame()

    traded_today = False
    last_date    = None

    log.info("Bot running. Waiting for London open...")
    log.info(f"Entry hours: {ENTRY_HOURS} UTC | Threshold: {ML_THRESHOLD}")

    # ================================================================
    # TEST MODE — runs one full signal cycle immediately
    # Set TEST_MODE = False for live trading
    # ================================================================
    TEST_MODE = True

    if TEST_MODE:
        log.info("="*55)
        log.info("  TEST MODE — Running single signal cycle")
        log.info("="*55)

        now = datetime.now(timezone.utc)
        log.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M')} UTC")

        # Check spread
        spread = get_current_spread(SYMBOL)
        log.info(f"Current spread: {spread:.1f} pips")

        # Fetch live data
        log.info("Fetching live market data...")
        h1_bars = get_h1_bars(SYMBOL, 300)
        h4_bars = get_h4_bars(SYMBOL, 200)
        m5_bars = get_m5_bars(SYMBOL, 500)

        if h1_bars is None or h4_bars is None:
            log.error("Failed to fetch bars")
            disconnect_mt5()
            exit()

        log.info(f"H1 bars: {len(h1_bars)}  H4 bars: {len(h4_bars)}  "
                f"M5 bars: {len(m5_bars)}")

        # Merge M5 with history
        if len(m5_hist) > 0 and m5_bars is not None:
            m5_combined = pd.concat([m5_hist, m5_bars])
            m5_combined = m5_combined[~m5_combined.index.duplicated(keep='last')]
            m5_combined = m5_combined.sort_index()
        else:
            m5_combined = m5_bars

        # Build features
        log.info("Computing features...")
        try:
            x_vec, row_dict = build_feature_vector(
                h1_bars, h4_bars, m5_combined, tick_df, now
            )
            log.info(f"Feature vector built: {len(x_vec)} features")
        except Exception as e:
            log.error(f"Feature computation failed: {e}")
            import traceback
            traceback.print_exc()
            disconnect_mt5()
            exit()

        # Model prediction
        x_scaled = scaler.transform(x_vec.reshape(1, -1))
        prob     = model.predict_proba(x_scaled)[0][1]
        regime   = get_regime(row_dict)

        log.info(f"")
        log.info(f"  ---- SIGNAL REPORT ----")
        log.info(f"  Time:       {now.strftime('%Y-%m-%d %H:%M')} UTC")
        log.info(f"  Regime:     {regime}")
        log.info(f"  ML prob:    {prob:.4f}")

        if regime in ['bull', 'weak_bull']:
            direction  = 'long'
            confidence = prob if regime == 'bull' else prob * 0.9
        elif regime in ['bear', 'weak_bear']:
            direction  = 'short'
            confidence = (1 - prob) if regime == 'bear' else (1 - prob) * 0.9
        else:
            direction  = 'none'
            confidence = 0.0

        log.info(f"  Direction:  {direction}")
        log.info(f"  Confidence: {confidence:.4f}")
        log.info(f"  Threshold:  {ML_THRESHOLD}")
        log.info(f"  Signal:     {'TRADE' if confidence >= ML_THRESHOLD else 'NO TRADE'}")
        log.info(f"  -----------------------")

        # Show key feature values
        log.info(f"")
        log.info(f"  ---- KEY FEATURES ----")
        key_features = [
            'H4_trend_aligned', 'H4_ema_cross_8_21', 'H4_rsi_slope',
            'H4_macd_diff', 'H1_ema_cross_8_21', 'H1_rsi_14',
            'tick_imbal_3h', 'spread_3h', 'overnight_ret',
            'last_hour_ret', 'overnight_macd'
        ]
        for f in key_features:
            val = row_dict.get(f, 'N/A')
            if isinstance(val, float):
                log.info(f"  {f:<25} {val:.4f}")
            else:
                log.info(f"  {f:<25} {val}")

        # ATR info
        atr_col = next((c for c in h1_bars.columns
                    if 'atr_14' in c.lower()), None)
        if atr_col:
            atr = float(h1_bars[atr_col].iloc[-1])
        else:
            import ta
            atr = float(ta.volatility.average_true_range(
                h1_bars['high'], h1_bars['low'],
                h1_bars['close'], 14
            ).iloc[-1])

        account  = mt5.account_info()
        capital  = account.balance
        risk_usd = capital * RISK_PER_TRADE
        sl_pips  = atr * 10000 * SL_R
        lot_size = risk_usd / (sl_pips * 10.0)

        log.info(f"")
        log.info(f"  ---- TRADE SIZING ----")
        log.info(f"  Capital:    ${capital:,.2f}")
        log.info(f"  Risk (1%):  ${risk_usd:.2f}")
        log.info(f"  ATR (H1):   {atr:.5f}  ({atr*10000:.1f} pips)")
        log.info(f"  SL pips:    {sl_pips:.1f}")
        log.info(f"  TP pips:    {sl_pips*TP_R:.1f}")
        log.info(f"  Lot size:   {lot_size:.2f}")
        log.info(f"  -----------------------")

        if confidence >= ML_THRESHOLD and direction != 'none':
            log.info(f"")
            log.info(f"  TEST MODE: Would place {direction.upper()} order")
            log.info(f"  Set TEST_MODE=False to execute real orders")
        else:
            log.info(f"")
            log.info(f"  TEST MODE: No trade — confidence below threshold")

        disconnect_mt5()
        exit()


    try:
        while True:
            now = datetime.now(timezone.utc)

            # Reset daily flag at midnight
            if last_date != now.date():
                traded_today = False
                last_date    = now.date()
                log.info(f"New day: {now.date()} | "
                         f"Month: {now.month} | "
                         f"Day: {now.strftime('%A')}")

            # Skip weekends
            if now.weekday() >= 5:
                time.sleep(60)
                continue

            # Skip months with poor performance
            if now.month in SKIP_MONTHS:
                log.info(f"Skipping month {now.month}")
                time.sleep(3600)
                continue

            # Skip Friday
            if now.weekday() == 4:
                time.sleep(300)
                continue

            # Only act at London open
            if now.hour not in ENTRY_HOURS or now.minute > 5:
                time.sleep(30)
                continue

            # One trade per day
            if traded_today:
                time.sleep(300)
                continue

            # Already have a position
            if has_open_position(SYMBOL, MAGIC):
                log.info("Position already open — skipping")
                traded_today = True
                time.sleep(300)
                continue

            log.info(f"London open detected: {now.strftime('%Y-%m-%d %H:%M')} UTC")

            # ---- Check spread ----
            spread = get_current_spread(SYMBOL)
            log.info(f"Current spread: {spread:.1f} pips")
            if spread > SPREAD_LIMIT:
                log.warning(f"Spread too wide ({spread:.1f} pips) — skipping")
                time.sleep(300)
                continue

            # ---- Fetch live data ----
            log.info("Fetching live market data...")
            h1_bars = get_h1_bars(SYMBOL, 300)
            h4_bars = get_h4_bars(SYMBOL, 200)
            m5_bars = get_m5_bars(SYMBOL, 500)

            if h1_bars is None or h4_bars is None:
                log.error("Failed to fetch bars — skipping")
                time.sleep(300)
                continue

            # Merge M5 live with historical for overnight context
            if len(m5_hist) > 0 and m5_bars is not None:
                m5_combined = pd.concat([m5_hist, m5_bars])
                m5_combined = m5_combined[~m5_combined.index.duplicated(keep='last')]
                m5_combined = m5_combined.sort_index()
            else:
                m5_combined = m5_bars if m5_bars is not None else pd.DataFrame()

            # ---- Build feature vector ----
            log.info("Computing features and model prediction...")
            try:
                x_vec, row_dict = build_feature_vector(
                    h1_bars, h4_bars, m5_combined, tick_df, now
                )
            except Exception as e:
                log.error(f"Feature computation failed: {e}")
                time.sleep(300)
                continue

            # ---- Model prediction ----
            x_scaled = scaler.transform(x_vec.reshape(1, -1))
            prob     = model.predict_proba(x_scaled)[0][1]
            regime   = get_regime(row_dict)

            log.info(f"Regime: {regime} | ML prob: {prob:.3f}")

            # ---- Direction + confidence ----
            if regime == 'bull':
                direction  = 'long'
                confidence = prob
            elif regime == 'bear':
                direction  = 'short'
                confidence = 1 - prob
            else:
                log.info("Ranging regime — no trade")
                traded_today = True
                time.sleep(300)
                continue

            log.info(f"Direction: {direction} | Confidence: {confidence:.3f}")

            # ---- ML filter ----
            if confidence < ML_THRESHOLD:
                log.info(f"Confidence {confidence:.3f} below threshold "
                         f"{ML_THRESHOLD} — no trade")
                traded_today = True
                time.sleep(300)
                continue

            # ---- ATR for position sizing ----
            atr_col = next((c for c in h1_bars.columns
                           if 'atr_14' in c.lower()), None)
            if atr_col:
                atr = float(h1_bars[atr_col].iloc[-1])
            else:
                import ta
                atr = float(ta.volatility.average_true_range(
                    h1_bars['high'], h1_bars['low'],
                    h1_bars['close'], 14
                ).iloc[-1])

            if np.isnan(atr) or atr == 0:
                log.error("ATR is zero or NaN — skipping")
                time.sleep(300)
                continue

            # ---- Position sizing ----
            account   = mt5.account_info()
            capital   = account.balance
            risk_usd  = capital * RISK_PER_TRADE
            sl_pips   = atr * 10000 * SL_R
            lot_size  = risk_usd / (sl_pips * 10.0)

            # ---- SL / TP prices ----
            tick      = mt5.symbol_info_tick(SYMBOL)
            if direction == 'long':
                entry    = tick.ask
                sl_price = entry - atr * SL_R
                tp_price = entry + atr * TP_R
            else:
                entry    = tick.bid
                sl_price = entry + atr * SL_R
                tp_price = entry - atr * TP_R

            log.info(f"Trade setup:")
            log.info(f"  Entry:    {entry:.5f}")
            log.info(f"  SL:       {sl_price:.5f} ({sl_pips:.1f} pips)")
            log.info(f"  TP:       {tp_price:.5f} ({sl_pips*TP_R:.1f} pips)")
            log.info(f"  Lots:     {lot_size:.2f}")
            log.info(f"  Risk:     ${risk_usd:.2f} ({RISK_PER_TRADE*100:.0f}%)")
            log.info(f"  Capital:  ${capital:,.2f}")

            # ---- Place order ----
            result = place_order(
                SYMBOL, direction, lot_size,
                sl_price, tp_price,
                f"{COMMENT}_{confidence:.2f}"
            )

            if result:
                traded_today = True
                log.info(f"Trade placed successfully — ticket: {result.order}")
            else:
                log.error("Trade placement failed")

            time.sleep(300)

    except KeyboardInterrupt:
        log.info("Bot stopped by user")
    finally:
        disconnect_mt5()

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_bot()