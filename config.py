# ================================================================
# config.py — Central configuration for EURUSD London Session Bot
# ================================================================

# --- Paths ---
DATA_RAW_DIR      = "data/raw"
DATA_FEAT_DIR     = "data/featured"
MODEL_DIR         = "models"
LOG_DIR           = "logs"
BACKTEST_OUT_DIR  = "backtest_results"

TICK_FEATURES     = "data/raw/EURUSD_tick_features_2023_2025.csv"
M5_FEATURES       = "data/raw/EURUSD_M5_clean.csv"
H1_FEATURES       = "data/raw/EURUSD_H1.csv"
H4_FEATURES       = "data/raw/EURUSD_H4.csv"
SESSION_FEATURES  = "data/featured/EURUSD_lean_session.csv"

MODEL_PATH        = "models/lgbm_session.pkl"
SCALER_PATH       = "models/lgbm_session_scaler.pkl"
FEAT_PATH         = "models/lgbm_session_features.pkl"

# --- Trading ---
SYMBOL            = "EURUSD"
MAGIC             = 20250101
COMMENT           = "LondonSessionBot"

# --- Strategy ---
ENTRY_HOURS       = [7, 8]       # London open UTC
RISK_PER_TRADE    = 0.01         # 1% risk
TP_R              = 1.5          # TP = 1.5 × ATR
SL_R              = 1.0          # SL = 1.0 × ATR
ML_THRESHOLD      = 0.70         # minimum confidence
SPREAD_LIMIT      = 2.0          # max spread pips
MAX_HOLD_BARS     = 13           # max H1 bars to hold

# --- Backtest ---
INITIAL_CAPITAL   = 10_000
BACKTEST_START    = "2025-06-13"  # true OOS start
SPREAD_COST_PIPS  = 1.2          # spread + commission

# --- Model ---
LGBM_PARAMS = {
    'n_estimators':       500,
    'max_depth':          3,
    'learning_rate':      0.02,
    'subsample':          0.7,
    'colsample_bytree':   0.6,
    'min_child_samples':  8,
    'reg_alpha':          0.1,
    'reg_lambda':         0.1,
    'random_state':       42,
    'n_jobs':             -1,
    'verbose':            -1,
}