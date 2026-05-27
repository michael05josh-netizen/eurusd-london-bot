import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = "data/featured/EURUSD_lean_session.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================================
# LOAD
# ================================================================
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")
print(f"Baseline long win rate: {df['label'].mean()*100:.1f}%")

# ================================================================
# FEATURE ENGINEERING — add interaction features
# Based on deep_analysis: signal is in combinations not individuals
# ================================================================
print("\nAdding interaction features...")

# Month — critical (Feb and Jun have massive edge)
df['month']      = df.index.month
df['is_feb']     = (df.index.month == 2).astype(int)
df['is_jun']     = (df.index.month == 6).astype(int)
df['is_mar']     = (df.index.month == 3).astype(int)
df['is_q1']      = (df.index.month.isin([1,2,3])).astype(int)
df['is_q2']      = (df.index.month.isin([4,5,6])).astype(int)

# Key interaction features
# H4 trend aligned × overnight momentum
if 'H4_trend_aligned' in df.columns and 'overnight_ret' in df.columns:
    df['trend_x_overnight'] = df['H4_trend_aligned'] * df['overnight_ret'] * 10000

# H4 RSI slope × last hour return
if 'H4_rsi_slope' in df.columns and 'last_hour_ret' in df.columns:
    df['h4rsi_x_lasthour'] = df['H4_rsi_slope'] * df['last_hour_ret'] * 10000

# Tick imbalance × H4 trend
if 'tick_imbal_3h' in df.columns and 'H4_trend_aligned' in df.columns:
    df['tick_x_trend'] = df['tick_imbal_3h'] * df['H4_trend_aligned']

# Spread context × overnight range
if 'spread_3h' in df.columns and 'overnight_range' in df.columns:
    df['spread_x_range'] = df['spread_3h'] * df['overnight_range'] * 10000

# ATR regime × RSI slope (volatility + momentum)
if 'H4_atr_ratio' in df.columns and 'H4_rsi_slope' in df.columns:
    df['atr_x_rsi'] = df['H4_atr_ratio'] * df['H4_rsi_slope']

# Month × trend alignment
if 'H4_trend_aligned' in df.columns:
    df['month_x_trend'] = df['month'] * df['H4_trend_aligned']

# Overnight RSI × MACD (momentum confluence)
if 'overnight_rsi' in df.columns and 'overnight_macd' in df.columns:
    df['ovn_rsi_x_macd'] = (df['overnight_rsi'] - 50) * np.sign(df['overnight_macd'])

# H1 distance from high × ATR ratio (mean reversion context)
if 'H1_dist_high_20' in df.columns and 'H1_atr_ratio' in df.columns:
    df['dist_x_atr'] = df['H1_dist_high_20'] * df['H1_atr_ratio']

print(f"Features after interactions: {df.shape[1]}")

# ================================================================
# FEATURES / TARGET
# ================================================================
DROP = ['label', 'direction', 'long_result', 'shrt_result',
        'pips_long', 'pips_short', 'open', 'high', 'low', 'close']
FEAT = [c for c in df.columns if c not in DROP]

# Fill any NaN from new interaction features
df[FEAT] = df[FEAT].fillna(0)

X = df[FEAT].values
y = df['label'].values

print(f"Total features: {len(FEAT)}")
print(f"Samples: {len(X):,}")

# ================================================================
# WALK-FORWARD VALIDATION — 5 folds
# ================================================================
print(f"\nWalk-forward validation...")
print(f"{'Fold':>5} {'Train':>8} {'Test':>8} {'AUC':>8} "
      f"{'Base%':>8} {'P@0.55':>8} {'Trades':>8}")
print("-" * 60)

n_splits  = 5
fold_size = len(df) // n_splits
aucs      = []
precs     = []

for fold in range(1, n_splits):
    train_end  = fold * fold_size
    test_start = train_end
    test_end   = min(test_start + fold_size, len(df))

    X_tr = X[:train_end];  y_tr = y[:train_end]
    X_te = X[test_start:test_end]; y_te = y[test_start:test_end]

    if len(X_te) < 30:
        continue

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    pos = max(y_tr.sum(), 1)
    neg = len(y_tr) - pos

    model = lgb.LGBMClassifier(
        n_estimators      = 300,
        max_depth         = 3,
        learning_rate     = 0.03,
        subsample         = 0.7,
        colsample_bytree  = 0.6,
        min_child_samples = 8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        scale_pos_weight  = neg / pos,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1
    )
    model.fit(X_tr_s, y_tr)

    y_prob = model.predict_proba(X_te_s)[:, 1]
    auc    = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5

    preds  = (y_prob >= 0.55).astype(int)
    trades = preds.sum()
    wins   = ((preds == 1) & (y_te == 1)).sum()
    prec   = wins / max(trades, 1)
    base   = y_te.mean() * 100

    aucs.append(auc)
    precs.append(prec)

    print(f"{fold:>5} {train_end:>8,} {len(X_te):>8,} "
          f"{auc:>8.4f} {base:>7.1f}% {prec:>7.1%} {trades:>8,}")

print(f"\nMean AUC:            {np.mean(aucs):.4f}")
print(f"Mean Precision@0.55: {np.mean(precs)*100:.1f}%")
print(f"Baseline win rate:   {y.mean()*100:.1f}%")

# ================================================================
# FINAL MODEL
# ================================================================
print("\n--- Final Model (80/20 split) ---")
split   = int(len(X) * 0.80)
X_tr    = X[:split];    y_tr = y[:split]
X_te    = X[split:];    y_te = y[split:]

print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
print(f"Test period: {df.index[split]} → {df.index[-1]}")
print(f"Test baseline: {y_te.mean()*100:.1f}%")

scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

pos = max(y_tr.sum(), 1)
neg = len(y_tr) - pos

final = lgb.LGBMClassifier(
    n_estimators      = 500,
    max_depth         = 3,
    learning_rate     = 0.02,
    subsample         = 0.7,
    colsample_bytree  = 0.6,
    min_child_samples = 8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    scale_pos_weight  = neg / pos,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1
)
final.fit(X_tr_s, y_tr)

y_prob = final.predict_proba(X_te_s)[:, 1]
auc    = roc_auc_score(y_te, y_prob)

print(f"\nFinal ROC-AUC: {auc:.4f}")
print(f"\nThreshold analysis (baseline = {y_te.mean()*100:.1f}%):")
print(f"{'Thresh':>8} {'Win%':>8} {'vs Base':>9} "
      f"{'Trades':>8} {'Recall':>8}")
print("-" * 50)

for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    preds  = (y_prob >= thresh).astype(int)
    trades = preds.sum()
    if trades == 0:
        continue
    wins = ((preds == 1) & (y_te == 1)).sum()
    prec = wins / trades
    rec  = wins / max((y_te == 1).sum(), 1)
    vs_base = (prec - y_te.mean()) * 100
    print(f"{thresh:>8.2f} {prec*100:>7.1f}% {vs_base:>+8.1f}% "
          f"{trades:>8,} {rec:>8.1%}")

print(f"\nClassification Report (threshold=0.55):")
print(classification_report(
    y_te, (y_prob >= 0.55).astype(int),
    target_names=['Short','Long']
))

# Feature importance
imp = pd.Series(
    final.feature_importances_, index=FEAT
).sort_values(ascending=False)
print(f"\nTop 20 features:")
print(imp.head(20).to_string())

# ================================================================
# EXPECTED VALUE CALCULATION
# ================================================================
print(f"\n--- Expected Value at Each Threshold ---")
print(f"(TP=1.5 × ATR, SL=1.0 × ATR, R:R = 1.5:1)")
print(f"{'Thresh':>8} {'Win%':>8} {'EV/trade':>12} {'Trades':>8}")
print("-" * 45)

for thresh in [0.45, 0.50, 0.55, 0.60, 0.65]:
    preds  = (y_prob >= thresh).astype(int)
    trades = preds.sum()
    if trades < 5:
        continue
    wins = ((preds == 1) & (y_te == 1)).sum()
    wr   = wins / trades
    # EV = (win_rate × 1.5) - (loss_rate × 1.0)
    ev   = (wr * 1.5) - ((1 - wr) * 1.0)
    print(f"{thresh:>8.2f} {wr*100:>7.1f}% {ev:>+11.3f}R {trades:>8,}")

# Save
joblib.dump(final,  f"{MODEL_DIR}/lgbm_session.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/lgbm_session_scaler.pkl")
joblib.dump(FEAT,   f"{MODEL_DIR}/lgbm_session_features.pkl")
print(f"\nSaved → {MODEL_DIR}/lgbm_session.pkl")