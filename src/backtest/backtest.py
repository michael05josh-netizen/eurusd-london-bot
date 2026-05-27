import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os

DATA_PATH   = "data/featured/EURUSD_lean_session.csv"
H1_PATH     = "data/raw/EURUSD_H1.csv"
MODEL_PATH  = "models/lgbm_session.pkl"
SCALER_PATH = "models/lgbm_session_scaler.pkl"
FEAT_PATH   = "models/lgbm_session_features.pkl"
OUT_DIR     = "backtest_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
# CONFIG
# ================================================================
INITIAL_CAPITAL = 10_000
RISK_PER_TRADE  = 0.01      # 1% risk per trade
TP_R            = 1.5
SL_R            = 1.0
SPREAD_PIPS     = 1.2       # realistic spread + commission
PIP_VALUE       = 10.0      # per standard lot
# Updated filters
ML_THRESHOLD = 0.70   # only take high-confidence trades
SKIP_MONTHS = []  # No hardcoded month filters

# CRITICAL: Only backtest on data the model never saw
# Model trained on 80% (up to 2025-06-12)
# We only trust results AFTER this date
BACKTEST_START  = "2025-06-13"

# ================================================================
# LOAD
# ================================================================
print("Loading data and model...")
df_feat = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df_h1   = pd.read_csv(H1_PATH,   index_col=0, parse_dates=True)
model   = joblib.load(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)
feats   = joblib.load(FEAT_PATH)

print(f"Feature data: {len(df_feat):,} sessions")
print(f"H1 data:      {len(df_h1):,} bars")

# ================================================================
# FEATURE ENGINEERING (match training exactly)
# ================================================================
def add_interaction_features(df):
    df = df.copy()
    df['month']   = df.index.month
    df['is_feb']  = (df.index.month == 2).astype(int)
    df['is_jun']  = (df.index.month == 6).astype(int)
    df['is_mar']  = (df.index.month == 3).astype(int)
    df['is_q1']   = (df.index.month.isin([1,2,3])).astype(int)
    df['is_q2']   = (df.index.month.isin([4,5,6])).astype(int)

    if 'H4_trend_aligned' in df.columns and 'overnight_ret' in df.columns:
        df['trend_x_overnight'] = df['H4_trend_aligned'] * df['overnight_ret'] * 10000
    if 'H4_rsi_slope' in df.columns and 'last_hour_ret' in df.columns:
        df['h4rsi_x_lasthour']  = df['H4_rsi_slope'] * df['last_hour_ret'] * 10000
    if 'tick_imbal_3h' in df.columns and 'H4_trend_aligned' in df.columns:
        df['tick_x_trend']      = df['tick_imbal_3h'] * df['H4_trend_aligned']
    if 'spread_3h' in df.columns and 'overnight_range' in df.columns:
        df['spread_x_range']    = df['spread_3h'] * df['overnight_range'] * 10000
    if 'H4_atr_ratio' in df.columns and 'H4_rsi_slope' in df.columns:
        df['atr_x_rsi']         = df['H4_atr_ratio'] * df['H4_rsi_slope']
    if 'H4_trend_aligned' in df.columns:
        df['month_x_trend']     = df['month'] * df['H4_trend_aligned']
    if 'overnight_rsi' in df.columns and 'overnight_macd' in df.columns:
        df['ovn_rsi_x_macd']    = (df['overnight_rsi'] - 50) * np.sign(df['overnight_macd'])
    if 'H1_dist_high_20' in df.columns and 'H1_atr_ratio' in df.columns:
        df['dist_x_atr']        = df['H1_dist_high_20'] * df['H1_atr_ratio']

    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
    df[feats] = df[feats].fillna(0)
    return df

df_feat = add_interaction_features(df_feat)

# ================================================================
# MODEL PREDICTIONS
# ================================================================
X         = df_feat[feats].values
X_s       = scaler.transform(X)
probs     = model.predict_proba(X_s)[:, 1]
df_feat['ml_prob'] = probs

# ================================================================
# REGIME DETECTION
# ================================================================
def get_regime(row):
    h4_bull = row.get('H4_trend_aligned', 0)
    h4_ema  = row.get('H4_ema_cross_8_21', 0)
    h4_rsi  = row.get('H4_rsi_slope', 0)
    if h4_bull == 1 and h4_ema == 1:
        return 'bull'
    elif h4_ema == 0 and h4_rsi < -0.3:
        return 'bear'
    else:
        return 'ranging'

df_feat['regime'] = df_feat.apply(get_regime, axis=1)

# ================================================================
# FILTER TO TRUE OUT-OF-SAMPLE PERIOD
# ================================================================
df_oos = df_feat[df_feat.index >= BACKTEST_START].copy()
print(f"\nOut-of-sample period: {df_oos.index[0]} → {df_oos.index[-1]}")
print(f"OOS sessions: {len(df_oos):,}")
print(f"Regime distribution:\n{df_oos['regime'].value_counts().to_string()}")

# ================================================================
# PROPER FORWARD SIMULATION
# For each London open bar in OOS period:
# 1. Get model signal (features known at bar close)
# 2. Determine direction from regime
# 3. Enter trade at NEXT bar open (realistic)
# 4. Simulate trade forward on H1 bars
# ================================================================
print("\nRunning forward simulation...")

# ATR column
atr_col = next((c for c in df_oos.columns if 'H1_atr_14' in c), None) or \
          next((c for c in df_oos.columns if 'atr_14' in c.lower()), None)
print(f"ATR column: {atr_col}")

capital   = INITIAL_CAPITAL
trades    = []
equity_ts = [{'datetime': pd.Timestamp(BACKTEST_START, tz='UTC'), 'equity': capital}]

# Align H1 data for forward simulation
df_h1_oos = df_h1[df_h1.index >= BACKTEST_START].copy()
h1_index  = df_h1_oos.index.tolist()

for ts, row in df_oos.iterrows():
    regime   = row['regime']
    ml_prob  = row['ml_prob']
    atr      = row.get(atr_col, np.nan)
    month    = ts.month

    # Skip ranging markets
    if regime == 'ranging':
        continue

    # Determine direction and confidence
    if regime == 'bull':
        direction  = 'long'
        confidence = ml_prob
    else:
        direction  = 'short'
        confidence = 1 - ml_prob

    # ML filter
    if confidence < ML_THRESHOLD:
        continue

    # NEW — skip only when current ATR is abnormally low
    # (indicates dead/choppy market regardless of month)
    atr_vs_mean = row.get('H1_atr_vs_mean', 1.0)
    if atr_vs_mean < 0.6:   # ATR less than 60% of its 50-period mean
        np.logo(f"Low volatility environment (ATR ratio: {atr_vs_mean:.2f}) — skipping")
        continue

    if np.isnan(atr) or atr == 0:
        continue

    # Skip the 0.60-0.65 dead zone
    if 0.60 <= confidence <= 0.65:
        continue

    # Disable shorts for now
    if direction == 'short':
        continue

    # ---- Find entry bar in H1 data ----
    # Entry = next H1 bar after signal bar
    try:
        signal_pos = h1_index.index(ts)
        entry_pos  = signal_pos + 1
    except ValueError:
        # ts not in h1_index exactly — find nearest
        h1_arr = np.array(h1_index)
        pos    = np.searchsorted(h1_arr, ts)
        entry_pos = pos + 1

    if entry_pos >= len(h1_index):
        continue

    entry_ts    = h1_index[entry_pos]
    entry_bar   = df_h1_oos.loc[entry_ts] if entry_ts in df_h1_oos.index else None
    if entry_bar is None:
        continue

    entry_price = float(entry_bar['open'])

    # ---- SL / TP ----
    sl_pips = atr * 10000 * SL_R
    tp_pips = atr * 10000 * TP_R

    if direction == 'long':
        sl_price = entry_price - atr * SL_R
        tp_price = entry_price + atr * TP_R
    else:
        sl_price = entry_price + atr * SL_R
        tp_price = entry_price - atr * TP_R

    # ---- Simulate forward on H1 bars ----
    won        = None
    exit_price = None
    exit_ts    = None
    MAX_BARS   = 13

    for j in range(1, MAX_BARS + 1):
        bar_pos = entry_pos + j
        if bar_pos >= len(h1_index):
            break

        bar_ts  = h1_index[bar_pos]
        bar     = df_h1_oos.loc[bar_ts] if bar_ts in df_h1_oos.index else None
        if bar is None:
            break

        # Exit at NY close (21:00) or next day
        if bar_ts.hour >= 21 or bar_ts.date() != entry_ts.date():
            # Time exit at bar close
            won        = None  # no result — time exit
            exit_price = float(bar['close'])
            exit_ts    = bar_ts
            break

        bar_high = float(bar['high'])
        bar_low  = float(bar['low'])

        if direction == 'long':
            if bar_low <= sl_price:
                won = False; exit_price = sl_price; exit_ts = bar_ts; break
            if bar_high >= tp_price:
                won = True;  exit_price = tp_price; exit_ts = bar_ts; break
        else:
            if bar_high >= sl_price:
                won = False; exit_price = sl_price; exit_ts = bar_ts; break
            if bar_low <= tp_price:
                won = True;  exit_price = tp_price; exit_ts = bar_ts; break

    # Skip time exits (neither SL nor TP hit)
    if won is None:
        equity_ts.append({'datetime': ts, 'equity': capital})
        continue

    # ---- Position sizing ----
    risk_usd    = capital * RISK_PER_TRADE
    lot_size    = risk_usd / (sl_pips * PIP_VALUE)
    lot_size    = round(min(lot_size, 10.0), 2)
    spread_cost = SPREAD_PIPS * PIP_VALUE * lot_size

    # ---- P&L ----
    if won:
        pnl = (tp_pips * PIP_VALUE * lot_size) - spread_cost
    else:
        pnl = -(sl_pips * PIP_VALUE * lot_size) - spread_cost

    capital += pnl

    trades.append({
        'entry_time': entry_ts,
        'exit_time':  exit_ts,
        'direction':  direction,
        'regime':     regime,
        'ml_prob':    round(ml_prob, 3),
        'confidence': round(confidence, 3),
        'month':      month,
        'won':        won,
        'entry_price':round(entry_price, 5),
        'exit_price': round(exit_price, 5),
        'sl_price':   round(sl_price, 5),
        'tp_price':   round(tp_price, 5),
        'sl_pips':    round(sl_pips, 1),
        'tp_pips':    round(tp_pips, 1),
        'lot_size':   round(lot_size, 2),
        'pnl':        round(pnl, 2),
        'capital':    round(capital, 2),
    })

    equity_ts.append({'datetime': ts, 'equity': capital})

# ================================================================
# RESULTS
# ================================================================
df_trades = pd.DataFrame(trades)
df_equity = pd.DataFrame(equity_ts).sort_values('datetime')

if len(df_trades) == 0:
    print("\nNo trades taken in OOS period.")
    print("Check: regime distribution, ML threshold, ATR column")
    exit()

total_trades  = len(df_trades)
wins          = df_trades['won'].sum()
win_rate      = wins / total_trades
total_pnl     = df_trades['pnl'].sum()
final_cap     = capital
total_return  = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# Drawdown
df_equity['equity'] = df_equity['equity'].ffill()
roll_max   = df_equity['equity'].cummax()
drawdown   = (df_equity['equity'] - roll_max) / roll_max * 100
max_dd     = drawdown.min()

# Profit factor
gp = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
gl = df_trades[df_trades['pnl'] < 0]['pnl'].abs().sum()
pf = gp / max(gl, 0.01)

# Sharpe
df_trades['date'] = pd.to_datetime(df_trades['entry_time']).dt.date
daily_pnl = df_trades.groupby('date')['pnl'].sum()
sharpe    = daily_pnl.mean() / max(daily_pnl.std(), 0.01) * np.sqrt(252)

# Avg trade
avg_win  = df_trades[df_trades['won']]['pnl'].mean()
avg_loss = df_trades[~df_trades['won']]['pnl'].mean()

import itertools
won_arr = df_trades['won'].values.tolist()
max_cw  = max((sum(1 for _ in g) for k,g in
               itertools.groupby(won_arr) if k), default=0)
max_cl  = max((sum(1 for _ in g) for k,g in
               itertools.groupby(won_arr) if not k), default=0)

print(f"\n{'='*58}")
print(f"  TRUE OOS BACKTEST — EURUSD London Session Bot")
print(f"  Period: {BACKTEST_START} → {df_trades['exit_time'].max().date()}")
print(f"{'='*58}")
print(f"  Initial capital:   ${INITIAL_CAPITAL:>10,.2f}")
print(f"  Final capital:     ${final_cap:>10,.2f}")
print(f"  Total return:      {total_return:>+10.2f}%")
print(f"  Total P&L:         ${total_pnl:>+10,.2f}")
print(f"{'='*58}")
print(f"  Total trades:      {total_trades}")
print(f"  Win rate:          {win_rate*100:.1f}%")
print(f"  Profit factor:     {pf:.2f}")
print(f"  Sharpe ratio:      {sharpe:.2f}")
print(f"  Max drawdown:      {max_dd:.2f}%")
print(f"  Avg win:           ${avg_win:+,.2f}")
print(f"  Avg loss:          ${avg_loss:+,.2f}")
print(f"  Max consec. wins:  {max_cw}")
print(f"  Max consec. losses:{max_cl}")
print(f"{'='*58}")

print(f"\nBy direction:")
for d in ['long','short']:
    sub = df_trades[df_trades['direction'] == d]
    if len(sub) == 0: continue
    print(f"  {d:6}: {len(sub):3} trades | "
          f"{sub['won'].mean()*100:.1f}% WR | "
          f"${sub['pnl'].sum():+,.2f}")

print(f"\nBy month:")
mo = df_trades.groupby('month').agg(
    trades=('won','count'),
    win_rate=('won','mean'),
    pnl=('pnl','sum')
)
mo['win_rate'] = (mo['win_rate']*100).round(1)
mo['pnl']      = mo['pnl'].round(2)
print(mo.to_string())

print(f"\nBy confidence band:")
df_trades['conf_band'] = pd.cut(
    df_trades['confidence'],
    bins=[0.55, 0.60, 0.65, 0.70, 1.0],
    labels=['0.55-0.60','0.60-0.65','0.65-0.70','0.70+']
)
cb = df_trades.groupby('conf_band', observed=True).agg(
    trades=('won','count'),
    win_rate=('won','mean'),
    pnl=('pnl','sum')
)
cb['win_rate'] = (cb['win_rate']*100).round(1)
print(cb.to_string())

# ================================================================
# CHARTS
# ================================================================
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Equity curve
ax1 = fig.add_subplot(gs[0, :])
eq  = df_equity.set_index('datetime')['equity'].ffill()
ax1.plot(eq.index, eq.values, '#2196F3', linewidth=1.5)
ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(eq.index, INITIAL_CAPITAL, eq.values,
                 where=eq.values >= INITIAL_CAPITAL,
                 alpha=0.15, color='green')
ax1.fill_between(eq.index, INITIAL_CAPITAL, eq.values,
                 where=eq.values < INITIAL_CAPITAL,
                 alpha=0.15, color='red')
ax1.set_title(f'Equity Curve (OOS) — ${final_cap:,.0f} '
              f'({total_return:+.1f}%)', fontsize=12)
ax1.set_ylabel('Capital ($)')
ax1.grid(True, alpha=0.3)

# Drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(df_equity['datetime'], drawdown.values, 0,
                 color='red', alpha=0.4)
ax2.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=11)
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# Monthly P&L
ax3 = fig.add_subplot(gs[2, 0])
mp  = df_trades.groupby('month')['pnl'].sum()
ax3.bar(mp.index, mp.values,
        color=['green' if v >= 0 else 'red' for v in mp.values], alpha=0.7)
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title('P&L by Month (OOS)', fontsize=11)
ax3.set_xlabel('Month'); ax3.set_ylabel('P&L ($)')
ax3.grid(True, alpha=0.3, axis='y')

# Confidence band win rate
ax4 = fig.add_subplot(gs[2, 1])
cb_wr = df_trades.groupby('conf_band', observed=True)['won'].mean() * 100
ax4.bar(range(len(cb_wr)), cb_wr.values,
        color=['green' if v >= 50 else 'red' for v in cb_wr.values], alpha=0.7)
ax4.axhline(50, color='black', linestyle='--', linewidth=0.8)
ax4.set_title('Win Rate by Confidence Band', fontsize=11)
ax4.set_xticks(range(len(cb_wr)))
ax4.set_xticklabels(cb_wr.index, fontsize=8)
ax4.set_ylabel('Win Rate %')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('EURUSD London Session Bot — True OOS Backtest',
             fontsize=13, fontweight='bold')
plt.savefig(f"{OUT_DIR}/oos_backtest.png", dpi=150, bbox_inches='tight')
plt.close()

df_trades.to_csv(f"{OUT_DIR}/oos_trade_log.csv")
print(f"\nChart  → {OUT_DIR}/oos_backtest.png")
print(f"Trades → {OUT_DIR}/oos_trade_log.csv")