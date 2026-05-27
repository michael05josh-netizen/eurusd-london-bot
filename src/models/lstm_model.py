import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

# ================================================================
# CONFIG
# ================================================================
DATA_PATH = "data/featured/EURUSD_final_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN    = 50
HORIZON    = 6
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 5e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ================================================================
# LOAD DATA
# ================================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")
print(f"Range: {df.index[0]} → {df.index[-1]}")

# ================================================================
# DEFINE FEATURE GROUPS
# ================================================================
# Columns to exclude from features
EXCLUDE = ['label', 'open', 'high', 'low', 'close', 'volume',
           'hour', 'day_of_week']

# M5 sequence features — what LSTM processes over 50 candles
M5_SEQ_COLS = [c for c in df.columns
               if c not in EXCLUDE
               and not c.startswith('H1_')
               and not c.startswith('H4_')]

# Context features — H1/H4 snapshot + cross-TF + tick features
CTX_COLS = [c for c in df.columns
            if c not in EXCLUDE
            and c not in M5_SEQ_COLS]

print(f"\nM5 sequence features: {len(M5_SEQ_COLS)}")
print(f"Context features:     {len(CTX_COLS)}")
print(f"Sample M5 cols: {M5_SEQ_COLS[:5]}")
print(f"Sample CTX cols: {CTX_COLS[:5]}")

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================
split    = int(len(df) * 0.80)
df_train = df.iloc[:split].copy()
df_test  = df.iloc[split:].copy()

print(f"\nTrain: {len(df_train):,}  Test: {len(df_test):,}")
print(f"Test period: {df_test.index[0]} → {df_test.index[-1]}")
print(f"Train label balance: {df_train['label'].mean()*100:.1f}% up")
print(f"Test  label balance: {df_test['label'].mean()*100:.1f}% up")

# ================================================================
# SCALE
# ================================================================
print("\nFitting scalers...")
scaler_m5  = StandardScaler()
scaler_ctx = StandardScaler()

df_train[M5_SEQ_COLS] = scaler_m5.fit_transform(df_train[M5_SEQ_COLS])
df_test[M5_SEQ_COLS]  = scaler_m5.transform(df_test[M5_SEQ_COLS])

df_train[CTX_COLS] = scaler_ctx.fit_transform(df_train[CTX_COLS])
df_test[CTX_COLS]  = scaler_ctx.transform(df_test[CTX_COLS])

joblib.dump(scaler_m5,   f"{MODEL_DIR}/lstm_scaler_m5.pkl")
joblib.dump(scaler_ctx,  f"{MODEL_DIR}/lstm_scaler_ctx.pkl")
joblib.dump(M5_SEQ_COLS, f"{MODEL_DIR}/lstm_m5_cols.pkl")
joblib.dump(CTX_COLS,    f"{MODEL_DIR}/lstm_ctx_cols.pkl")

# ================================================================
# DATASET
# ================================================================
class ForexDataset(Dataset):
    def __init__(self, df, m5_cols, ctx_cols, seq_len):
        self.seq_len  = seq_len
        self.m5_vals  = df[m5_cols].values.astype(np.float32)
        self.ctx_vals = df[ctx_cols].values.astype(np.float32)
        self.labels   = df['label'].values.astype(np.float32)
        self.n        = len(df) - seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = self.m5_vals[idx : idx + self.seq_len]
        ctx = self.ctx_vals[idx + self.seq_len - 1]
        lbl = self.labels[idx + self.seq_len - 1]
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(ctx, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.float32)
        )

train_ds = ForexDataset(df_train, M5_SEQ_COLS, CTX_COLS, SEQ_LEN)
test_ds  = ForexDataset(df_test,  M5_SEQ_COLS, CTX_COLS, SEQ_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  num_workers=0, pin_memory=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=False)

print(f"Train batches: {len(train_dl)}  Test batches: {len(test_dl)}")

# ================================================================
# MODEL
# ================================================================
class LSTMScalper(nn.Module):
    def __init__(self, m5_features, ctx_features,
                 lstm_hidden=128, lstm_layers=2,
                 ctx_hidden=64, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = m5_features,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_features, ctx_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_hidden, ctx_hidden),
            nn.ReLU(),
        )

        fusion_in = lstm_hidden + ctx_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, ctx):
        lstm_out, _  = self.lstm(seq)
        lstm_last    = self.lstm_norm(lstm_out[:, -1, :])
        ctx_out      = self.ctx_net(ctx)
        fused        = torch.cat([lstm_last, ctx_out], dim=1)
        return self.head(fused).squeeze(1)

    def get_logits(self, seq, ctx):
        lstm_out, _  = self.lstm(seq)
        lstm_last    = self.lstm_norm(lstm_out[:, -1, :])
        ctx_out      = self.ctx_net(ctx)
        fused        = torch.cat([lstm_last, ctx_out], dim=1)
        # Return pre-sigmoid logits
        x = fused
        for layer in self.head[:-1]:
            x = layer(x)
        return x.squeeze(1)

model = LSTMScalper(
    m5_features  = len(M5_SEQ_COLS),
    ctx_features = len(CTX_COLS)
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ================================================================
# TRAINING
# ================================================================
n_pos      = (df_train['label'] == 1).sum()
n_neg      = (df_train['label'] == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"Pos weight: {pos_weight.item():.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

best_auc   = 0.0
best_epoch = 0
no_improve = 0
patience   = 12

print(f"\nTraining for up to {EPOCHS} epochs on {DEVICE}...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val AUC':>10} {'LR':>12}")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for seq, ctx, lbl in train_dl:
        seq, ctx, lbl = seq.to(DEVICE), ctx.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        logits = model.get_logits(seq, ctx)
        loss   = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    # --- Validate ---
    model.eval()
    val_loss  = 0.0
    all_probs = []
    all_lbls  = []
    with torch.no_grad():
        for seq, ctx, lbl in test_dl:
            seq, ctx, lbl = seq.to(DEVICE), ctx.to(DEVICE), lbl.to(DEVICE)
            probs  = model(seq, ctx)
            logits = model.get_logits(seq, ctx)
            loss   = criterion(logits, lbl)
            val_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_lbls.extend(lbl.cpu().numpy())

    val_loss /= len(test_dl)
    val_auc   = roc_auc_score(all_lbls, all_probs)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_auc)

    print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
          f"{val_auc:>10.4f} {current_lr:>12.6f}")

    if val_auc > best_auc:
        best_auc   = val_auc
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_scalper_best.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nBest AUC: {best_auc:.4f} at epoch {best_epoch}")

# ================================================================
# FINAL EVALUATION
# ================================================================
model.load_state_dict(torch.load(
    f"{MODEL_DIR}/lstm_scalper_best.pt", map_location=DEVICE
))
model.eval()

all_probs, all_lbls = [], []
with torch.no_grad():
    for seq, ctx, lbl in test_dl:
        seq, ctx = seq.to(DEVICE), ctx.to(DEVICE)
        probs    = model(seq, ctx)
        all_probs.extend(probs.cpu().numpy())
        all_lbls.extend(lbl.numpy())

all_probs = np.array(all_probs)
all_lbls  = np.array(all_lbls)

print(f"\n--- Final Test Results ---")
print(f"ROC-AUC: {roc_auc_score(all_lbls, all_probs):.4f}")
print(f"\nThreshold analysis:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} "
      f"{'Trades':>10} {'Win%':>8}")
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    preds  = (all_probs >= thresh).astype(int)
    trades = preds.sum()
    if trades == 0:
        continue
    wins = ((preds == 1) & (all_lbls == 1)).sum()
    prec = wins / trades
    rec  = wins / max((all_lbls == 1).sum(), 1)
    print(f"{thresh:>10.2f} {prec:>10.3f} {rec:>10.3f} "
          f"{trades:>10,} {prec*100:>7.1f}%")

print("\nClassification Report (threshold=0.50):")
print(classification_report(
    all_lbls, (all_probs >= 0.5).astype(int),
    target_names=['Down', 'Up']
))

torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_scalper_final.pt")
print(f"Saved → {MODEL_DIR}/lstm_scalper_best.pt")
print(f"Saved → {MODEL_DIR}/lstm_scalper_final.pt")