"""Execute rnn_final_train.ipynb cells sequentially."""
import os, sys, glob, math, warnings, io, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# === Cell 2: Config ===
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, 'artifacts')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_final')

SL_PIPS = 15.0
TP_PIPS = 20.0
HOLD_BARS = 20
INITIAL_CAPITAL = 700.0
N_FEATURES = 23
N_BARS = 20
N_CLASSES = 2
BATCH_SIZE = 64
FINAL_EPOCHS = 120

BUY_PARAMS = {
    'seq_len': 9,
    'hidden_size': 80,
    'num_layers': 3,
    'dropout': 0.35,
    'lr': 0.002147,
    'conf_threshold': 0.8,
}

SELL_PARAMS = {
    'seq_len': 11,
    'hidden_size': 112,
    'num_layers': 1,
    'dropout': 0.25,
    'lr': 0.003045,
    'conf_threshold': 0.5,
}

print(f'Artifacts: {ARTIFACTS_DIR}')
print(f'Output: {OUTPUT_DIR}')
print(f'BUY params: {BUY_PARAMS}')
print(f'SELL params: {SELL_PARAMS}')

# === Cell 3: NPZ Data Loader ===
def load_and_split_npz(filepath, seq_len, eps=1e-6):
    data = np.load(filepath, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(int)
    results = data['result'].astype(np.float64)
    datetimes = data['datetime']

    n_total = len(X)
    actual_seq = min(seq_len, N_BARS)
    X = X[:, :actual_seq, :]

    n_train = int(n_total * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    results_test = results[n_train:]
    dt_test = datetimes[n_train:]

    # Number of months in test set (fixed to 3 for trades baseline)
    n_months = 3

    num_classes = len(np.unique(y))
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    df_test = pd.DataFrame({'_result': results_test})
    df_test['datetime'] = dt_test

    print(f"  {os.path.basename(filepath)}: {n_total} rows, train={n_train}, test={n_total - n_train}")
    print(f"    X shape: {X.shape}, features={X.shape[-1]}, seq_len={actual_seq}, test_months={n_months}")
    print(f"    Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_oh))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test_oh))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, df_test, actual_seq, N_FEATURES, num_classes, n_months

# Find NPZ files
buy_npz = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, 'data_BUY_*_normalized.npz')))
sell_npz = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, 'data_SELL_*_normalized.npz')))
print(f'BUY:  {buy_npz}')
print(f'SELL: {sell_npz}')

# === Cell 4: GRU Model ===
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# === Cell 5: Train one model ===
def train_one_model(model, train_loader, epochs, lr, device, verbose=True):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    best_acc = 0.0
    best_state = None

    it = tqdm(range(epochs), desc='Train', leave=False) if verbose else range(epochs)
    for epoch in it:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += (y_pred.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
            total += X_batch.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()
        if train_acc > best_acc:
            best_acc = train_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and hasattr(it, 'set_postfix'):
            it.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, train_loss

# === Cell 6: ONNX Export ===
import onnx
import onnxruntime

def export_to_onnx(model, seq_len, n_features, out_path, name):
    model.eval()
    model.to('cpu')
    dummy = torch.randn(1, seq_len, n_features)
    tmp = out_path + '.tmp'
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        torch.onnx.export(model, dummy, tmp,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          opset_version=18)
    finally:
        sys.stdout = old
    m = onnx.load(tmp)
    raw_bytes = m.SerializeToString()
    if b'.onnx.data' in raw_bytes:
        raise RuntimeError(f'{name}: external .data reference!')
    with open(out_path, 'wb') as f:
        f.write(raw_bytes)
    for x in (tmp, tmp + '.data', out_path + '.data'):
        if os.path.exists(x): os.remove(x)
    kb = os.path.getsize(out_path) / 1024
    print(f'  {name}: {out_path} ({kb:.1f} KB)')

def generate_scaler_mqh(buy_seq_len, sell_seq_len, n_features, eps, out_path):
    lines = [
        '//+------------------------------------------------------------------+',
        '//|                                              RNN_Scaler.mqh      |',
        '//|   Pre-normalized NPZ input (RNN_gold).                         |',
        '//|   Сгенерировано: Colab.                                        |',
        '//+------------------------------------------------------------------+',
        '#ifndef RNN_SCALER_MQH',
        '#define RNN_SCALER_MQH',
        '',
        f'#define NN_BUY_SEQ_LEN      {buy_seq_len}       // BUY input window',
        f'#define NN_SELL_SEQ_LEN     {sell_seq_len}       // SELL input window',
        f'#define NN_MAX_SEQ_LEN      {max(buy_seq_len, sell_seq_len)}',
        f'#define NN_FEATURES      {n_features}    // признаков на бар',
        f'#define NN_NORM_STD_EPS  {eps:g}         // epsilon',
        '',
        '/* Direction-neutral aliases for existing data preparation tools. */',
        '#define NN_SEQ_LEN     NN_BUY_SEQ_LEN',
        '',
        '#endif // RNN_SCALER_MQH',
        '',
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  RNN_Scaler.mqh -> {out_path}')

# === Cell 7: Final Training + Export ===
def final_train_and_export(npz_path, params, mode_name):
    seq_len = params['seq_len']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    dropout = params['dropout']
    lr = params['lr']
    conf_thresh = params['conf_threshold']

    print(f"\n{'='*70}")
    print(f"FINAL TRAIN: {mode_name} | s={seq_len} h={hidden_size} L={num_layers} "
          f"dr={dropout:.2f} lr={lr:.6f} th={conf_thresh:.2f}")
    print(f"{'='*70}")

    train_loader, test_loader, df_test, seq_len_actual, n_features, num_classes, n_months = \
        load_and_split_npz(npz_path, seq_len=seq_len)

    model = GRUClassifier(n_features, hidden_size, num_layers, num_classes, dropout)
    model, train_acc, train_loss = train_one_model(
        model, train_loader, FINAL_EPOCHS, lr, DEVICE, verbose=True
    )
    print(f"\n  Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")

    return model, seq_len, n_features

# Create output dirs
onnx_dir = os.path.join(OUTPUT_DIR, 'onnx')
models_dir = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- BUY ---
print("\n\n========== STARTING BUY TRAINING ==========")
buy_model, buy_seq_len, buy_nf = final_train_and_export(buy_npz[0], BUY_PARAMS, 'BUY')

# --- SELL ---
print("\n\n========== STARTING SELL TRAINING ==========")
sell_model, sell_seq_len, sell_nf = final_train_and_export(sell_npz[0], SELL_PARAMS, 'SELL')

# --- Export ONNX ---
print("\n\n========== EXPORTING ONNX ==========")
export_to_onnx(buy_model, buy_seq_len, buy_nf, os.path.join(onnx_dir, 'buy_model.onnx'), 'buy')
export_to_onnx(sell_model, sell_seq_len, sell_nf, os.path.join(onnx_dir, 'sell_model.onnx'), 'sell')

# --- Save .pth ---
torch.save(buy_model.state_dict(), os.path.join(models_dir, 'model_buy_final.pth'))
torch.save(sell_model.state_dict(), os.path.join(models_dir, 'model_sell_final.pth'))
print(f'\nModels saved: {models_dir}/')

# --- Generate RNN_Scaler.mqh ---
generate_scaler_mqh(buy_seq_len, sell_seq_len, buy_nf, 1e-6,
                    os.path.join(OUTPUT_DIR, 'RNN_Scaler.mqh'))

# --- Summary JSON ---
def safe(v):
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v

final_summary = {
    'buy_params': {k: safe(v) for k, v in BUY_PARAMS.items()},
    'sell_params': {k: safe(v) for k, v in SELL_PARAMS.items()},
    'final_epochs': FINAL_EPOCHS,
    'sl_pips': SL_PIPS,
    'tp_pips': TP_PIPS,
}
with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(final_summary, f, indent=2, default=safe)
print(f'Summary: {OUTPUT_DIR}/summary.json')

# --- Final report ---
print(f"\n{'='*70}")
print(f'DONE! Files in: {OUTPUT_DIR}/')
print(f"{'='*70}")
print(f"\nFiles to copy to MT5:\n")
print(f'  1. {onnx_dir}/buy_model.onnx  -> Common\\Files\\RNN\\buy_model.onnx')
print(f'  2. {onnx_dir}/sell_model.onnx -> Common\\Files\\RNN\\sell_model.onnx')
print(f'  3. {OUTPUT_DIR}/RNN_Scaler.mqh -> MQL5\\Experts\\RNN_gold\\RNN_Scaler.mqh')
print(f"\nBUY:  seq_len={buy_seq_len}  features={buy_nf}")
print(f"SELL: seq_len={sell_seq_len}  features={sell_nf}")
