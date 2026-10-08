"""
Test ONNX models on raw CSV data with full preprocessing pipeline.
Copied from feature_pipeline.py for standalone use.
"""
import os
import sys
import glob
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort


# === Constants (from feature_pipeline.py) ===
RAW_BARS = 20
RAW_FEATURES = 18
MODEL_FEATURES = 23
EPS = 1e-6

RAW_NAMES = (
    "open", "high", "low", "close", "volume", "ema8", "ema21", "rsi",
    "stoch_k", "stoch_d", "macd_main", "macd_signal", "atr", "sin_hour",
    "cos_hour", "day_of_week", "direction", "body_range",
)

NORMALIZE_MASK = tuple(range(8)) + (9, 10, 11, 12, 13, 14, 15)


def clip01(values):
    return np.clip(values, 0.0, 1.0)


def raw_tensor(df):
    """Extract lag columns and reshape to [N, RAW_BARS, RAW_FEATURES]."""
    cols = [f"lag{lag}_{name}" for lag in range(RAW_BARS) for name in RAW_NAMES]
    values = df[cols].to_numpy(dtype=np.float64)
    raw = values.reshape(len(df), RAW_BARS, RAW_FEATURES)
    if not np.isfinite(raw).all():
        raise ValueError("raw feature matrix contains NaN or Inf")
    return raw


def derive_features(raw):
    """18 raw features -> 23 model features (ATR-normalized)."""
    if raw.ndim != 3 or raw.shape[1:] != (RAW_BARS, RAW_FEATURES):
        raise ValueError(f"expected raw shape [N,{RAW_BARS},{RAW_FEATURES}], got {raw.shape}")

    open_, high, low, close, volume = (raw[:, :, i] for i in range(5))
    ema8, ema21, rsi = (raw[:, :, i] for i in (5, 6, 7))
    stoch_k, stoch_d = (raw[:, :, i] for i in (8, 9))
    macd_main, macd_signal, atr = (raw[:, :, i] for i in (10, 11, 12))

    lag0_close = close[:, :1]
    lag0_atr = np.maximum(atr[:, :1], EPS)
    atr_scale = np.maximum(atr, EPS)
    candle_range = np.maximum(high - low, EPS)

    features = np.stack((
        (open_ - lag0_close) / lag0_atr,
        (high - lag0_close) / lag0_atr,
        (low - lag0_close) / lag0_atr,
        (close - lag0_close) / lag0_atr,
        (close - open_) / atr_scale,
        (high - low) / atr_scale,
        (high - np.maximum(open_, close)) / atr_scale,
        (np.minimum(open_, close) - low) / atr_scale,
        clip01((close - low) / candle_range),
        np.log1p(np.maximum(volume, 0.0)),
        (ema8 - close) / atr_scale,
        (ema21 - close) / atr_scale,
        (ema8 - ema21) / atr_scale,
        macd_main / atr_scale,
        (macd_main - macd_signal) / atr_scale,
        atr / lag0_atr,
        clip01(rsi / 100.0),
        clip01(stoch_k / 100.0),
        clip01(stoch_d / 100.0),
        raw[:, :, 13],
        raw[:, :, 14],
        np.sin(2.0 * math.pi * raw[:, :, 15] / 7.0),
        np.cos(2.0 * math.pi * raw[:, :, 15] / 7.0),
    ), axis=-1)

    # Median-center log_volume_relative (channel 9)
    features[:, :, 9] -= np.median(features[:, :, 9], axis=1, keepdims=True)

    if features.shape != (len(raw), RAW_BARS, MODEL_FEATURES):
        raise AssertionError(f"unexpected derived shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("derived feature matrix contains NaN or Inf")

    return features


def normalize_features(features):
    """Row-local z-score on selected channels (ddof=0)."""
    normalized = features.astype(np.float64, copy=True)
    selected = np.asarray(NORMALIZE_MASK, dtype=np.int64)
    mean = normalized[:, :, selected].mean(axis=1, keepdims=True)
    std = normalized[:, :, selected].std(axis=1, ddof=0, keepdims=True)
    normalized[:, :, selected] = (normalized[:, :, selected] - mean) / np.maximum(std, EPS)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized feature matrix contains NaN or Inf")
    return normalized.astype(np.float32)


def preprocess_csv(csv_path):
    """Full pipeline: CSV -> normalized tensor [N, 20, 23]."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M", errors="raise")
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    raw = raw_tensor(df)
    derived = derive_features(raw)
    normalized = normalize_features(derived)

    labels = df["label"].to_numpy(dtype=np.int64)
    results = df["result"].to_numpy(dtype=np.float64)
    datetimes = df["datetime"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()

    return normalized, labels, results, datetimes


def load_onnx(model_path):
    """Load ONNX model."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def predict(sess, input_name, X_seq, conf_threshold):
    """Single sample prediction through ONNX model.

    X_seq: already sliced to seq_len bars [seq_len, 23]
    """
    input_tensor = X_seq[np.newaxis, ...].astype(np.float32)
    output = sess.run(None, {input_name: input_tensor})[0]

    # Softmax
    probs = np.exp(output[0]) / np.sum(np.exp(output[0]))
    conf = probs[1]  # confidence for class 1 (profit)

    if conf < conf_threshold:
        pred = 0
    else:
        pred = int(np.argmax(probs))

    return pred, float(conf), float(probs[0]), float(probs[1])


def process_direction(direction, csv_pattern, onnx_path, seq_len, conf_threshold, top_n=5):
    """Process one direction: load CSV, preprocess, run ONNX, print results."""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        print(f"  ERROR: No CSV files found matching {csv_pattern}")
        return None

    csv_path = csv_files[-1]  # take latest
    print(f"\n{'='*70}")
    print(f"=== {direction} Model ===")
    print(f"  CSV: {os.path.basename(csv_path)}")
    print(f"  ONNX: {os.path.basename(onnx_path)}")
    print(f"  seq_len={seq_len}, conf_threshold={conf_threshold}")
    print(f"{'='*70}")

    # Load and preprocess
    X, y, results, datetimes = preprocess_csv(csv_path)
    n_total = len(X)
    print(f"  Preprocessed: {n_total} rows, shape={X.shape}")

    # Load ONNX
    sess, input_name = load_onnx(onnx_path)

    # Run predictions
    correct = 0
    tp = fp = tn = fn = 0  # confusion matrix
    pred_0_count = pred_0_correct = 0
    pred_1_count = pred_1_correct = 0

    print(f"\n--- First {top_n} rows ---")
    for i in range(min(top_n, n_total)):
        X_slice = X[i, :seq_len, :]  # [seq_len, 23]
        pred, conf, prob0, prob1 = predict(sess, input_name, X_slice, conf_threshold)
        label = int(y[i])
        result = float(results[i])
        is_correct = (pred == label)
        if is_correct:
            correct += 1
        # Confusion matrix
        if pred == 0 and label == 0: tn += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        elif pred == 1 and label == 1: tp += 1
        # Track pred=0 and pred=1 correctness
        if pred == 0:
            pred_0_count += 1
            if label == 0: pred_0_correct += 1
        else:
            pred_1_count += 1
            if label == 1: pred_1_correct += 1

        symbol = "OK" if is_correct else "FAIL"
        dt = datetimes[i]
        print(f"Row {i:4d}: dt={dt} | pred={pred} | conf={conf:.4f} | "
              f"p0={prob0:.4f} p1={prob1:.4f} | label={label} | result={result:7.2f} {symbol}")

    # Run ALL predictions for statistics
    print(f"\n--- Running all {n_total} samples for statistics ---")
    all_preds = []
    all_confs = []
    for i in range(n_total):
        X_slice = X[i, :seq_len, :]  # [seq_len, 23]
        pred, conf, _, _ = predict(sess, input_name, X_slice, conf_threshold)
        all_preds.append(pred)
        all_confs.append(conf)

    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)

    # Recalculate full stats
    correct = int(np.sum(all_preds == y))
    tn = fp = fn = tp = 0
    pred_0_count = pred_0_correct = 0
    pred_1_count = pred_1_correct = 0

    for i in range(n_total):
        pred, label = int(all_preds[i]), int(y[i])
        if pred == 0 and label == 0: tn += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        elif pred == 1 and label == 1: tp += 1
        if pred == 0:
            pred_0_count += 1
            if label == 0: pred_0_correct += 1
        else:
            pred_1_count += 1
            if label == 1: pred_1_correct += 1

    accuracy = correct / n_total if n_total > 0 else 0

    # Print summary
    class_0_count = int(np.sum(y == 0))
    class_1_count = int(np.sum(y == 1))

    print(f"\n--- {direction} Statistics ---")
    print(f"Total samples: {n_total}")
    print(f"Class 0 (label): {class_0_count} ({class_0_count/n_total*100:.1f}%)")
    print(f"Class 1 (label): {class_1_count} ({class_1_count/n_total*100:.1f}%)")
    print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{n_total})")
    print(f"\nConfusion Matrix:")
    print(f"              pred=0   pred=1")
    print(f"label=0:       {tn:6d}      {fp:6d}")
    print(f"label=1:       {fn:6d}      {tp:6d}")

    wr_0 = (pred_0_correct / pred_0_count * 100) if pred_0_count > 0 else 0
    wr_1 = (pred_1_correct / pred_1_count * 100) if pred_1_count > 0 else 0
    print(f"\nWinrate (pred=0): {wr_0:.1f}%  ({pred_0_correct}/{pred_0_count} — correctly predicted 0)")
    print(f"Winrate (pred=1): {wr_1:.1f}%  ({pred_1_correct}/{pred_1_count} — correctly predicted 1)")
    print(f"False negatives: {fn} (pred=0, label=1)")
    print(f"False positives: {fp} (pred=1, label=0)")

    return {
        "direction": direction,
        "total": n_total,
        "accuracy": accuracy,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "pred_0_count": pred_0_count, "pred_0_correct": pred_0_correct,
        "pred_1_count": pred_1_count, "pred_1_correct": pred_1_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Test ONNX models on raw CSV data")
    parser.add_argument("--csv-dir", type=str, default="data", help="Directory with raw CSV files")
    parser.add_argument("--onnx-dir", type=str, default="output", help="Directory with ONNX models")
    parser.add_argument("--top-n", type=int, default=50, help="Number of first rows to show")
    args = parser.parse_args()

    # BUY
    buy_csv = os.path.join(args.csv_dir, "data_BUY_*.csv")
    buy_onnx = os.path.join(args.onnx_dir, "buy_model.onnx")
    stats_buy = process_direction("BUY", buy_csv, buy_onnx, seq_len=9, conf_threshold=0.8, top_n=args.top_n)

    # SELL
    sell_csv = os.path.join(args.csv_dir, "data_SELL_*.csv")
    sell_onnx = os.path.join(args.onnx_dir, "sell_model.onnx")
    stats_sell = process_direction("SELL", sell_csv, sell_onnx, seq_len=11, conf_threshold=0.5, top_n=args.top_n)

    # Combined summary
    if stats_buy and stats_sell:
        print(f"\n{'='*70}")
        print(f"=== COMBINED SUMMARY ===")
        print(f"{'='*70}")
        for s in [stats_buy, stats_sell]:
            print(f"\n{s['direction']}: Accuracy={s['accuracy']*100:.1f}% | "
                  f"TN={s['tn']} FP={s['fp']} FN={s['fn']} TP={s['tp']}")


if __name__ == "__main__":
    main()
