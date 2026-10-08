"""Canonical local preprocessing for the 20-bar, 23-feature ONNX contract."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_BARS = 20
RAW_FEATURES = 18
MODEL_FEATURES = 23
EPS = 1e-6

RAW_NAMES = (
    "open", "high", "low", "close", "volume", "ema8", "ema21", "rsi",
    "stoch_k", "stoch_d", "macd_main", "macd_signal", "atr", "sin_hour",
    "cos_hour", "day_of_week", "direction", "body_range",
)
MODEL_NAMES = (
    "open_rel_atr", "high_rel_atr", "low_rel_atr", "close_rel_atr",
    "body_atr", "range_atr", "upper_wick_atr", "lower_wick_atr",
    "close_position", "log_volume_relative", "ema8_rel_atr", "ema21_rel_atr",
    "ema_spread_atr", "macd_main_atr", "macd_hist_atr", "atr_relative",
    "rsi_01", "stoch_k_01", "stoch_d_01", "sin_hour", "cos_hour",
    "sin_day_of_week", "cos_day_of_week",
)
# Row-local z-score is applied only to unbounded channels.
NORMALIZE_MASK = tuple(range(8)) + (9, 10, 11, 12, 13, 14, 15)


def expected_columns() -> list[str]:
    columns = ["datetime", "signal"]
    columns.extend(f"lag{lag}_{name}" for lag in range(RAW_BARS) for name in RAW_NAMES)
    columns.extend(("result", "label"))
    return columns


def schema_fingerprint() -> str:
    return hashlib.sha256("\n".join(expected_columns()).encode("utf-8")).hexdigest()


def validate_frame(frame: pd.DataFrame, source: str = "CSV") -> None:
    required = {"datetime", "signal", "result", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    actual_lags = [c for c in frame.columns if c.startswith("lag")]
    expected_lags = expected_columns()[2:-2]
    if actual_lags != expected_lags:
        if len(actual_lags) != len(expected_lags):
            raise ValueError(f"{source}: expected {len(expected_lags)} lag columns, got {len(actual_lags)}")
        raise ValueError(f"{source}: lag column order/schema differs from canonical contract")
    if len(actual_lags) != RAW_BARS * RAW_FEATURES:
        raise ValueError(f"{source}: expected {RAW_BARS}x{RAW_FEATURES} raw fields")


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_frame(frame, str(path))
    frame["datetime"] = pd.to_datetime(frame["datetime"], format="%Y.%m.%d %H:%M", errors="raise")
    frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    return frame


def raw_tensor(frame: pd.DataFrame) -> np.ndarray:
    values = frame[[f"lag{lag}_{name}" for lag in range(RAW_BARS) for name in RAW_NAMES]].to_numpy(dtype=np.float64)
    raw = values.reshape(len(frame), RAW_BARS, RAW_FEATURES)
    if not np.isfinite(raw).all():
        raise ValueError("raw feature matrix contains NaN or Inf")
    return raw


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def derive_features(raw: np.ndarray) -> np.ndarray:
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
        _clip01((close - low) / candle_range),
        np.log1p(np.maximum(volume, 0.0)),
        (ema8 - close) / atr_scale,
        (ema21 - close) / atr_scale,
        (ema8 - ema21) / atr_scale,
        macd_main / atr_scale,
        (macd_main - macd_signal) / atr_scale,
        atr / lag0_atr,
        _clip01(rsi / 100.0),
        _clip01(stoch_k / 100.0),
        _clip01(stoch_d / 100.0),
        raw[:, :, 13],
        raw[:, :, 14],
        np.sin(2.0 * math.pi * raw[:, :, 15] / 7.0),
        np.cos(2.0 * math.pi * raw[:, :, 15] / 7.0),
    ), axis=-1)
    features[:, :, 9] -= np.median(features[:, :, 9], axis=1, keepdims=True)
    if features.shape != (len(raw), RAW_BARS, MODEL_FEATURES):
        raise AssertionError(f"unexpected derived shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("derived feature matrix contains NaN or Inf")
    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    normalized = features.astype(np.float64, copy=True)
    selected = np.asarray(NORMALIZE_MASK, dtype=np.int64)
    mean = normalized[:, :, selected].mean(axis=1, keepdims=True)
    std = normalized[:, :, selected].std(axis=1, ddof=0, keepdims=True)
    normalized[:, :, selected] = (normalized[:, :, selected] - mean) / np.maximum(std, EPS)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized feature matrix contains NaN or Inf")
    return normalized.astype(np.float32)


def transform_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = raw_tensor(frame)
    model = normalize_features(derive_features(raw))
    labels = frame["label"].to_numpy(dtype=np.int64)
    results = frame["result"].to_numpy(dtype=np.float64)
    timestamps = frame["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    return model, labels, results, timestamps


def quality_report(model: np.ndarray, labels: np.ndarray, source: str) -> dict:
    flat = model.reshape(-1, model.shape[-1])
    finite = np.isfinite(flat)
    return {
        "source": source,
        "rows": int(len(model)),
        "shape": list(model.shape),
        "model_features": list(MODEL_NAMES),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "zero_std_channels": [MODEL_NAMES[i] for i, value in enumerate(flat.std(axis=0)) if value < EPS],
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "global_min": float(np.min(flat[finite])) if finite.any() else None,
        "global_max": float(np.max(flat[finite])) if finite.any() else None,
        "schema_fingerprint": schema_fingerprint(),
    }


def process_file(path: Path, output_dir: Path) -> dict:
    frame = load_csv(path)
    model, labels, results, timestamps = transform_frame(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{path.stem}_normalized.npz"
    np.savez_compressed(target, X=model, y=labels, result=results, datetime=timestamps)
    report = quality_report(model, labels, path.name)
    report["output"] = target.name
    return report


def write_contract(output_dir: Path) -> None:
    contract = {
        "version": 2,
        "raw_bars": RAW_BARS,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "raw_names": list(RAW_NAMES),
        "model_names": list(MODEL_NAMES),
        "normalize_mask": list(NORMALIZE_MASK),
        "normalization": "row_local_zscore_ddof0_selected_channels",
        "eps": EPS,
        "schema_fingerprint": schema_fingerprint(),
        "onnx_input_shape": [1, RAW_BARS, MODEL_FEATURES],
    }
    (output_dir / "model_features.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw MQL5 CSV files for Colab training")
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/"))
    parser.add_argument("--pattern", default="data_*.csv")
    args = parser.parse_args()
    paths = sorted(Path(p) for p in glob.glob(str(args.input_dir / args.pattern)))
    if not paths:
        raise SystemExit(f"No CSV files found in {args.input_dir} matching {args.pattern}")
    reports = [process_file(path, args.output_dir) for path in paths]
    write_contract(args.output_dir)
    (args.output_dir / "preparation_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for report in reports:
        print(f"{report['source']}: {report['rows']} rows -> {report['output']} | shape={report['shape']}")


if __name__ == "__main__":
    main()
