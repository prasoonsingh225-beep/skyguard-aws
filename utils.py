"""
Shared utilities for the SkyGuard-AWS prototype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


FEATURES = ["temperature", "pressure", "humidity"]


def make_windows(df: pd.DataFrame, window_size: int = 60, features: list[str] | None = None):
    if features is None:
        features = FEATURES
    values = df[features].to_numpy(dtype=np.float32)
    if len(values) < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {len(values)}")

    windows = []
    for i in range(len(values) - window_size + 1):
        windows.append(values[i : i + window_size])
    return np.stack(windows).astype(np.float32)


def fit_scaler(df: pd.DataFrame, features: list[str] | None = None, out_path: str | None = None):
    if features is None:
        features = FEATURES
    scaler = StandardScaler()
    scaler.fit(df[features].to_numpy(dtype=np.float32))
    if out_path is not None:
        joblib.dump(scaler, out_path)
    return scaler


def scale_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = X.shape
    flat = X.reshape(-1, original_shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(original_shape)


def reconstruction_error(X: np.ndarray, X_rec: np.ndarray) -> np.ndarray:
    return np.mean(np.square(X - X_rec), axis=(1, 2))


def classify_root_cause(point: dict, previous: dict | None = None, short_buffer: list[dict] | None = None) -> str:
    if previous is None:
        return "insufficient_context"

    diffs = {
        "temperature": abs(point["temperature"] - previous["temperature"]),
        "pressure": abs(point["pressure"] - previous["pressure"]),
        "humidity": abs(point["humidity"] - previous["humidity"]),
    }

    if diffs["temperature"] > 12 or diffs["pressure"] > 20 or diffs["humidity"] > 30:
        return "spike"

    if short_buffer is not None and len(short_buffer) >= 10:
        arr = np.array(
            [[b["temperature"], b["pressure"], b["humidity"]] for b in short_buffer],
            dtype=np.float32,
        )
        var = np.var(arr, axis=0)
        if np.any(var < np.array([0.0008, 0.0008, 0.0015])):
            return "stuck_or_frozen"

    if (
        abs(point["temperature"] - previous["temperature"]) > 3
        or abs(point["pressure"] - previous["pressure"]) > 4
        or abs(point["humidity"] - previous["humidity"]) > 10
    ):
        return "drift_or_abrupt_change"

    return "sensor_anomaly"


def severity_from_score(score: float, threshold: float) -> str:
    ratio = score / (threshold + 1e-9)
    if ratio < 1.0:
        return "normal"
    if ratio < 1.8:
        return "low"
    if ratio < 3.0:
        return "medium"
    return "high"
