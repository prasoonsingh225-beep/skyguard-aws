"""
FastAPI service that accepts single AWS measurements and produces anomaly verdicts.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

from utils import FEATURES, classify_root_cause, make_windows, reconstruction_error, scale_windows, severity_from_score

app = FastAPI(title="SkyGuard-AWS API")

model = None
scaler = None
threshold = 0.25
WINDOW = 60
buffer: list[dict] = []

# Fallback operational mode for deployment environments that cannot train a full model on boot.
if os.path.exists("model.h5") and os.path.exists("scaler.joblib") and os.path.exists("threshold.npy"):
    try:
        model = tf.keras.models.load_model("model.h5")
        scaler = joblib.load("scaler.joblib")
        threshold = float(np.load("threshold.npy")[0])
    except Exception:
        model = None
        scaler = None
        threshold = 0.25


class Point(BaseModel):
    ts: Optional[str] = None
    temperature: float
    pressure: float
    humidity: float


class PredictionResult(BaseModel):
    score: float
    threshold: float
    is_anomaly: bool
    severity: str
    root_cause: str
    confidence: float


def fallback_heuristic(point: dict, prev_point: Optional[dict], short_buffer: list[dict]) -> tuple[float, bool, str, float]:
    if prev_point is None:
        return 0.0, False, "insufficient_context", 0.10

    temp_delta = abs(point["temperature"] - prev_point["temperature"])
    pressure_delta = abs(point["pressure"] - prev_point["pressure"])
    humidity_delta = abs(point["humidity"] - prev_point["humidity"])

    score = float(max(temp_delta / 10.0, pressure_delta / 20.0, humidity_delta / 25.0))
    if score > 1.4:
        is_anomaly = True
        root_cause = "spike"
    elif temp_delta > 8 or pressure_delta > 15 or humidity_delta > 20:
        is_anomaly = True
        root_cause = "drift_or_abrupt_change"
    else:
        is_anomaly = False
        root_cause = "normal"

    if is_anomaly:
        confidence = min(0.99, 0.4 + score / 2.0)
    else:
        confidence = 0.05
    return score, is_anomaly, root_cause, confidence


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "window": WINDOW, "threshold": threshold, "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResult)
def predict(point: Point) -> PredictionResult:
    global buffer

    current = {
        "temperature": float(point.temperature),
        "pressure": float(point.pressure),
        "humidity": float(point.humidity),
    }

    buffer.append(current)
    if len(buffer) > WINDOW:
        buffer = buffer[-WINDOW:]

    if len(buffer) < WINDOW:
        return PredictionResult(
            score=0.0,
            threshold=threshold,
            is_anomaly=False,
            severity="normal",
            root_cause="insufficient_context",
            confidence=0.10,
        )

    if model is not None and scaler is not None:
        recent = pd.DataFrame(buffer, columns=FEATURES)
        X = make_windows(recent, window_size=WINDOW, features=FEATURES)
        X_scaled = scale_windows(X, scaler)
        rec = model.predict(X_scaled[-1:], verbose=0)
        score = float(reconstruction_error(X_scaled[-1:], rec)[0])
        is_anomaly = bool(score > threshold)

        prev_point = buffer[-2] if len(buffer) >= 2 else None
        root_cause = classify_root_cause(current, prev_point, buffer)
        sev = severity_from_score(score, threshold)
        confidence = float(np.clip((score - threshold) / (threshold * 2.0 + 1e-9), 0.0, 1.0))

        return PredictionResult(
            score=score,
            threshold=threshold,
            is_anomaly=is_anomaly,
            severity=sev,
            root_cause=root_cause,
            confidence=confidence,
        )

    # Fallback: no model files present on deployment; use heuristic anomaly checks.
    prev_point = buffer[-2] if len(buffer) >= 2 else None
    score, is_anomaly, root_cause, conf = fallback_heuristic(current, prev_point, buffer)
    sev = severity_from_score(score, threshold if threshold > 0 else 1.0)

    return PredictionResult(
        score=float(score),
        threshold=float(threshold if threshold > 0 else 1.0),
        is_anomaly=bool(is_anomaly),
        severity=sev,
        root_cause=root_cause,
        confidence=float(conf),
    )
