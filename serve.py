"""
FastAPI service that accepts single AWS measurements and produces anomaly verdicts.
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

from utils import FEATURES, classify_root_cause, make_windows, reconstruction_error, scale_windows, severity_from_score

app = FastAPI(title="SkyGuard-AWS API")

model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.joblib")
threshold = float(np.load("threshold.npy")[0])
WINDOW = 60
buffer: list[dict] = []


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "window": WINDOW, "threshold": threshold}


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
