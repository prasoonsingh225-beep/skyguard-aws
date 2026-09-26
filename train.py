"""
Train a multivariate LSTM autoencoder for AWS anomaly detection.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import FEATURES, fit_scaler, make_windows, reconstruction_error, scale_windows


def build_model(window_size: int, n_features: int, latent_dim: int = 32):
    inp = layers.Input((window_size, n_features), name="input_window")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    x = layers.RepeatVector(window_size)(x)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SkyGuard-AWS LSTM autoencoder")
    parser.add_argument("--data", type=str, default="data.csv", help="Input CSV file with AWS readings")
    parser.add_argument("--model", type=str, default="model.h5", help="Model output path")
    parser.add_argument("--window", type=int, default=60, help="Sliding window length")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    scaler = fit_scaler(df, FEATURES, out_path="scaler.joblib")

    X = make_windows(df, window_size=args.window, features=FEATURES)
    X_scaled = scale_windows(X, scaler)

    model = build_model(args.window, len(FEATURES))
    model.fit(X_scaled, X_scaled, epochs=args.epochs, batch_size=args.batch, validation_split=0.1, verbose=1)

    model.save(args.model)
    pred = model.predict(X_scaled, verbose=0)
    errors = reconstruction_error(X_scaled, pred)
    threshold = float(np.percentile(errors, 95))
    np.save("threshold.npy", np.array([threshold], dtype=np.float32))

    print(f"Training complete. Threshold = {threshold}")
    print(f"Saved model to {args.model}")
    print("Saved scaler to scaler.joblib")
    print("Saved threshold to threshold.npy")


if __name__ == "__main__":
    main()
