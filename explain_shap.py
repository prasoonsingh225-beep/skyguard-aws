"""
SHAP-based explainability example for the LSTM autoencoder output.
This is a simple demonstration and may be computationally heavy.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from utils import FEATURES, make_windows, scale_windows


model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.joblib")


def explain_window(df_window: pd.DataFrame):
    X = make_windows(df_window, window_size=len(df_window), features=FEATURES)
    X_scaled = scale_windows(X, scaler)
    x = X_scaled[0]

    def model_wrapper(values):
        values = values.reshape((-1, x.shape[0], x.shape[1]))
        pred = model.predict(values, verbose=0)
        errors = np.mean((values - pred) ** 2, axis=(1, 2))
        return errors

    base = np.zeros((1, x.shape[0] * x.shape[1]))
    explainer = shap.KernelExplainer(model_wrapper, base)
    shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=100)
    return shap_values


if __name__ == "__main__":
    sample = pd.read_csv("data.csv").head(60)
    values = explain_window(sample)
    print("SHAP explanation computed successfully.")
    print(type(values))
    if isinstance(values, list):
        print("Number of outputs:", len(values))
