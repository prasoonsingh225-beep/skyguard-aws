"""
Send example AWS readings to the FastAPI service.
"""

import time

import pandas as pd
import requests


def send_sample_rows(csv_path: str = "data.csv", url: str = "http://127.0.0.1:8000/predict", step: int = 50):
    df = pd.read_csv(csv_path)
    for _, row in df.iloc[::step].iterrows():
        payload = {
            "ts": str(row.get("ts", "")),
            "temperature": float(row["temperature"]),
            "pressure": float(row["pressure"]),
            "humidity": float(row["humidity"]),
        }
        response = requests.post(url, json=payload, timeout=5)
        print(payload)
        print(response.json())
        time.sleep(0.05)


if __name__ == "__main__":
    send_sample_rows()
