"""
Generate synthetic AWS-like multivariate time-series and inject anomalies.
Outputs CSV with columns: ts, temperature, pressure, humidity, label, anomaly_type.
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_base(n: int, seed: int = 0) -> pd.DataFrame:
    np.random.seed(seed)
    timestamps = [datetime.utcnow() + timedelta(minutes=i) for i in range(n)]
    t = np.arange(n)

    temperature = (
        25
        + 8 * np.sin(2 * np.pi * t / 1440)
        + 2.5 * np.sin(2 * np.pi * t / 150)
        + np.random.normal(0, 0.55, n)
    )

    pressure = (
        1013
        + 4 * np.sin(2 * np.pi * t / 9600)
        + 0.8 * np.cos(2 * np.pi * t / 300)
        + np.random.normal(0, 0.25, n)
    )

    humidity = (
        60
        + 18 * np.cos(2 * np.pi * t / 1440)
        + 5 * np.sin(2 * np.pi * t / 360)
        + np.random.normal(0, 1.4, n)
    )

    df = pd.DataFrame(
        {
            "ts": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
        }
    )
    return df


def inject_spike(df: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    n = len(df)
    for _ in range(count):
        idx = np.random.randint(0, n)
        sensor = np.random.choice(["temperature", "pressure", "humidity"])
        magnitude = {"temperature": 18, "pressure": 30, "humidity": 35}[sensor]
        direction = np.random.choice([-1, 1])
        df.loc[idx, sensor] += direction * magnitude
        df.at[idx, "label"] = 1
        df.at[idx, "anomaly_type"] = "spike"
    return df


def inject_stuck(df: pd.DataFrame, count: int = 8, length: int = 40) -> pd.DataFrame:
    n = len(df)
    for _ in range(count):
        start = np.random.randint(0, n - length)
        sensor = np.random.choice(["temperature", "pressure", "humidity"])
        base_value = df.loc[start, sensor]
        for j in range(length):
            idx = start + j
            df.loc[idx, sensor] = base_value + np.random.normal(0, 0.01)
            df.at[idx, "label"] = 1
            df.at[idx, "anomaly_type"] = "stuck"
    return df


def inject_drift(df: pd.DataFrame, count: int = 6, length: int = 150) -> pd.DataFrame:
    n = len(df)
    for _ in range(count):
        start = np.random.randint(0, n - length)
        sensor = np.random.choice(["temperature", "pressure", "humidity"])
        slope = np.random.choice([0.04, -0.04, 0.02, -0.02])
        for j in range(length):
            idx = start + j
            df.loc[idx, sensor] += slope * j + np.random.normal(0, 0.25)
            df.at[idx, "label"] = 1
            df.at[idx, "anomaly_type"] = "drift"
    return df


def inject_communication_loss(df: pd.DataFrame, count: int = 4, length: int = 60) -> pd.DataFrame:
    n = len(df)
    for _ in range(count):
        start = np.random.randint(0, n - length)
        stop = start + length
        df.loc[start:stop, ["temperature", "pressure", "humidity"]] = np.nan
        df.loc[start:stop, "label"] = 1
        df.loc[start:stop, "anomaly_type"] = "communication_loss"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AWS data with anomaly injections.")
    parser.add_argument("--out", type=str, default="data.csv", help="Output CSV file path")
    parser.add_argument("--n", type=int, default=20000, help="Number of samples to generate")
    parser.add_argument("--spikes", type=int, default=20, help="Number of spike anomalies")
    parser.add_argument("--stuck", type=int, default=8, help="Number of frozen-value anomalies")
    parser.add_argument("--drift", type=int, default=6, help="Number of drift anomalies")
    parser.add_argument("--comm", type=int, default=4, help="Number of communication loss anomalies")
    parser.add_argument("--stuck_len", type=int, default=40, help="Length of frozen sensor segment")
    parser.add_argument("--drift_len", type=int, default=150, help="Length of drift segment")
    parser.add_argument("--comm_len", type=int, default=60, help="Length of communication loss segment")
    args = parser.parse_args()

    df = generate_base(args.n, seed=42)
    df["label"] = 0
    df["anomaly_type"] = "normal"

    df = inject_spike(df, count=args.spikes)
    df = inject_stuck(df, count=args.stuck, length=args.stuck_len)
    df = inject_drift(df, count=args.drift, length=args.drift_len)
    df = inject_communication_loss(df, count=args.comm, length=args.comm_len)

    # Fill NaNs after injection for safe downstream processing, but label stays anomaly
    df = df.ffill().bfill()
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows and saved to {args.out}")


if __name__ == "__main__":
    main()
