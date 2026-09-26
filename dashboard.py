"""
Simple Streamlit dashboard for visualizing SkyGuard-AWS anomaly predictions.
"""

import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SkyGuard-AWS Dashboard", layout="wide")

SERVER_URL = st.text_input("API endpoint", value="http://127.0.0.1:8000/predict")

if "records" not in st.session_state:
    st.session_state.records = []

st.title("SkyGuard-AWS Dashboard")

st.write("This dashboard sends station readings to the FastAPI anomaly service and displays the result.")

uploaded_file = st.file_uploader("Upload CSV of AWS measurements", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if {"temperature", "pressure", "humidity"}.issubset(df.columns):
        for _, row in df.head(200).iterrows():
            payload = {
                "ts": str(row.get("ts", "")),
                "temperature": float(row["temperature"]),
                "pressure": float(row["pressure"]),
                "humidity": float(row["humidity"]),
            }
            try:
                resp = requests.post(SERVER_URL, json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()
                st.session_state.records.append({**payload, **result})
            except Exception as exc:
                st.warning(f"Request failed: {exc}")
                break
            time.sleep(0.15)

if st.session_state.records:
    table = pd.DataFrame(st.session_state.records)
    st.subheader("Recent predictions")
    st.dataframe(table.tail(20), use_container_width=True)

    st.subheader("Anomaly summary")
    anomaly_summary = table["is_anomaly"].value_counts().to_dict()
    st.write(anomaly_summary)
else:
    st.info("Upload a CSV file or use the example client to start sending readings.")
