"""Streamlit dashboard for the deployed AERO Guard API."""

import os
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AERO Guard", layout="wide")

# Render injects API_URL. Local users can override it in the sidebar.
default_api = os.getenv("API_URL", "http://127.0.0.1:8000")
if default_api.endswith("/predict"):
    default_api = default_api[:-8]

st.sidebar.title("AERO Guard")
api_base = st.sidebar.text_input("API base URL", value=default_api).rstrip("/")
predict_url = f"{api_base}/predict"

if "records" not in st.session_state:
    st.session_state.records = []

st.title("AERO Guard — AWS Anomaly Detection")
st.caption("SIH 26073 | Real-time temperature, pressure and humidity quality control")

try:
    health = requests.get(f"{api_base}/health", timeout=5)
    if health.ok:
        st.success("API connected")
    else:
        st.warning("API responded, but is not healthy")
except requests.RequestException:
    st.error("API is not reachable. Check the API URL in the sidebar.")

uploaded_file = st.file_uploader("Upload AWS CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required = {"temperature", "pressure", "humidity"}
    if not required.issubset(df.columns):
        st.error("CSV must contain temperature, pressure, and humidity columns.")
    else:
        if st.button("Analyze first 200 readings"):
            progress = st.progress(0)
            for index, row in df.head(200).iterrows():
                payload = {
                    "ts": str(row.get("ts", "")),
                    "temperature": float(row["temperature"]),
                    "pressure": float(row["pressure"]),
                    "humidity": float(row["humidity"]),
                }
                try:
                    response = requests.post(predict_url, json=payload, timeout=20)
                    response.raise_for_status()
                    st.session_state.records.append({**payload, **response.json()})
                except requests.RequestException as exc:
                    st.error(f"Prediction failed: {exc}")
                    break
                progress.progress((index + 1) / min(len(df), 200))
                time.sleep(0.02)

if st.session_state.records:
    results = pd.DataFrame(st.session_state.records)
    left, middle, right = st.columns(3)
    left.metric("Readings analyzed", len(results))
    middle.metric("Anomalies", int(results["is_anomaly"].sum()))
    right.metric("Latest severity", str(results.iloc[-1]["severity"]))

    st.subheader("Sensor readings")
    st.line_chart(results.set_index("ts")[["temperature", "pressure", "humidity"]].tail(100))
    st.subheader("Recent alerts")
    st.dataframe(results.tail(30), use_container_width=True)
else:
    st.info("Upload data.csv, click Analyze, and the anomaly results will appear here.")
