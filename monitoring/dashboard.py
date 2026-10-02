import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path

st.title("NLP Pipeline — Live Monitoring")

LOG_PATH = Path("logs/spark_consumer.log")

def load_batches():
    records = []

    if not LOG_PATH.exists():
        return pd.DataFrame()

    with open(LOG_PATH, 'r') as f:
        for line in f:

            if '"batch_id"' not in line:
                continue
            
            try:
                json_str = line.split("INFO:__main__:", 1)[-1].strip()
                record = json.loads(json_str)
                if 'reviews' in record:
                    records.append(record)
            except (json.JSONDecodeError, IndexError):
                continue

    return pd.DataFrame(records)


df = load_batches()

if df.empty:
    st.warning("No batch data found yet.")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total batches", len(df))
col2.metric("Total Reviews", int(df["reviews"].sum()))
col3.metric("Avg Confidence", f"{df["avg_confidence"].mean():.2f}")
col4.metric("Avg Throughput", f"{df["throughput"].mean():.2f} rev/sec")


st.subheader("Confidence over time")
st.line_chart(df.set_index("timestamp")["avg_confidence"])

st.subheader("Throughput over time")
st.line_chart(df.set_index("timestamp")["throughput"])

st.subheader("Latency over time")
st.line_chart(df.set_index("timestamp")["latency"])

st.subheader("Recent batches")
st.dataframe(df.tail(20).sort_values("timestamp", ascending=False), use_container_width=True)


# Flag low-confidence batches
low_conf = df[df["avg_confidence"] < 0.6]
if not low_conf.empty:
    st.subheader("Low-confidence batches (<0.6)")
    st.dataframe(low_conf, use_container_width=True)

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()