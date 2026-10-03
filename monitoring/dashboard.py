import time

import pandas as pd
import streamlit as st

from drift import LOG_PATH, load_batches, load_baseline, check_drift

st.title("NLP Pipeline — Live Monitoring")

df = load_batches(LOG_PATH)

if df.empty:
    st.warning("No batch data found yet.")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Health check based on the last 5 batches, to avoid false alarms from one noisy batch
    recent = df.tail(5)
    avg_recent_confidence = recent["avg_confidence"].mean()
    avg_recent_latency = recent["latency"].mean()

    CONFIDENCE_THRESHOLD = 0.6
    LATENCY_THRESHOLD = 2.0

    if avg_recent_confidence < CONFIDENCE_THRESHOLD:
        st.error(f"Low confidence alert: avg confidence over last 5 batches is {avg_recent_confidence:.2f}")
    elif avg_recent_latency > LATENCY_THRESHOLD:
        st.error(f"High latency alert: avg latency over last 5 batches is {avg_recent_latency:.2f}s")
    else:
        st.success("Pipeline healthy")

    # Drift check: last 20 batches vs the saved baseline
    baseline = load_baseline()
    drift = check_drift(df, baseline)

    if baseline is None:
        st.info("No baseline yet: run drift.py to create one")
    elif drift is None:
        st.info("Drift check waiting for more batches")
    else:
        if drift["drifted"]:
            st.error(
                f"Drift detected: positive ratio changed by {drift['ratio_change']:.2f}, "
                f"confidence dropped by {drift['confidence_drop']:.2f}"
            )
        else:
            st.success("No drift detected")

        # Baseline vs current, side by side
        d1, d2 = st.columns(2)
        d1.metric(
            "Positive ratio",
            f"{drift['current_ratio']:.2f}",
            delta=f"{drift['current_ratio'] - baseline['positive_ratio']:+.2f} vs baseline",
            delta_color="off",
        )
        d2.metric(
            "Mean confidence",
            f"{drift['current_confidence']:.2f}",
            delta=f"{drift['current_confidence'] - baseline['mean_confidence']:+.2f} vs baseline",
        )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total batches", len(df))
    col2.metric("Total Reviews", int(df["reviews"].sum()))
    col3.metric("Avg Confidence", f"{df['avg_confidence'].mean():.2f}")
    col4.metric("Avg Throughput", f"{df['throughput'].mean():.2f} rev/sec")

    st.subheader("Confidence over time")
    st.line_chart(df.set_index("timestamp")["avg_confidence"])

    st.subheader("Throughput over time")
    st.line_chart(df.set_index("timestamp")["throughput"])

    st.subheader("Latency over time")
    st.line_chart(df.set_index("timestamp")["latency"])

    st.subheader("Recent batches")
    st.dataframe(df.tail(20).sort_values("timestamp", ascending=False), use_container_width=True)

    # Flag low-confidence batches
    low_conf = df[df["avg_confidence"] < CONFIDENCE_THRESHOLD]
    if not low_conf.empty:
        st.subheader("Low-confidence batches (<0.6)")
        st.dataframe(low_conf, use_container_width=True)

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()