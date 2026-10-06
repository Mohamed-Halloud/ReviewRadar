import json
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PATH = Path("logs/spark_consumer.log")
BASELINE_PATH = Path("baseline.json")
BASELINE_BATCHES = 18  # how many healthy batches define "normal"
DRIFT_WINDOW = 10      # recent batches compared to the baseline
RATIO_THRESHOLD = 0.10       # max allowed change in positive ratio
CONFIDENCE_THRESHOLD = 0.05  # max allowed drop in mean confidence


def load_batches(path=LOG_PATH):
    """Read the consumer log and return one row per batch."""
    records = []

    if not path.exists():
        return pd.DataFrame()

    with open(path, "r") as f:
        for line in f:
            # Skip any log line that isn't one of our batch records
            if '"batch_id"' not in line:
                continue

            try:
                # Strip the logging prefix and parse the remaining JSON
                json_str = line[line.index("{"):].strip()
                record = json.loads(json_str)
                if "reviews" in record:
                    records.append(record)
            except (json.JSONDecodeError, IndexError):
                # Skip malformed/partial lines instead of crashing
                continue

    return pd.DataFrame(records)


def clean_batches(df):
    if df.empty or "positives" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["positives"])  # old log lines have no 'positives'
    return df[df["batch_id"] != 0]       


def compute_stats(df):
    """Positive ratio and mean confidence, both weighted by batch size."""
    if df.empty:
        return None
    
    positive_ratio = df["positives"].sum() / df["reviews"].sum()
    # Bigger batches count more
    mean_confidence = np.average(df["avg_confidence"], weights=df["reviews"])

    return {
        "positive_ratio": round(float(positive_ratio), 4),
        "mean_confidence": round(float(mean_confidence), 4),
    }


def save_baseline(n=BASELINE_BATCHES, log_path=LOG_PATH, out_path=BASELINE_PATH):
    """Compute stats on the first n clean batches and save them to baseline.json."""
    df = clean_batches(load_batches(log_path))

    if len(df) < n:
        raise ValueError(f"Need at least {n} clean batches, found {len(df)}")

    baseline_df = df.head(n)
    stats = compute_stats(baseline_df)
    stats["n_batches"] = len(baseline_df)
    stats["n_reviews"] = int(baseline_df["reviews"].sum())

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def load_baseline(path=BASELINE_PATH):
    """Read baseline.json, or return None if it doesn't exist yet."""
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def check_drift(df, baseline, window=DRIFT_WINDOW):

    if baseline is None:
        return None

    recent = clean_batches(df).tail(window)
    if len(recent) < window:
        return None  # too few batches: a small sample is too noisy

    current = compute_stats(recent)

    # Positive ratio: any big move counts. Confidence: only a drop is a problem.
    ratio_change = abs(current["positive_ratio"] - baseline["positive_ratio"])
    confidence_drop = baseline["mean_confidence"] - current["mean_confidence"]

    return {
        "current_ratio": current["positive_ratio"],
        "current_confidence": current["mean_confidence"],
        "ratio_change": round(ratio_change, 4),
        "confidence_drop": round(confidence_drop, 4),
        "drifted": bool(
            ratio_change > RATIO_THRESHOLD or confidence_drop > CONFIDENCE_THRESHOLD
        ),
    }


if __name__ == "__main__":
    # Run from the project root: python <path>/drift.py
    print(save_baseline())