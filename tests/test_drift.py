import pandas as pd
import pytest

from monitoring.drift import DRIFT_WINDOW, check_drift, clean_batches, compute_stats

# A healthy baseline, like the one saved in baseline.json
BASELINE = {"positive_ratio": 0.8, "mean_confidence": 0.86}


def make_batches(n, reviews=10, positives=8, confidence=0.86, start_id=1):
    """Build n identical batches (default: 80% positive, confidence 0.86)."""
    return pd.DataFrame({
        "batch_id": list(range(start_id, start_id + n)),
        "reviews": [reviews] * n,
        "positives": [positives] * n,
        "avg_confidence": [confidence] * n,
    })


# ---------- compute_stats ----------

def test_compute_stats():
    df = pd.DataFrame({
        "batch_id":       [1,    2,    3],
        "reviews":        [2,    10,   8],
        "positives":      [2,    5,    1],
        "avg_confidence": [0.50, 0.90, 0.80],
    })

    stats = compute_stats(df)

    assert stats["positive_ratio"] == pytest.approx(0.4)
    assert stats["mean_confidence"] == pytest.approx(0.82)


def test_compute_stats_empty_dataframe():
    assert compute_stats(pd.DataFrame()) is None


# ---------- clean_batches ----------

def test_clean_batches_drops_batch_zero_and_missing_positives():
    df = pd.DataFrame({
        "batch_id":       [0, 1, 2],
        "reviews":        [3, 5, 5],
        "positives":      [1, None, 4],   # batch 1 is an old log line
        "avg_confidence": [0.6, 0.8, 0.9],
    })

    result = clean_batches(df)

    assert list(result["batch_id"]) == [2]


def test_clean_batches_empty_or_no_positives_column():
    assert clean_batches(pd.DataFrame()).empty

    old_logs = pd.DataFrame({"batch_id": [1], "reviews": [5], "avg_confidence": [0.8]})
    assert clean_batches(old_logs).empty


# ---------- check_drift ----------
# All tests use DRIFT_WINDOW, so they keep working if the window size changes.

def test_check_drift_no_drift():
    result = check_drift(make_batches(DRIFT_WINDOW), BASELINE)

    assert result["drifted"] is False
    assert result["ratio_change"] == pytest.approx(0.0)


def test_check_drift_positive_ratio_shift():
    # Only 2 positives out of 10 -> ratio 0.2 instead of 0.8
    result = check_drift(make_batches(DRIFT_WINDOW, positives=2), BASELINE)

    assert result["drifted"] is True
    assert result["ratio_change"] == pytest.approx(0.6)


def test_check_drift_confidence_drop():
    # Ratio is unchanged, only confidence falls from 0.86 to 0.70
    result = check_drift(make_batches(DRIFT_WINDOW, confidence=0.70), BASELINE)

    assert result["drifted"] is True
    assert result["confidence_drop"] == pytest.approx(0.16)


def test_check_drift_higher_confidence_is_not_drift():
    result = check_drift(make_batches(DRIFT_WINDOW, confidence=0.95), BASELINE)

    assert result["drifted"] is False


def test_check_drift_not_enough_batches():
    # One batch short of a full window
    assert check_drift(make_batches(DRIFT_WINDOW - 1), BASELINE) is None


def test_check_drift_no_baseline():
    assert check_drift(make_batches(DRIFT_WINDOW), None) is None


def test_check_drift_only_looks_at_recent_window():
    # Old batches drifted, but the last DRIFT_WINDOW batches are healthy -> no drift
    old_bad = make_batches(30, positives=2, start_id=1)
    recent_good = make_batches(DRIFT_WINDOW, start_id=31)
    df = pd.concat([old_bad, recent_good], ignore_index=True)

    assert check_drift(df, BASELINE)["drifted"] is False