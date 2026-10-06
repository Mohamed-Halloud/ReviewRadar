import pandas as pd
import pytest

from monitoring.drift import compute_stats


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