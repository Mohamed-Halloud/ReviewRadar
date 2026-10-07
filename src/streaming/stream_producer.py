import argparse
import json
import logging
import os
import random
import time

import pandas as pd
from confluent_kafka import Producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/books_reviews_clean/"
TOPIC = "reviews-stream"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

NEGATIVE_LABEL = 0         # label id of the "negative" class (check against your training labels)
DELAY_RANGE = (0.1, 0.5)   # random delay between reviews (about 3 reviews/second)

FIELDS = [
    "Id", "Title", "User_id", "review/score",
    "review/time", "review/summary", "review/text", "label",
]


def clean_value(v):
    """Convert pandas/numpy values (NaN, numpy ints and floats) into JSON-serializable types."""
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def review_report(err, msg):
    """Callback fired once Kafka acknowledges (or fails) delivery of a message."""
    if err:
        logger.error(f"Delivery failed: {err}")
    else:
        print(f'review: {msg.value().decode("utf-8")}')


def build_sample(df, mode, n):
    """Pick the reviews to stream.

    healthy: random sample with the natural class mix (matches baseline.json)
    drift:   only negative reviews, to test the drift alert
    """
    if mode == "drift":
        df = df[df["label"] == NEGATIVE_LABEL]
        logger.warning("DRIFT MODE: streaming only negative reviews")
        if df.empty:
            raise SystemExit(f"No reviews with label == {NEGATIVE_LABEL}: check NEGATIVE_LABEL")
    return df.sample(min(n, len(df)), random_state=42)


def main():
    parser = argparse.ArgumentParser(description="Stream reviews to Kafka")
    parser.add_argument(
        "--mode", choices=["healthy", "drift"], default="healthy",
        help="healthy = natural class mix (default), drift = negative reviews only",
    )
    parser.add_argument("--n", type=int, default=1000, help="number of reviews to stream")
    args = parser.parse_args()

    # Load cleaned reviews dataset
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {df.shape[0]} reviews, columns: {df.columns.tolist()}")

    sample = build_sample(df, args.mode, args.n)
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    start = time.time()
    sent = 0
    for _, row in sample.iterrows():
        # Build a clean dict for this review, handling NaNs/numpy types
        review = {k: clean_value(row[k]) for k in FIELDS}

        # Send the review to the Kafka topic
        producer.produce(
            topic=TOPIC,
            value=json.dumps(review).encode("utf-8"),
            callback=review_report
        )
        sent += 1

        # Trigger delivery callbacks without blocking
        producer.poll(0)
        # Random delay to simulate real-time arrival of reviews
        time.sleep(random.uniform(*DELAY_RANGE))

    # Block until all pending messages are delivered
    producer.flush()

    elapsed = time.time() - start
    logger.info(f"Sent {sent} reviews in {elapsed:.1f}s ({sent / elapsed:.2f} reviews/sec)")


if __name__ == "__main__":
    main()