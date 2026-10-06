import json
import logging
import random
import time

import pandas as pd
from confluent_kafka import Producer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Load cleaned reviews dataset
df = pd.read_parquet("data/processed/books_reviews_clean/")
print(df.shape)

# Take a random sample of 1000 reviews to stream
sample = df.sample(1000, random_state=42)

producer_config = {
     'bootstrap.servers': 'localhost:29092'
}

producer = Producer(producer_config)

# Callback fired once Kafka acknowledges (or fails) delivery of a message
def review_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")
    else:
        print(f'review: {msg.value().decode("utf-8")}')


# Convert pandas/NaN values into JSON-serializable Python types
def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return v.item() if hasattr(v, "item") else v
    return v

print(df.columns.tolist())

start = time.time()
i = 0
for index, row in sample.iterrows():
    # Build a clean dict for this review, handling NaNs/numpy types
    review = {k: clean_value(v) for k, v in {
        "Id": row["Id"],
        "Title": row["Title"],
        "User_id": row["User_id"],
        "review/score": row["review/score"],
        "review/time": row["review/time"],
        "review/summary": row["review/summary"],
        "review/text": row["review/text"],
        "label": row["label"]
    }.items()}

    value = json.dumps(review).encode('utf-8')

    # Send the review to the Kafka topic
    producer.produce(
        topic="reviews-stream",
        value=value,
        callback=review_report
    )

    i = i + 1

    # Trigger delivery callbacks without blocking
    producer.poll(0)
    # Random delay to simulate real-time arrival of reviews
    time.sleep(random.uniform(0.1, 0.5))

# Block until all pending messages are delivered
producer.flush()

elapsed = time.time() - start
logger.info(f"Sent {i} reviews in {elapsed:.1f}s ({i/elapsed:.2f} reviews/sec)")