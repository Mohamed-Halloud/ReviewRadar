import json
import random

from confluent_kafka import Producer
import time
import pandas as pd


df = pd.read_parquet("data/processed/books_reviews_clean/")
print(df.shape)

sample = df.sample(1000, random_state=42)

producer_config = {
     'bootstrap.servers': 'localhost:29092'
}

producer = Producer(producer_config)

def review_report(err, msg):
    if err:
        print(f'review error: {err}')
    else:
        print(f'review: {msg.value().decode("utf-8")}')


def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return v.item() if hasattr(v, "item") else v
    return v

print(df.columns.tolist())

for index, row in sample.iterrows():
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

    producer.produce(
        topic="reviews-stream",
        value=value,
        callback=review_report
    )

    producer.poll(0)
    time.sleep(random.uniform(0.1, 0.3))

producer.flush()
