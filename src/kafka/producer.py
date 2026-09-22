import json
import time
from confluent_kafka import Producer
import pandas as pd
import numpy as np

books_rating = pd.read_csv("data/raw/Books_rating.csv", nrows=5000)


books_rating = books_rating.astype(object).where(pd.notnull(books_rating), None)

producer_config = {
    'bootstrap.servers': 'localhost:29092'
}

producer = Producer(producer_config)

def review_report(err, msg):
    if err:
        print(f'review error: {err}')
    else:
        print(f'review: {msg.value().decode("utf-8")}')


for index, row in books_rating.iterrows():
    review = {
        "Id": row["Id"],
        "Title": row["Title"],
        "Price": row["Price"],
        "User_id": row["User_id"],
        "profileName": row["profileName"],
        "review/helpfulness": row["review/helpfulness"],
        "review/score": row["review/score"],
        "review/time": row["review/time"],
        "review/summary": row["review/summary"],
        "review/text": row["review/text"]
    }

    value = json.dumps(review).encode("utf-8")

    producer.produce(
        topic="reviews-stream",
        value=value,
        callback=review_report
    )

    producer.poll(0)  
    time.sleep(0.02)

producer.flush()  