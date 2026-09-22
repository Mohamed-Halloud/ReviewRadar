import json

from confluent_kafka import Consumer

consumer_config = {
    "bootstrap.servers": "localhost:29092",
    "group.id": "review-tracker",
    "auto.offset.reset": "earliest"
}

consumer = Consumer(consumer_config)

consumer.subscribe(["reviews-stream"])

print("consumer is running and subscribed to reviews-stream topic")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"error: {msg.error()}")
            continue

        value = msg.value().decode('utf-8')
        review = json.loads(value)

        print(f"received review: {review['review/score']} stars for '{review['Title']}' by {review['profileName']}")

except KeyboardInterrupt:
    pass

finally:
    consumer.close()