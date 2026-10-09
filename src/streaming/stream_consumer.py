import json
import logging
import os
from datetime import UTC, datetime
from time import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, LongType, StringType, StructType

from src.model.inference import model, predict_batch, tokenizer

# Label id the model uses for "positive" 
POSITIVE_LABEL = 2


MAX_BATCH_SIZE = 8       # max reviews per micro-batch
TRIGGER_SECONDS = 2      # a new micro-batch every 2 seconds

# Kafka address
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

# Logging to file
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename="logs/spark_consumer.log"
)
logger = logging.getLogger(__name__)

# Local Spark session with the Kafka connector
builder = (
    SparkSession.builder
    .appName("KafkaIntegration")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.driver.memory", "2g")  # prevents JVM crashes
)
if os.name == "nt":
    # Windows only: temp dir for Spark
    builder = builder.config("spark.local.dir", "C:/spark-tmp")
spark = builder.getOrCreate()

# Read the raw stream from Kafka
df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", "reviews-stream")
    .option("maxOffsetsPerTrigger", MAX_BATCH_SIZE)  # cap reviews per micro-batch
    .load()
)

# Schema of the JSON review sent by the producer
schema = (
    StructType()
    .add("Id", StringType())
    .add("Title", StringType())
    .add("User_id", StringType())
    .add("review/score", DoubleType())
    .add("review/time", LongType())
    .add("review/summary", StringType())
    .add("review/text", StringType())
    .add("label", LongType())
)

# Kafka value (bytes) -> string -> JSON -> columns
parsed = (
    df.selectExpr("CAST(value AS STRING) as json_str")
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)


def log_batch_stats(batch_id, num_reviews, latency, avg_confidence, positives):
    """Write one JSON line of stats per batch"""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "batch_id": batch_id,
        "reviews": num_reviews,
        "latency": round(latency, 3),
        "throughput": round(num_reviews / latency, 2) if latency > 0 else 0,
        "avg_confidence": round(float(avg_confidence), 3),
        "positives": int(positives),
    }
    logger.info(json.dumps(record))


def count_positives(predictions):
    """Count how many predictions are the positive class."""
    return sum(1 for p in predictions if p == POSITIVE_LABEL)


def process_batch(batch_df, batch_id):
    """Called by Spark for every micro-batch: run inference and log stats."""
    pandas_df = batch_df.toPandas()

    # Nothing to do for empty batches
    if pandas_df.empty:
        logger.info(json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "batch_id": batch_id,
            "status": "empty",
        }))
        return

    # Time only the model inference
    start = time()
    try:
        predictions, confidences = predict_batch(
            pandas_df["review/text"].tolist(), tokenizer, model
        )
    except Exception as e:  # noqa: BLE001
        logger.error(json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "batch_id": batch_id,
            "error": str(e),
        }))
        return
    latency = time() - start

    pandas_df["prediction"] = predictions
    pandas_df["confidence"] = confidences

    log_batch_stats(
        batch_id,
        len(pandas_df),
        latency,
        pandas_df["confidence"].mean(),
        count_positives(pandas_df["prediction"]),
    )

    # Debug output in the console
    print(pandas_df[["Id", "review/text", "prediction", "confidence"]])


# Start the stream:
query = (
    parsed.writeStream
    .foreachBatch(process_batch)
    .trigger(processingTime=f"{TRIGGER_SECONDS} seconds")
    .start()
)
query.awaitTermination()