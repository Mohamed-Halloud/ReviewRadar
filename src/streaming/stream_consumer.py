from datetime import datetime
from time import time  
import json

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
from pyspark.sql.functions import from_json, col
import logging

from src.model.inference import tokenizer, model, predict_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='logs/spark_consumer.log'
)
logger = logging.getLogger(__name__)

spark = (
    SparkSession.builder
    .appName("KafkaIntegration")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.local.dir", "C:/spark-tmp")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:29092")
    .option("subscribe", "reviews-stream")
    .load()
)

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

parsed = (
    df.selectExpr("CAST(value AS STRING) as json_str")
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)

def log_batch_stats(batch_id, num_reviews, latency, avg_confidence):
    record = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "reviews": num_reviews,
        "latency": round(latency, 3),
        "throughput": round(num_reviews / latency, 2) if latency > 0 else 0,
        "avg_confidence": round(avg_confidence, 3)
    }
    logger.info(json.dumps(record))

def process_batch(batch_df, batch_id):
    pandas_df = batch_df.toPandas()

    if pandas_df.empty:
        logger.info(json.dumps({"timestamp": datetime.now().isoformat(), "batch_id": batch_id, "status": "empty"}))
        return

    start = time()
    try:
        predictions, confidences = predict_batch(pandas_df["review/text"].tolist(), tokenizer, model)
    except Exception as e:
        logger.error(json.dumps({"timestamp": datetime.now().isoformat(), "batch_id": batch_id, "error": str(e)}))
        return 

    latency = time() - start

    pandas_df["prediction"] = predictions
    pandas_df["confidence"] = confidences

    log_batch_stats(batch_id, len(pandas_df), latency, pandas_df['confidence'].mean())
    print(pandas_df[["Id", "review/text", "prediction", "confidence"]])

parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()