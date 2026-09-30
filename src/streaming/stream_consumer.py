from time import time  

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
from pyspark.sql.functions import from_json, col
import logging
from src.model.inference import tokenizer, model, predict_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

spark = (
    SparkSession.builder
    .appName("KafkaIntegration")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.local.dir", "C:/spark-tmp")
    .getOrCreate()
)

# Read stream from Kafka topic (localhost since running outside Docker)
df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:29092")
    .option("subscribe", "reviews-stream")
    .load()
)

# Schema of the incoming JSON messages
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

# Parse raw Kafka bytes into structured columns
parsed = (
    df.selectExpr("CAST(value AS STRING) as json_str")
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)


def process_batch(batch_df, batch_id):
    pandas_df = batch_df.toPandas()

    if pandas_df.empty:
        logger.info(f"batch {batch_id} empty, skipping")
        return

    start = time()
    try:
        # Run inference on the whole batch
        predictions, confidences = predict_batch(pandas_df["review/text"].tolist(), tokenizer, model)

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {str(e)}")
        return 

    latency = time() - start

    pandas_df["prediction"] = predictions
    pandas_df["confidence"] = confidences
    
    # Log batch stats
    logger.info(
        f"Batch {batch_id}: processed {len(pandas_df)} reviews in {latency:.3f}s "
        f"({len(pandas_df)/latency:.1f} reviews/sec), "
        f"avg_confidence={pandas_df['confidence'].mean():.2f}"
    )
    print(pandas_df[["Id", "review/text", "prediction", "confidence"]])

parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()