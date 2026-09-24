import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
from pyspark.sql.functions import from_json, col

from model.inference import tokenizer, model, predict_batch

spark = SparkSession.builder.appName("KafkaIntegration").getOrCreate()

df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
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


def process_batch(batch_df, batch_id):
    pandas_df = batch_df.toPandas()
    if pandas_df.empty:
        return
    pandas_df["prediction"] = predict_batch(pandas_df["review/text"].tolist(), tokenizer, model)
    print(pandas_df[["Id", "review/text", "prediction"]])

parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()