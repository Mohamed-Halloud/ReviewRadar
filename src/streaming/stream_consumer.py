from typing import Iterator

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
from pyspark.sql.functions import from_json, col

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


MODEL_PATH = "/home/jovyan/work/model/best_model"
MAX_LENGTH    = 128 
BATCH_SIZE    = 16
TORCH_THREADS = 1  

def load_model():
    torch.set_num_threads(TORCH_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return tokenizer, model

tokenizer, model = load_model()  

def predict_batch(text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).tolist()
    return prediction



def process_batch(batch_df, batch_id):
    pandas_df = batch_df.toPandas()
    if pandas_df.empty:
        return
    pandas_df["prediction"] = predict_batch(pandas_df["review/text"].tolist(), tokenizer, model)
    print(pandas_df[["Id", "review/text", "prediction"]])

parsed.writeStream.foreachBatch(process_batch).start().awaitTermination()