from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("KafkaIntegration").getOrCreate()

df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "reviews-stream")
    .load()
)

(
    df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    .writeStream
    .outputMode("append")
    .format("console")
    .start()
    .awaitTermination()
)
