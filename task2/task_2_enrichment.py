from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, to_timestamp, dayofweek, hour, avg, count, min, max
)

# Initialize Spark session
spark = SparkSession.builder.appName("AirQualityEnrichment").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load cleaned air quality data from Task 1
aq_path = "./task1/output/task_1_cleaned.csv"
aq_df = spark.read.option("header", True).csv(aq_path)
aq_df = aq_df.withColumn("timestamp", to_timestamp("timestamp"))

# Load weather data (provide a real or mock file as needed)
weather_path = "./task2/input/weather.csv"
weather_df = spark.read.option("header", True).csv(weather_path)
weather_df = weather_df.withColumn("timestamp", to_timestamp("timestamp"))
weather_df = weather_df.withColumn("temperature", col("temperature").cast("double"))
weather_df = weather_df.withColumn("humidity", col("humidity").cast("double"))

# Join on location and timestamp (exact match)
joined_df = aq_df.join(
    weather_df,
    on=["location", "timestamp"],
    how="inner"
)

# Outlier removal
joined_df = joined_df.filter(
    (col("value").cast("double") > 0) & (col("value").cast("double") < 1000) &
    (col("temperature") > -30) & (col("temperature") < 60) &
    (col("humidity") >= 0) & (col("humidity") <= 100)
)

# Feature engineering
window_spec = Window.partitionBy("location").orderBy("timestamp").rowsBetween(-2, 0)
joined_df = joined_df.withColumn("pm25_rolling_avg", avg(col("value").cast("double")).over(window_spec))
joined_df = joined_df.withColumn("day_of_week", dayofweek("timestamp"))
joined_df = joined_df.withColumn("hour_of_day", hour("timestamp"))
joined_df = joined_df.withColumn("pm25_x_humidity", col("value").cast("double") * col("humidity"))

# Save the enriched data
output_path = "./task2/output/task_2_enriched.csv"
joined_df.coalesce(1).write.mode("overwrite").csv(output_path, header=True)
