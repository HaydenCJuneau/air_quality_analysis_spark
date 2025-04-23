from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import desc, count, rank, col

# Initialize Spark Session
spark: SparkSession = SparkSession.builder.appName("AirQualityAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load DataFrames
df = spark.read.parquet("/opt/bitnami/spark/Air/parsed")

def explore(df: DataFrame) -> DataFrame:
    df = df.na.drop(subset=["value"]).drop("parsed_values")
    df.show()
    return df
    

# Save result
explore(df).coalesce(1).write.mode("overwrite").csv("/opt/bitnami/spark/Air/output", header=True)
