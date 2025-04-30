# Section 4: Spark MLlib - Predictive Modeling Pipeline (corrected features, realistic evaluation)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import os

# Redirect output to a file
import sys
sys.stdout = open("outputs/task4_output.txt", "w")

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("AirQualityMLModeling") \
    .getOrCreate()

# 2. Check if Parquet file exists, otherwise convert CSV to Parquet
csv_path = "input/input.csv"
parquet_path = "input/feature/"

if not os.path.exists(parquet_path):
    print(f"Converting CSV at {csv_path} to Parquet at {parquet_path}...")
    df_csv = spark.read.csv(csv_path, header=True, inferSchema=True)
    df_csv.write.parquet(parquet_path)
    print("Conversion complete.")

# 3. Load feature-engineered dataset
df = spark.read.parquet(parquet_path)

# Rename 'value' column to 'PM25' for consistency
df = df.withColumnRenamed("value", "PM25")

# Add AQI_Category column based on PM25 values
df = df.withColumn(
    "AQI_Category",
    when(col("PM25") <= 50, "Good")
    .when((col("PM25") > 50) & (col("PM25") <= 100), "Moderate")
    .when((col("PM25") > 100) & (col("PM25") <= 150), "Unhealthy for Sensitive Groups")
    .when((col("PM25") > 150) & (col("PM25") <= 200), "Unhealthy")
    .when((col("PM25") > 200) & (col("PM25") <= 300), "Very Unhealthy")
    .otherwise("Hazardous")
)

# 4. Select features (excluding PM25 for regression)
feature_cols = ["temperature", "humidity", "pm25_rolling_avg", "pm25_x_humidity"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)

# 5. Normalize features
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# 6. Prepare regression dataset (label = PM2.5)
df_reg = df.select("features", col("PM25").alias("label"))
train_reg, test_reg = df_reg.randomSplit([0.8, 0.2], seed=42)

# 7. Train and evaluate regression model
reg_model = RandomForestRegressor(featuresCol="features", labelCol="label")
reg_model_fitted = reg_model.fit(train_reg)
predictions_reg = reg_model_fitted.transform(test_reg)

reg_evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
reg_evaluator_r2 = RegressionEvaluator(metricName="r2", labelCol="label", predictionCol="prediction")
print(f"Regression RMSE: {reg_evaluator_rmse.evaluate(predictions_reg):.2f}")
print(f"Regression R²: {reg_evaluator_r2.evaluate(predictions_reg):.2f}")

# 8. Hyperparameter tuning for regression
param_grid_reg = ParamGridBuilder() \
    .addGrid(reg_model.numTrees, [20, 50]) \
    .addGrid(reg_model.maxDepth, [5, 10]) \
    .build()

crossval_reg = CrossValidator(estimator=reg_model,
                               estimatorParamMaps=param_grid_reg,
                               evaluator=reg_evaluator_rmse,
                               numFolds=3)

cv_model_reg = crossval_reg.fit(train_reg)
print("Tuned RMSE (Regression):", reg_evaluator_rmse.evaluate(cv_model_reg.transform(test_reg)))

# 9. Prepare classification dataset (label = AQI Category)
indexer = StringIndexer(inputCol="AQI_Category", outputCol="label")
df_class = indexer.fit(df).transform(df).select("features", "label")
train_class, test_class = df_class.randomSplit([0.8, 0.2], seed=42)

# 10. Train and evaluate classification model
clf_model = RandomForestClassifier(featuresCol="features", labelCol="label")
clf_model_fitted = clf_model.fit(train_class)
predictions_clf = clf_model_fitted.transform(test_class)

clf_evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
clf_evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
print(f"Classification Accuracy: {clf_evaluator_acc.evaluate(predictions_clf):.2f}")
print(f"Classification F1 Score: {clf_evaluator_f1.evaluate(predictions_clf):.2f}")

# 11. Hyperparameter tuning for classification
param_grid_clf = ParamGridBuilder() \
    .addGrid(clf_model.numTrees, [20, 50]) \
    .addGrid(clf_model.maxDepth, [5, 10]) \
    .build()

crossval_clf = CrossValidator(estimator=clf_model,
                               estimatorParamMaps=param_grid_clf,
                               evaluator=clf_evaluator_acc,
                               numFolds=3)

cv_model_clf = crossval_clf.fit(train_class)
print("Tuned Accuracy (Classification):", clf_evaluator_acc.evaluate(cv_model_clf.transform(test_class)))

# 12. Real-time prediction prototype (not active streaming)
print("\n--- Real-Time Prediction Plan ---")
print("1. Ingest streaming data from TCP using Spark Structured Streaming.")
print("2. Apply same preprocessing (assemble + scale).")
print("3. Use best trained model to predict in real-time.")
print("4. Write predictions to sink (e.g., Parquet or PostgreSQL).")

spark.stop()
sys.stdout.close()
