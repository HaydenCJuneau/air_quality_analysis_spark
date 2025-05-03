import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import plotly.express as px

#Ingestion
#Start Spark session
spark = SparkSession.builder.appName("AirQualityPipeline").getOrCreate()
#Redirect output to a file
sys.stdout = open("outputs/task5_output.txt", "w")

#Check if Parquet file exists, otherwise convert CSV to Parquet
csv_path = "input/input.csv"
parquet_path = "input/feature/"
if not os.path.exists(parquet_path):
    print(f"Converting CSV at {csv_path} to Parquet at {parquet_path}...")
    df_csv = spark.read.csv(csv_path, header=True, inferSchema=True)
    df_csv.write.parquet(parquet_path)
    print("Conversion complete.")
# Load feature-engineered dataset
df = spark.read.parquet(parquet_path)
df = df.withColumnRenamed("value", "PM25")
#Data Transformation
#Add AQI_Category column based on PM25 values
df = df.withColumn("AQI_Category", when(col("PM25") <= 50, "Good").when((col("PM25") > 50) & (col("PM25") <= 100), "Moderate").when((col("PM25") > 100) & (col("PM25") <= 150), "Unhealthy for Sensitive Groups").when((col("PM25") > 150) & (col("PM25") <= 200), "Unhealthy").when((col("PM25") > 200) & (col("PM25") <= 300), "Very Unhealthy").otherwise("Hazardous"))
#SQL Analysis
df.createOrReplaceTempView("df_view")
aqi_category_counts_sql = spark.sql("SELECT AQI_Category, COUNT(*) AS count FROM df_view GROUP BY AQI_Category").toPandas()

#ML Pipeline (Regression and Classification)

#Regression Model Preparation
feature_cols = ["temperature", "humidity", "pm25_rolling_avg", "pm25_x_humidity"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df = assembler.transform(df)
#Normalize features
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)
#Prepare regression dataset
df_reg = df.select("features", col("PM25").alias("label"), "temperature", "humidity", "timestamp")
train_reg, test_reg = df_reg.randomSplit([0.8, 0.2], seed=42)
#Train and evaluate regression model
reg_model = RandomForestRegressor(featuresCol="features", labelCol="label")
reg_model_fitted = reg_model.fit(train_reg)
predictions_reg = reg_model_fitted.transform(test_reg)
predictions_reg_df = predictions_reg.toPandas()
predictions_reg_df.rename(columns={"label": "PM25"}, inplace=True)
reg_evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
reg_evaluator_r2 = RegressionEvaluator(metricName="r2", labelCol="label", predictionCol="prediction")
print(f"Regression RMSE: {reg_evaluator_rmse.evaluate(predictions_reg):.2f}")
print(f"Regression R²: {reg_evaluator_r2.evaluate(predictions_reg):.2f}")
#Hyperparameter tuning for regression
param_grid_reg = ParamGridBuilder().addGrid(reg_model.numTrees, [20, 50]).addGrid(reg_model.maxDepth, [5, 10]).build()
crossval_reg = CrossValidator(estimator=reg_model,estimatorParamMaps=param_grid_reg,evaluator=reg_evaluator_rmse,numFolds=3)
cv_model_reg = crossval_reg.fit(train_reg)
print("Tuned RMSE (Regression):", reg_evaluator_rmse.evaluate(cv_model_reg.transform(test_reg)))
#Classification Model Preparation
indexer = StringIndexer(inputCol="AQI_Category", outputCol="label")
df_class = indexer.fit(df).transform(df).select("features", "label")
train_class, test_class = df_class.randomSplit([0.8, 0.2], seed=42)
#Train and evaluate classification model
clf_model = RandomForestClassifier(featuresCol="features", labelCol="label")
clf_model_fitted = clf_model.fit(train_class)
predictions_clf = clf_model_fitted.transform(test_class)
predictions_clf_df = predictions_clf.toPandas()
clf_evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
clf_evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label", predictionCol="prediction")
print(f"Classification Accuracy: {clf_evaluator_acc.evaluate(predictions_clf):.2f}")
print(f"Classification F1 Score: {clf_evaluator_f1.evaluate(predictions_clf):.2f}")
#Hyperparameter tuning for classification
param_grid_clf = ParamGridBuilder().addGrid(clf_model.numTrees, [20, 50]).addGrid(clf_model.maxDepth, [5, 10]).build()
crossval_clf = CrossValidator(estimator=clf_model, estimatorParamMaps=param_grid_clf, evaluator=clf_evaluator_acc, numFolds=3)
cv_model_clf = crossval_clf.fit(train_class)
print("Tuned Accuracy (Classification):", clf_evaluator_acc.evaluate(cv_model_clf.transform(test_class)))

#Visualization and Dashboard

#Time-Series Line Charts: Overlaid actual vs. predicted PM2.5 levels
fig = px.line(predictions_reg.toPandas(), x='timestamp', y=['PM25', 'prediction'], title='Actual vs Predicted PM2.5 Levels')
fig.update_layout(xaxis_title='Time', yaxis_title='PM2.5')
fig.show()
#Spike Event Timelines: Highlight intervals exceeding safe thresholds
spike_events = predictions_reg.filter(predictions_reg['prediction'] > 100).toPandas()
fig = px.scatter(spike_events, x='timestamp', y='prediction', color='AQI_Category', title='PM2.5 Spike Events')
fig.update_layout(xaxis_title='Time', yaxis_title='PM2.5')
fig.show()
#AQI Classification Breakdown: Illustrate proportions of Good/Moderate/Unhealthy categories
fig = px.pie(aqi_category_counts_sql, names='AQI_Category', values='count', title='AQI Category Breakdown')
fig.show()
#Correlation Plots: Show relationships among PM2.5, temperature, and humidity
correlation_data = predictions_reg_df[['PM25', 'temperature', 'humidity']]
correlation_matrix = correlation_data.corr()
fig = plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()

#Persistence & Reporting

#Save Regression Predictions (Actual vs Predicted)
predictions_reg_df.to_parquet("output/ml_regression_predictions.parquet")
print("Regression predictions saved to Parquet.")
#Save Classification Predictions (AQI Categories)
predictions_clf_df.to_parquet("output/ml_classification_predictions.parquet")
print("Classification predictions saved to Parquet.")
#Save Aggregated Statistics (e.g., Mean, Min, Max of PM2.5)
aggregated_stats = predictions_reg_df[['PM25']].describe()
aggregated_stats.to_csv("output/aggregated_stats.csv")
print("Aggregated statistics saved to CSV.")
#Save SQL Query Results (e.g., AQI Category Counts)
aqi_category_counts_sql.to_csv("output/aqi_category_counts.csv")
print("SQL query results saved to CSV.")
#Save Final Reports
final_report = {"Regression RMSE": reg_evaluator_rmse.evaluate(predictions_reg), "Regression R²": reg_evaluator_r2.evaluate(predictions_reg), "Classification Accuracy": clf_evaluator_acc.evaluate(predictions_clf), "Classification F1 Score": clf_evaluator_f1.evaluate(predictions_clf)}
final_report_df = pd.DataFrame(list(final_report.items()), columns=["Metric", "Value"])
final_report_df.to_csv("output/final_report.csv", index=False)
print("Final report saved to CSV.")

#Closing the Script
spark.stop()
print("Spark session stopped.")
sys.stdout.close()
print("Script execution complete.")