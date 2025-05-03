# Project 2: Air Quality Analysis with Spark

In this project, we are given a large dataset and must work with various spark tools to analyze the contents.
Each task will be completed by a different team member in sequence. We will update this readme file with our results.

## Task 1 (Hayden): Ingestion and Cleaning
In task 1, the goal is to collect the data from a simulated tcp server, and do some initial cleaning steps.
- A script was provided to download the zipped files from an s3 bucket, and open them.
- Another script was provided to set up a tcp server. This server waits until a client is connected then it sends all of its data.
- I created a script (`test_reading_client`)which uses spark streaming to ingest the data, ensure it conforms to the correct schema, and place its output into a set of `snappy.parquet` files.
- Next, a script (`task_1_cleaning`) was created to collect these files, turn them into a dataframe, and export them into a single csv for easy further processing. This is how `task-1-raw.csv` was created.
- Finally, some cleaning and aggregation steps were done. The instructions state to look for weather related data to be included in the set along with pm25 readings. However, none were provided in the original data so this part was ignored. Therefore, the processing that was needed was to remove some rows, and convert columns into their proper formats. This is how `task-one-cleaned.csv` was created.
- In addition to this, some statistics were calculated.
    - Value Count: 2394
    - Min Value: 8.0
    - Max Value: 442.0

# Task 2: . . .
# Task 3: . . .
# Task 4: Machine Learning with Spark MLlib
This section focuses on building, evaluating, and optimizing machine learning models to predict PM2.5 levels or classify AQI categories using Apache Spark MLlib. It also includes a plan for integrating real-time predictions using a streaming pipeline.

### Step 1: Feature Selection & Dataset Preparation
Script: feature_selection.py

1) Load cleaned & feature-engineered data (e.g., PM2.5 lag, temperature, humidity, rate-of-change).

2) Select features based on domain knowledge or correlation analysis.

3) Choose regression (PM2.5 prediction) or classification (AQI category).

4) Split dataset into train (80%) and test (20%).
```bash
spark-submit feature_selection.py
```
### Step 2: Train & Evaluate Models
#### Option A: Regression
Models: Linear Regression, Random Forest Regressor
Metrics: RMSE, R²

#### Option B: Classification
Models: Logistic Regression, Random Forest Classifier
Metrics: Accuracy, F1-score

### Step 3: Hyperparameter Tuning

1) Uses ParamGridBuilder and CrossValidator for grid search.

2) Performs 3-fold or 5-fold cross-validation.

3) Outputs best parameters and performance comparison.

Output:
1) Best model parameters

2) Improved RMSE/F1-score

3) Comparison with default model

### Step 4: Real-Time Prediction Integration

#### Contents include:

1) How to load the trained model in a Spark Streaming/TCP pipeline

2) Convert streaming input into feature vector

3) Apply .transform() on the stream for real-time inference

4) Save predictions in output/stream_results/ as Parquet/CSV or PostgreSQL

### Final Outcome
1) Curated feature list and training/test split

2) Trained regression/classification model

3) Evaluation metrics and tuning results

4) Strategy for integrating model into streaming predictions

# Task 5: . . .
