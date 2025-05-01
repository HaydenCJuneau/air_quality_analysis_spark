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

## Task 3 (Michael): Spark SQL Analysis
In task 3, the goal is to use Spark SQL to analyze air quality data and discover patterns in PM2.5 readings across regions.
- I created a script (`task3_spark_sql_analysis.py`) that reads the cleaned weather data and registers it as a temporary view.
- A UDF was implemented to categorize PM2.5 readings into AQI categories (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous).
- Several analysis types were performed:
  - Hourly aggregation of PM2.5 values to identify daily patterns
  - Window functions (LAG/LEAD) to track sequential changes in readings
  - Correlation analysis between PM2.5, temperature, and humidity
  - Regional analysis to identify pollution hotspots and sustained increases
- The script generates 8 output CSV files:
  - `hourly_analysis.csv` with statistics by hour of day
  - `daily_trend.csv` showing sequential PM2.5 readings
  - `correlation_analysis.csv` with correlation coefficients 
  - `aqi_distribution.csv` showing the distribution of AQI categories
  - `last_24h_avg.csv` with regional 24-hour averages
  - `peak_intervals.csv` identifying top pollution peaks
  - `sustained_increases.csv` showing periods of rising pollution
  - `region_risk.csv` ranking areas by pollution risk
- Key findings from correlation analysis:
  - PM2.5 has a moderate negative correlation with temperature (-0.43)
  - PM2.5 has a weak positive correlation with humidity (0.21)
  - Regions with over 25% "Unhealthy" readings were flagged for priority monitoring

# Task 4: . . .
# Task 5: . . .