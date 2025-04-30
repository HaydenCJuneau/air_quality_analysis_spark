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

# Task 2 (Vinay):  Data Enrichment and Feature Engineering
## Overview
 This folder contains scripts and outputs for Task 2 of the air_quality_analysis_spark project.
 Objective: Enrich cleaned air quality data by joining with weather data (temperature and humidity), handle outliers, and engineer features for downstream analytics and modeling.##

## Folder Structure
    air_quality_analysis_spark/
      └── task2/
          ├── input/
          │     └── weather.csv        # Generated weather data (randomized, >1000 rows)
          ├── output/
          │     └── task_2_enriched.csv  # Enriched dataset after joining and feature engineering
          └── task_2_enrichment.py     # Main script for Task 2
          └── get_weather_data.py      # Script to generate mock weather data

### How It Works
 1. Generate Weather Data
 - Since real weather API access is unreliable or limited, we generate a mock weather.csv with random but realistic temperature and humidity values for the required date range and location.
 - To generate historical weather data:
 ```bash
    python3 task2/get_weather_data.py
 ```
- This creates hourly weather data for "New Delhi-8118" for a date range covering all air quality timestamps, ensuring >1000 rows.

- Columns: location, timestamp, temperature, humidity

2. Enrich Air Quality Data
    The script task_2_enrichment.py performs the following:
    - Loads cleaned air quality data from task1/output/task_1_cleaned.csv
    - Loads weather data from task2/input/weather.csv
    - Joins both datasets on location and timestamp (with a configurable time window for robust matching)
    - Handles outliers in both air quality and weather features
    Engineers new features: 
    - Day of week and hour of day
    - Rolling average of PM2.5
    - Interaction term: PM2.5 × humidity
    - Saves the enriched result to task2/output/task_2_enriched.csv
- To run enrichment:
```bash
spark-submit task2/task_2_enrichment.py
```

# Task 3: . . .
# Task 4: . . .
# Task 5: . . .
