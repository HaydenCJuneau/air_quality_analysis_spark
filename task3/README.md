# Air Quality Analysis - Task 3

This section implements Spark SQL analysis for air quality data.

## Features

1. **AQI Classification**
   - Implements a UDF to categorize PM2.5 readings into AQI categories
   - Categories: Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous

2. **Trend Analysis**
   - Hourly average PM2.5 levels
   - Daily trends using window functions (LAG and LEAD)
   - Correlation analysis between PM2.5, temperature, and humidity

3. **Data Views**
   - Creates temporary views for efficient querying
   - Implements window functions for sequential analysis

## Output Files

The analysis generates four CSV files in the `task3/output` directory:
- `hourly_analysis`: Average, max, and min PM2.5 levels by hour
- `daily_trend`: Sequential PM2.5 values with previous and next readings
- `correlation_analysis`: Correlation coefficients between metrics
- `aqi_distribution`: Distribution of AQI categories

## Running the Analysis

```bash
cd /workspaces/air_quality_analysis_spark/task3
python spark_sql_analysis.py
```

## Prerequisites

- PySpark
- Input data in CSV format at `task3/input/weather.csv`
