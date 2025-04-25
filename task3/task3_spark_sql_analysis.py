
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, max, min, count
from pyspark.sql.functions import lag, lead, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

def create_spark_session():
    return SparkSession.builder \
        .appName("Air Quality SQL Analysis") \
        .getOrCreate()

def calculate_aqi_category(pm25):
    """Calculate AQI category based on PM2.5 value"""
    if pm25 <= 50:
        return "Good"
    elif pm25 <= 100:
        return "Moderate"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 200:
        return "Unhealthy"
    elif pm25 <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def main():
    # Create Spark session
    spark = create_spark_session()
    
    # Read the input data
    df = spark.read.csv("input/weather.csv", header=True, inferSchema=True)
    
    # Register the DataFrame as a temporary view
    df.createOrReplaceTempView("air_quality")
    
    # Register AQI UDF
    aqi_udf = udf(calculate_aqi_category, StringType())
    spark.udf.register("calculate_aqi", aqi_udf)
    
    # 1. Hourly average PM2.5 levels
    hourly_analysis = spark.sql("""
        SELECT 
            hour_of_day,
            AVG(value) as avg_pm25,
            MAX(value) as max_pm25,
            MIN(value) as min_pm25
        FROM air_quality
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """)
    
    # 2. Daily trend analysis using window functions
    daily_trend = spark.sql("""
        SELECT 
            timestamp,
            value as pm25,
            LAG(value) OVER (ORDER BY timestamp) as prev_pm25,
            LEAD(value) OVER (ORDER BY timestamp) as next_pm25,
            calculate_aqi(value) as aqi_category
        FROM air_quality
        ORDER BY timestamp
    """)
    
    # 3. Correlation analysis between PM2.5, temperature, and humidity
    correlation_analysis = spark.sql("""
        SELECT 
            CORR(value, temperature) as pm25_temp_correlation,
            CORR(value, humidity) as pm25_humidity_correlation,
            CORR(temperature, humidity) as temp_humidity_correlation
        FROM air_quality
    """)
    
    # 4. AQI category distribution
    aqi_distribution = spark.sql("""
        SELECT 
            calculate_aqi(value) as aqi_category,
            COUNT(*) as count
        FROM air_quality
        GROUP BY calculate_aqi(value)
        ORDER BY count DESC
    """)
    
    # New Analysis 1: 24-hour average PM2.5 by region
    last_24h_avg = spark.sql("""
        SELECT 
            location,
            AVG(value) as avg_pm25_24h,
            MAX(value) as max_pm25_24h,
            calculate_aqi(AVG(value)) as aqi_category
        FROM air_quality
        WHERE timestamp >= (SELECT MAX(timestamp) FROM air_quality) - INTERVAL 24 HOURS
        GROUP BY location
        ORDER BY avg_pm25_24h DESC
    """)
    
    # New Analysis 2: Peak pollution intervals
    peak_intervals = spark.sql("""
        WITH ranked_data AS (
            SELECT 
                timestamp,
                value as pm25,
                location,
                ROW_NUMBER() OVER (PARTITION BY location ORDER BY value DESC) as peak_rank
            FROM air_quality
        )
        SELECT *
        FROM ranked_data
        WHERE peak_rank <= 5
        ORDER BY location, peak_rank
    """)
    
    # New Analysis 3: Modified trends analysis
    sustained_increases = spark.sql("""
        WITH hourly_avg AS (
            SELECT 
                timestamp,
                location,
                value as pm25,
                LAG(value, 1) OVER (PARTITION BY location ORDER BY timestamp) as prev_hour,
                LAG(value, 3) OVER (PARTITION BY location ORDER BY timestamp) as prev_3hour,
                LAG(value, 6) OVER (PARTITION BY location ORDER BY timestamp) as prev_6hour
            FROM air_quality
        )
        SELECT 
            timestamp,
            location,
            pm25,
            prev_hour,
            prev_3hour,
            prev_6hour,
            ROUND(((pm25 - prev_hour) / prev_hour * 100), 2) as hour_change_pct,
            ROUND(((pm25 - prev_3hour) / prev_3hour * 100), 2) as three_hour_change_pct,
            ROUND(((pm25 - prev_6hour) / prev_6hour * 100), 2) as six_hour_change_pct
        FROM hourly_avg
        WHERE (pm25 > prev_hour AND pm25 > prev_3hour AND pm25 > prev_6hour)
            OR (pm25 > prev_hour AND ((pm25 - prev_hour) / prev_hour * 100) >= 10)
        ORDER BY timestamp
    """)
    
    # New Analysis 4: Region risk ranking
    region_risk = spark.sql("""
        WITH risk_levels AS (
            SELECT 
                location,
                calculate_aqi(AVG(value)) as avg_aqi_category,
                AVG(value) as avg_pm25,
                COUNT(*) as measurements,
                COUNT(CASE WHEN value > 150 THEN 1 END) as unhealthy_counts
            FROM air_quality
            GROUP BY location
        )
        SELECT 
            location,
            avg_aqi_category,
            avg_pm25,
            (unhealthy_counts * 100.0 / measurements) as unhealthy_percentage
        FROM risk_levels
        ORDER BY avg_pm25 DESC
    """)
    
    # Save results to CSV files with headers
    hourly_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/hourly_analysis")
    daily_trend.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/daily_trend")
    correlation_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/correlation_analysis")
    aqi_distribution.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/aqi_distribution")
    last_24h_avg.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/last_24h_avg")
    peak_intervals.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/peak_intervals")
    sustained_increases.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/sustained_increases")
    region_risk.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/region_risk")
    
    # Display some results
    print("Hourly Analysis:")
    hourly_analysis.show()
    
    print("\nCorrelation Analysis:")
    correlation_analysis.show()
    
    print("\nAQI Distribution:")
    aqi_distribution.show()
    
    print("\nLast 24 Hours Average by Region:")
    last_24h_avg.show()
    
    print("\nTop 5 Peak Pollution Intervals by Region:")
    peak_intervals.show()
    
    print("\nSustained Increases in Pollution:")
    sustained_increases.show()
    
    print("\nRegion Risk Ranking:")
    region_risk.show()
    
    spark.stop()

if __name__ == "__main__":
    main()
