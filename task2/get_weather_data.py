import csv
import pandas as pd
from datetime import datetime, timedelta
import random

def create_random_weather_data(start_date, end_date, location_name, output_file):
    """Creates a weather.csv file with random temperature and humidity data."""
    
    # Convert start and end dates to datetime objects
    start = datetime.fromisoformat(start_date + 'T00:00:00')
    end = datetime.fromisoformat(end_date + 'T23:00:00')

    # Generate a list of hourly timestamps
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
        current += timedelta(hours=1)

    # Generate random temperature and humidity values
    temperatures = [round(random.uniform(5, 35), 1) for _ in range(len(timestamps))]  # Temperature between 5 and 35
    humidities = [round(random.uniform(30, 90), 1) for _ in range(len(timestamps))]  # Humidity between 30 and 90

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'location': [location_name] * len(timestamps),
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidities
    })

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Weather data saved to {output_file}")

if __name__ == "__main__":
    start_date = "2025-01-01"  # Start of 2025
    end_date = "2025-02-10"  # Extend the end date to get more than 1000 rows(40 days)
    output_file = "task2/input/weather.csv"
    location_name = "New Delhi-8118"

    create_random_weather_data(start_date, end_date, location_name, output_file)
