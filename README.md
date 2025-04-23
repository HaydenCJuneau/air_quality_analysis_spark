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
# Task 4: . . .
# Task 5: . . .