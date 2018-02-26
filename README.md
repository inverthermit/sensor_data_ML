# sensor_data_ML
A project for using machine learning techniques for sensor data activity recognition.

## Running instruction

1. Config config.json. Fill in the directories
2. Run script/process/chunk_csv_to_jsons.py to chunk the real csv data into N json files.
3. Run script/process/compute_position.py
4. Run script/process/to_numpy.py
5. Run script/clustering/cluster_statistic.py
6. Label the CSV file
7. Run script/clustering/classify_cluster_result.py
