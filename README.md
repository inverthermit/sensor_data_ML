# sensor_data_ML
A project for using machine learning techniques for sensor data activity recognition.

## Running instruction
0. System configuration
Config config.json under C:\..\sensor_data_ML-first_release. Fill in the directories. This needs to be done only for the first time of using algorithm.

1. Sensor data to be examined 
Put the quaternion.csv (or any other data file) from the sensor into the path \sensor_data_ML-first_release\real_data

2. Converting CSV into JSON for the script
Goto C:\..\sensor_data_ML-first_release,
Shift+Mouse Right Click, 
and click "Open command window here" to run in cmd:

cd script/process/
python chunk_csv_to_jsons.py

to chunk the real csv data into N json files. Output folder for chunking is C:\..\sensor_data_ML-first_release\real_data\chunked. Frequency and chunking are configured in the script now but later in the config file


3. Creating images for the clustering
Run in cmd:
python compute_position.py
to generate images in the folder tmp/kcm/

NOTE: you need to delete existing images first before running the script

4. Convert images into python data
Run in cmd:
python to_numpy.py
to read all the images into python data files in nparray/kcm/

5. K-means clustering https://en.wikipedia.org/wiki/K-means_clustering
Because it gives similar results compared to other clustering methods but is simpler and faster to use
Run in cmd:
cd ../clustering
python cluster_statistic.py
to generate the csv file recording the number of each cluster in each trial

6. Labelling i.e. supervised learning
Manually label the trial sets in the CSV file by adding a label column. The CSV file is in the path \sensor_data_ML-first_release\tmp\cluster_result_csv
and copy the labeled file to the folder \sensor_data_ML-first_release\tmp\cluster_result_csv\labeled_csv
Please don't change the filename.
Labelling the data following the example file \sensor_data_ML-first_release\tmp\cluster_result_csv\labeled_csv\example.csv

7. Classification by using the ANN https://en.wikipedia.org/wiki/Artificial_neural_network
Because it is sufficient enough and does the job
Run in cmd: 
python classify_cluster_result.py
to generate the final result.
