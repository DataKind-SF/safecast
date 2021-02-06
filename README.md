# Safecast
Repo for DKSF Project with Safecast

- **Data Ambassadors**: Edwin Zhang (edwin.james.zhang@gmail.com) & others

- **Data Exploration**: First analyzed the Solarcast data. For information about data source and known issues with data, see the file: `exploration_notebooks/Issues_with_SolarcastDevice_data.md`. We’ve accommodated these known data discrepancies through our cleaning protocol (next bullet).

- **Data Cleansing Protocol**: See `exploration_notebooks/Solarcast_data_cleansing.md`

- **Data Cleansing Protocol Base code**:`data_cleaner.py`

- **Automated Anomaly Detection**: See `anomaly_detector.py` 

## Installation and usage instructions
Here is a step-by-step guide to running the `data_cleaner.py` and `anomaly_detector.py` scripts:
1. (recommended/optional) Create a new Python virtual environment for maximum virtual reproducibility. I recommend `conda create anomaly_detection python=3.8`; virtualenv is another option if you don’t have Anaconda installed. Then activate it: `conda activate anomaly_detection`
2. Pull the most recent version of the repo: `git clone https://github.com/sakshamg94/safecast-unsupervised-anomaly-detection.git` and navigate to the directory
3. Install the requirements using the new requirements.txt file: `pip install -r requirements.txt`. At this point we should be set up to run the `data_cleaner.py` and `anomaly_detector.py` scripts in the next two steps.
4. `data_cleaner` takes two inputs: `start_yyyymm`, which is a string with format YYYY-MM for the earliest month for which data is available (e.g., 2017-09), and `end_yyyymm` for the latest month for which data is available (e.g., 2020-07). It assumes that you have the raw data files in the folder `raw_data` and that the raw data files are named like this example: `output-2017-07-01T00_00_00+00_00`. To run `data_cleaner` , use this command: `python3 -m data_cleaner 2017-09 2020-07` (feel free to replace with desired YYYY-MM dates). You should now see individual cleaned files in the `processed_data` folder, as well as a file named `Solarcast-01_Main_Cleaned_Sorted.csv` . This last file is the one used by `anomaly_detector`.
5. `anomaly_detector` takes two optional inputs: devices, a list of device numbers separated by spaces, and anomaly_types, a list of anomaly types to check for. One example command you might run would be: `python anomaly_detector.py --devices 1660294046 3373827677 --anomaly_types negativeFields rollingMedianDev` . If you wanted to run the default of all devices and the following anomaly types (`['negativeFields', 'rollingMedianDev', 'rollingStdevDev']` are the default; `['dataContinuityCheck', 'PMorderingCheck', and 'nightDayDisparity']` are also available), you could also just run `python anomaly_detector.py` without specifying devices or anomaly_types. Running this script should create the output `Anomalies.csv` in the `anomaly_data` directory, which contains information on the anomaly type, affected data field, time of capture, device, and a normalized severity score from 0 to 1. You'll need to create the folder `final_anomaly_data` to write the final .csv.

Summarizing the commands from above:
```
conda create -n anomaly_detection python=3.8
conda activate anomaly_detection
git clone https://github.com/sakshamg94/safecast-unsupervised-anomaly-detection.git
cd safecast-unsupervised-anomaly-detection
pip install -r requirements.txt
python -m data_cleaner 2017-09 2020-07
mkdir final_anomaly_data
python anomaly_detector.py
```
Example outputs from running the above commands have been included in the `cleaned_data` directory. However, two files you'd expect from the commands, a main file containing all the cleaned data (`cleaned_data/Solarcast-01_Main_Cleaned_Sorted.csv`) and the final anomaly data (`final_anomaly_data/anomalies.csv'`) have been omitted due to repo size constraints.

## Self-generating Safecast data
For ease of access, we've provided the data in `raw_data`, which comes from Safecast and has already undergone some basic processing. To download the data and do this basic processing from scratch in the future (if one were updating this analysis for post-July 2020 data, for example), please refer to [this Safecast open-source Ruby script](https://github.com/Safecast/ingest/blob/master/db/generate_reports.rb). The set-up for this Ruby script can be found [here](https://github.com/Safecast/safecastapi/wiki) with additional details linked for [Windows](https://github.com/Safecast/safecastapi/wiki/Dev:-Setup-on-Windows) and [Linux/OS-X](https://github.com/Safecast/safecastapi/wiki/Dev:-Setup-on-Linux-and-OS-X-using-Docker) users. Edwin (contact email addresses at the top of this file) can also forward any questions to Safecast's development team.


