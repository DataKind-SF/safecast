# Data Cleaning scripts
These files are linked for usage.
## TO DO: list file dependencies -- Done. See below

- [`Data_Cleansing_Single_file.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Data_Cleansing_Single_file.ipynb) file has the implemented data cleaning rules based on what is [documented here](https://github.com/DataKind-SF/safecast/blob/master/Solarcast_data_cleansing.md)
- The [`Run_Data_Cleansing_Solarcasts.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Run_Data_Cleansing_Solarcasts.ipynb) file will internally run the [`Data_Cleansing_Single_file.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Data_Cleansing_Single_file.ipynb) file to call all of its functions. These functions will be then called & run by the function writen inside [`Run_Data_Cleansing_Solarcasts.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Run_Data_Cleansing_Solarcasts.ipynb) . This function returns the cleaned data frame for the raw file.
- [`Write_cleaned_sorted_solarcast_data.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Write_cleaned_sorted_solarcast_data.ipynb) file writes the clean data for each month into a csv file. In addition, it reads all this data back again and concatenates  it into a master datafile and writes it. Ignore parts of this file after the 'Group by device_urn" section
- [`Plotting_Testing_DataFilter_And_anomaly_detection_SC.ipynb`](https://github.com/DataKind-SF/safecast/blob/master/Data%20Cleaning/Plotting_Testing_DataFilter_And_anomaly_detection_SC.ipynb) file reads the master data frame and involves playing around with the data in an attempt to extract seasonality via two methods -- 
  1.  seasonal additive decomposition -- see slides in [google drive](https://drive.google.com/file/d/1aK1UdVJOkmxR94EzqPkTADrSdZElI6Uu/view?usp=sharing) 
  2. Fourier Transforms
 
 In addition, in this file, I have implemented the luminol package for anomaly detection and playing around with it. This pkg should return 
   1. an anomaly score for each measurement in the time series, 
   2. which events are anomalous (when anomaly score crosses a threshold)
   3. and finally the time window *start and end time stamps* for when the anomaly is present.
   
 ## Python script
 In case it is easier to run the data cleaning process as a Python script, please use the following command while in this directory:
 ```
 python -m data_cleaner 2017-09 2020-07
 ```
 The last two arguments in the command are defined as follows:
   1. start_yyyymm -- string with format YYYY-MM; earliest month for which data is available (e.g., 2017-09)
   2. end_yyyymm -- string with format YYYY-MM; latest month for which data is available (e.g., 2020-07)
 
 Contact Saksham Gakhar for more details (saksham.gakhar94@gmail.com) or Edwin Zhang (edwin.james.zhang@gmail.com)
