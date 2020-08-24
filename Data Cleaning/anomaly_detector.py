import numpy as np 
import pandas as pd
from tqdm import tqdm
import holoviews as hv
hv.extension('bokeh')


def negativeFields(fields, df):
    """
    Function to filter anomalous records based on negative (therefore Medianingless) values of the `field`
    
    args:
    fields: list containing the string name(s) of the field(s) of interest for anomaly detection 
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    
    return:
    anomaly_dfs: pandas dataframe with containing anaomalous records: 4 fields are in the list,
                ['anomaly_type','device','normalized_severity_score','when_captured']
    """
    anomaly_dfs = []
    for field in fields:
        anomaly_df = df[df[field] < 0][['when_captured', 'device']]
        anomaly_df['anomaly_type'] = np.repeat(field + ' < 0', anomaly_df.shape[0])
        anomaly_df['normalized_severity_score'] = 1
        anomaly_dfs.append(anomaly_df)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    
    return anomaly_dfs


def rollingMedianDev(fields, df, window, min_period, numStd):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling Median
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling Median and stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from Median for record to be anomalous
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with running Median, +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = []
    for device in tqdm(df['device'].unique(), mininterval = 30):
        overlay_batch = []
        for field in fields:
            deviceFilter = df['device'] == device
            rollingMedian = df[deviceFilter][field].rolling(window, min_periods=min_period).median()
            rollingStdev = df[deviceFilter][field].rolling(window, min_periods=min_period).std()
            upper = rollingMedian + (rollingStdev * numStd)
            lower = rollingMedian - (rollingStdev * numStd)

            # visualize
            x = pd.to_datetime(df[deviceFilter]['when_captured']).values
            line1 = hv.Scatter((x,df[deviceFilter][field]), label='data').opts(width=700, ylabel = field)
            line1Median = hv.Curve((x, rollingMedian), label='Median').opts(width=700, color='red', xlabel = 'date')
            line1Upper = hv.Curve((x, upper), label='Median+ 3.stdev').opts(width=700, color='blue')
            line1Lower = hv.Curve((x, lower), label='Median- 3.stdev').opts(width=700, color='blue')

            overlay = line1Upper * line1Median * line1 * line1Lower
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies : records where deviation is >= num_std away from Median
            temp = df[deviceFilter].copy()
            temp['rollingMedian'] = rollingMedian
            temp['rollingStdev'] = rollingStdev
            temp['normalized_severity_score'] = (temp[field]-temp['rollingMedian'])/(numStd*temp['rollingStdev'])
            temp = temp[abs(temp['normalized_severity_score']) >= 1]

            # instead of string for anomaly type, we can also make a key:value dictionary 
            # and use that as meta-data for reporting
            temp['anomaly_type'] = np.repeat(field + ' ' + str(numStd) + ' or more away from Median', temp.shape[0])
            anomaly_dfs.append(temp[['anomaly_type', 'device', 'normalized_severity_score', 'when_captured']])
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    
    return anomaly_dfs, overlays

def rollingStdevDev(fields, df, window, min_period, numStd):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling stdev
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling stdev and its stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from rolling stdev for record to be anomalous
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with running stdev +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = []
    for device in tqdm(df['device'].unique(), mininterval = 30):
        overlay_batch = []
        for field in fields:
            deviceFilter = df['device'] == device
            rollingStdev = df[deviceFilter][field].rolling(window, min_periods=min_period).std()
            rollingStdevStdev = rollingStdev.rolling(window, min_periods=min_period).std()
            upper = rollingStdev + (rollingStdevStdev * numStd)
            lower = rollingStdev - (rollingStdevStdev * numStd)

            # visualize
            x = pd.to_datetime(df[deviceFilter]['when_captured']).values
            line1 = hv.Scatter((x,df[deviceFilter][field]), label='data').opts(width=700, ylabel = field)
            line1Stdev = hv.Curve((x, rollingStdev), label='Rolling Stdev').opts(width=700, color='red', xlabel = 'date')
            line1Upper = hv.Curve((x, upper), label='Roll. Stdev+ 3.stdev').opts(width=700, color='blue')
            line1Lower = hv.Curve((x, lower), label='Roll. Stdev- 3.stdev').opts(width=700, color='blue')

            overlay = line1Upper * line1Stdev * line1 * line1Lower
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies : records where deviation is >= num_std away from Median
            temp = df[deviceFilter].copy()
            temp['rollingStdev'] = rollingStdev
            temp['rollingStdevStdev'] = rollingStdevStdev
            temp['normalized_severity_score'] = (temp[field]-temp['rollingStdev'])/(numStd*temp['rollingStdevStdev'])
            temp = temp[abs(temp['normalized_severity_score']) >= 1]

            # instead of string for anomaly type, we can also make a key:value dictionary 
            # and use that as meta-data for reporting
            temp['anomaly_type'] = np.repeat(field + ' ' + str(numStd) + ' or more away from local stdev', temp.shape[0])
            anomaly_dfs.append(temp[['anomaly_type', 'device', 'normalized_severity_score', 'when_captured']])
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    
    return anomaly_dfs, overlays

def anomalyDetector(fields, df, anomaly_types=['negativeFields', 'rollingMedianDev'], 
                    rmd_window=None, rmd_min_period=None, rmd_numStd=None,
                    rsd_window=None, rsd_min_period=None, rsd_numStd=None):
    """
    Wrapper function to run checks for specific types of anomalies and aggregate the results/plots
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    anomaly_types: list containing the string name(s) of the anomaly type(s) of interest
    rmd_window: moving window size for the rolling Median and stddev for rollingMedianDev
    rmd_min_period: Minimum number of observations in window required to have a value (otherwise result is NA) for rollingMedianDev
    rmd_numStd: tolerance in number of standard deviations away from Median for record to be anomalous for rollingMedianDev
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: dict of holoviews plot objects with running Median, +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = {}
    
    if 'negativeFields' in anomaly_types:
        anomaly_dfs.append(negativeFields(fields, df))
    if 'rollingMedianDev' in anomaly_types:
        anomaly_dfs_rmd, overlays_rmd = rollingMedianDev(fields, df, rmd_window, rmd_min_period, rmd_numStd)
        anomaly_dfs.append(anomaly_dfs_rmd)
        overlays['rollingMedianDev'] = overlays_rmd
    if 'rollingStdevDev' in anomaly_types:
        anomaly_dfs_rsd, overlays_rsd = rollingStdevDev(fields, df, rsd_window, rsd_min_period, rsd_numStd)
        anomaly_dfs.append(anomaly_dfs_rsd)
        overlays['rollingStdevDev'] = overlays_rsd
    if len(anomaly_dfs) > 1:
        anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
        
    return anomaly_dfs, overlays