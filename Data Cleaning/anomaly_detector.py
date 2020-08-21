import numpy as np 
import pandas as pd
from tqdm import tqdm
import holoviews as hv
hv.extension('bokeh')


def negativeFields(fields, df):
    """
    Function to filter anomalous records based on negative (therefore meaningless) values of the `field`
    
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


def rollingMeanDev(fields, df, window, min_period, numStd):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling mean
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling mean and stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from mean for record to be anomalous
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with running mean, +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = []
    for device in tqdm(df['device'].unique(), mininterval = 30):
        overlay_batch = []
        for field in fields:
            deviceFilter = df['device'] == device
            rollingMean = df[deviceFilter][field].rolling(window, min_periods=min_period).mean()
            rollingStdev = df[deviceFilter][field].rolling(window, min_periods=min_period).std()
            upper = rollingMean + (rollingStdev * numStd)
            lower = rollingMean - (rollingStdev * numStd)

            # visualize
            x = pd.to_datetime(df[deviceFilter]['when_captured']).values
            line1 = hv.Scatter((x,df[deviceFilter][field]), label='data').opts(width=700, ylabel = field)
            line1Mean = hv.Curve((x, rollingMean), label='mean').opts(width=700, color='red', xlabel = 'date')
            line1Upper = hv.Curve((x, upper), label='mean+ 3.stdev').opts(width=700, color='blue')
            line1Lower = hv.Curve((x, lower), label='mean- 3.stdev').opts(width=700, color='blue')

            overlay = line1Upper * line1Mean * line1 * line1Lower
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies : records where deviation is >= num_std away from mean
            temp = df[deviceFilter].copy()
            temp['rollingMean'] = rollingMean
            temp['rollingStdev'] = rollingStdev
            temp['normalized_severity_score'] = (temp[field]-temp['rollingMean'])/(numStd*temp['rollingStdev'])
            temp = temp[abs(temp['normalized_severity_score']) >= 1]

            # instead of string for anomaly type, we can also make a key:value dictionary 
            # and use that as meta-data for reporting
            temp['anomaly_type'] = np.repeat(field + ' ' + str(numStd) + ' or more away from mean', temp.shape[0])
            anomaly_dfs.append(temp[['anomaly_type', 'device', 'normalized_severity_score', 'when_captured']])
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    
    return anomaly_dfs, overlays


def anomalyDetector(fields, df, anomaly_types=['negativeFields', 'rollingMeanDev'], 
                    rmd_window=None, rmd_min_period=None, rmd_numStd=None):
    """
    Wrapper function to run checks for specific types of anomalies and aggregate the results/plots
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    anomaly_types: list containing the string name(s) of the anomaly type(s) of interest
    rmd_window: moving window size for the rolling mean and stddev for rollingMeanDev
    rmd_min_period: Minimum number of observations in window required to have a value (otherwise result is NA) for rollingMeanDev
    rmd_numStd: tolerance in number of standard deviations away from mean for record to be anomalous for rollingMeanDev
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: dict of holoviews plot objects with running mean, +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = {}
    
    if 'negativeFields' in anomaly_types:
        anomaly_dfs.append(negativeFields(fields, df))
    if 'rollingMeanDev' in anomaly_types:
        anomaly_dfs_rmd, overlays_rmd = rollingMeanDev(fields, df, rmd_window, rmd_min_period, rmd_numStd)
        anomaly_dfs.append(anomaly_dfs_rmd)
        overlays['rollingMeanDev'] = overlays_rmd
    if len(anomaly_dfs) > 1:
        anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
        
    return anomaly_dfs, overlays