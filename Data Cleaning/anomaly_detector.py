import numpy as np 
import pandas as pd


def negativeField(field, curr_dev, test=None):
    """
    Function to filter anomlous records based on negative (therefore meaningless) values of the `field`
    
    args:
    field: string name of the field of interest for anomaly detection 
    curr_dev: `device` field of the device in question 
    
    return:
    faults: pandas dataframe with containing anaomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    """
    faults = curr_dev[curr_dev[field] < 0][['when_captured', 'device']]
    faults['anomaly_type'] = np.repeat(field + ' < 0', faults.shape[0])
    faults['normalized_severity_score'] = 1
    return faults


def rollingMeanDev(field, curr_dev, window, min_period, numStd):
    """
    Function to filter anomlous records based on `numStd` number of deviations away from rolling mean
    
    args:
    field: string name of the field of interest for anomaly detection 
    curr_dev: `device` field of the device in question 
    window: moving window size for the rolling mean and stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from mean for record to be anomalous
    
    return:
    pandas dataframe with containing anaomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlay: holoviews plot object with running mean, +/- numStd lines, and data
    """
    rollingMean = curr_dev[field].rolling(window, min_periods=min_period).mean()
    rollingStdev = curr_dev[field].rolling(window, min_periods=min_period).std()
    upper = rollingMean + (rollingStdev * numStd)
    lower = rollingMean - (rollingStdev * numStd)
    
    # visualize
    x = pd.to_datetime(curr_dev['when_captured']).values
    line1 = hv.Scatter((x,curr_dev[field]), label='data').opts(width=700, ylabel = field)
    line1Mean = hv.Curve((x, rollingMean), label='mean').opts(width=700, color='red', xlabel = 'date')
    line1Upper = hv.Curve((x, upper), label='mean+ 3.stdev').opts(width=700, color='blue')
    line1Lower = hv.Curve((x, lower), label='mean- 3.stdev').opts(width=700, color='blue')
    
    overlay = line1Upper * line1Mean * line1 * line1Lower
    overlay # there exists a save command too

    # return the list of anomalies : records where deviation is >= num_std away from mean
    temp = curr_dev.copy()
    temp['rollingMean'] = rollingMean
    temp['rollingStdev'] = rollingStdev
    temp['normalized_severity_score'] = (temp[field]-temp['rollingMean'])/(numStd*temp['rollingStdev'])
    temp = temp[abs(temp['normalized_severity_score']) >= 1]
    
    # instead of string for anomaly type, we can also make a key:value dictionary 
    # and use that as meta-data for reporting
    temp['anomaly_type'] = np.repeat(field + ' ' + str(numStd) + ' or more away from mean', temp.shape[0])
    return temp[['anomaly_type', 'device', 'normalized_severity_score', 'when_captured']], overlay