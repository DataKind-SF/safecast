import numpy as np 
import pandas as pd
from tqdm import tqdm
import holoviews as hv
import datetime
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

def PMorderingCheck(df):
    """
    Function to check PM1.0<PM2.5<PM10.0 (by definition)
    If any record violates above definition, report as anomaly
    
    args:
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    
    return:
    anomaly_dfs: pandas dataframe with containing anaomalous records: 4 fields are in the list,
                ['anomaly_type','device','normalized_severity_score','when_captured']
    """
    anomaly_dfs = []

    anomaly_df1 = df[df['pms_pm02_5'] < df['pms_pm01_0']][['when_captured', 'device']]
    anomaly_df1['anomaly_type'] = np.repeat('pms_pm02_5 < pms_pm01_0', anomaly_df1.shape[0])
    anomaly_df1['normalized_severity_score'] = 1

    anomaly_df2 = df[df['pms_pm10_0'] < df['pms_pm01_0']][['when_captured', 'device']]
    anomaly_df2['anomaly_type'] = np.repeat('pms_pm10_0 < pms_pm01_0', anomaly_df2.shape[0])
    anomaly_df2['normalized_severity_score'] = 1

    anomaly_df3 = df[df['pms_pm10_0'] < df['pms_pm02_5']][['when_captured', 'device']]
    anomaly_df3['anomaly_type'] = np.repeat('pms_pm10_0 < pms_pm02_5', anomaly_df3.shape[0])
    anomaly_df3['normalized_severity_score'] = 1

    anomaly_dfs.append(anomaly_df1)
    anomaly_dfs.append(anomaly_df2)
    anomaly_dfs.append(anomaly_df3)

    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    
    return anomaly_dfs.reset_index(drop=True)

def dataContinuityCheck(df, day_gap = 2, devices = None):
    """
    Function to check if two consecutive records for a device differ by more than `day_gap` (double)
    If there are consec records that differ by > day_gap, report them as anomalies
    
    args:
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    day_gap(double): maximum allowed gap (in units of days) between two consecutive records for a device
    devices: List of devices to run the check for. If None, then all devices in dataframe df
    
    return:
    anomaly_dfs: pandas dataframe with containing anaomalous records: 4 fields are in the list,
                ['anomaly_type','device','normalized_severity_score','when_captured']
    """
    anomaly_dfs = []
    if devices is None:
        devices = df['device'].unique()
    gf = df.groupby('device')
    for device in devices:
        curr_dev = gf.get_group(device)
        curr_dev_noNan = curr_dev.dropna()
        mask = pd.to_datetime(curr_dev_noNan['when_captured']).diff().dt.days >=day_gap
        anomaly_df = curr_dev_noNan[mask][['when_captured', 'device']]
        anomaly_df['anomaly_type'] = np.repeat('more than ' + str(day_gap) +  ' days of gap between consecutive valid records (post cleaning)', anomaly_df.shape[0])
        anomaly_df['normalized_severity_score'] = 1
        anomaly_dfs.append(anomaly_df)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0).reset_index(drop=True) 
    
    return anomaly_dfs

def rollingMedianDev(fields, df, window, min_period, numStd, devices=None):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling Median
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling Median and stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from Median for record to be anomalous
    devices: List of devices to run the check for. If None, then all devices in dataframe df
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with running Median, +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = []
    if devices is None:
        devices = df['device'].unique()
    for device in tqdm(devices, mininterval = 30):
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

def rollingStdevDev(fields, df, window, min_period, numStd, devices=None):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling stdev
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling stdev and its stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NA)
    numStd: tolerance in number of standard deviations away from rolling stdev for record to be anomalous
    devices: List of devices to run the check for. If None, then all devices in dataframe df
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with running stdev +/- numStd lines, and data
    """
    anomaly_dfs = []
    overlays = []
    if devices is None:
        devices = df['device'].unique()
    for device in tqdm(devices, mininterval = 30):
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

def nightDayDisparity(fields, df, min_rec, day_start, day_end, devices = None):
    """
    Function to filter anomalous records for nights where the median 
    nighttime field value > median daytime field value for the preceding day
    (1 anomaly per 24 hr period)
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    Note these can only bete AQ fields
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    min_rec: minimum nuymber of readings for there to be a defined (non-Nan) daytime or nightime median field value
    day_start: start hour of the day (like 6 for 0600hrs)
    day_end: end hour for the day (like 18 for 1800hrs)
    Note: daytime on a date is defined as the hours between `day_start` and `day_end`
    nighttime on a date is defined as the hours between `day_end` of the current_date and 
    the `day_start` of the next_date (if next date records are available in the records)
    devices: List of devices to run the check for. If None, then all devices in dataframe df
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: holoviews plot objects with  data segregated by day, night and anomalous positions that need attention
    """
    anomaly_dfs = pd.DataFrame()
    overlays = []
    gf = df.groupby('device')
    # add the hour of day column to the data frame
    df['hod'] = pd.to_datetime(df['when_captured']).dt.hour
    df['date'] = pd.to_datetime(df['when_captured']).dt.date
    
    if devices is None:
        devices = df['device'].unique()
        
    for device in tqdm(devices, mininterval = 30):
        deviceFilter = df['device'] == device
        overlay_batch = []
        df_dev = gf.get_group(device)
        sf = df_dev.groupby('date')
        dates = np.unique(df_dev['date'])
        # ensure dates are all in increasing order
        assert(np.all(np.diff(dates).astype('timedelta64[h]').astype('float')) >=0) 
        
        for field in fields:
            # dictionary of [day_median, night_median]for the "field" on that date (with date as the key)
            medians_sf = {}     
            # day and night medians are close together, differ by 1-2 or sometimes 0.
            # report dates and devices as anomalies when the date's night time median is larger than day time median 
            # -- works for PMS only -- not for radiation
            for curr_date in dates:
                # get the median for daytime value for the current date
                day_mask = (sf.get_group(curr_date).hod >=day_start) & (sf.get_group(curr_date).hod < day_end)
                day_vals= sf.get_group(curr_date)[field][day_mask]
                if len(day_vals) >= min_rec:
                    day_median = day_vals.median()
                else:
                    day_median = np.nan

                # get the median nighttime value for the current date
                night_mask_curr = sf.get_group(curr_date).hod >= day_end
                night_vals_curr = sf.get_group(curr_date)[field][night_mask_curr]

                night_vals_next = []
                if curr_date + datetime.timedelta(1) in dates:
                    night_mask_next = sf.get_group(curr_date + datetime.timedelta(1)).hod < day_start
                    night_vals_next = sf.get_group(curr_date+ datetime.timedelta(1))[field][night_mask_next]

                night_vals = night_vals_curr.append(night_vals_next)
                if len(night_vals) >= min_rec:
                    night_median = night_vals.median()
                else:
                    night_median = np.nan

                medians_sf[curr_date] = [day_median, night_median]
            
            # create a temporary dictionary to store the "start" timestamps 
            # for night datapoints where night median>day's median
            temp= {}
            temp['when_captured']  = []
            temp['normalized_severity_score'] = []
            for date, lst in medians_sf.items():
                if lst[1]>lst[0]: # nighttime PM > daytime PM
                    temp['when_captured'].append(sf.get_group(date).iloc[0].when_captured) 
                    # the the first record to direct user to anomalous night
                    temp['normalized_severity_score'].append((lst[1]-lst[0] + 1)/(lst[0]+1))
            temp['anomaly_type'] = np.repeat('nighttime median ' + field + ' > daytime median', len(temp['when_captured']))
            temp['device'] = np.repeat(device, len(temp['when_captured']))
            
            #convert dictionary to dataframe
            temp_df = pd.DataFrame.from_dict(temp)
            
            # visualize
            night_filter = (df_dev["hod"] >day_end) | (df_dev["hod"] <day_start)
            morng_filter = ~(night_filter)

            night = hv.Scatter((pd.to_datetime(df_dev['when_captured'][night_filter]).values,\
                                              df_dev[field][night_filter]), label='night').opts(width=800, ylabel = field)
            morng = hv.Scatter((pd.to_datetime(df_dev['when_captured'][morng_filter]).values,\
                                                       df_dev[field][morng_filter]), label='day').opts(width=800, xlabel = 'date')

            limit = max(df_dev[field][morng_filter]) # only for illustration purposes -- constant 
            x_anom = pd.to_datetime(temp_df['when_captured']).values
            anoms = hv.Points((x_anom, np.repeat(limit, len(x_anom))), label='day-night anomalies').opts(width=800, color='black')

            overlay = morng*night*anoms
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies 
            anomaly_dfs = anomaly_dfs.append(temp_df, ignore_index=True)
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    
    return anomaly_dfs, overlays

def anomalyDetector(fields, df, anomaly_types=['negativeFields', 'rollingMedianDev', 'rollingStdevDev'], 
                    rmd_window=None, rmd_min_period=None, rmd_numStd=None,
                    rsd_window=None, rsd_min_period=None, rsd_numStd=None, 
                    ndd_min_rec = 5, ndd_day_start = 6, ndd_day_end = 18, 
                    day_gap = 2, devices = None):
    """
    Wrapper function to run checks for specific types of anomalies and aggregate the results/plots
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    anomaly_types: list containing the string name(s) of the anomaly type(s) of interest
    rmd_window: moving window size for the rolling Median and stddev for rollingMedianDev
    rmd_min_period: Minimum number of observations in window required to have a value (otherwise result is NA) for rollingMedianDev
    rmd_numStd: tolerance in number of standard deviations away from Median for record to be anomalous for rollingMedianDev
    ndd_min_rec: minimum nuymber of readings for there to be a defined (non-Nan) daytime or nightime median field value
    ndd_day_start: start hour of the day (like 6 for 0600hrs)
    ndd_day_end: end hour for the day (like 18 for 1800hrs)
    day_gap(double): maximum allowed gap (in units of days) between two consecutive records for a device
    devices: List of devices to run the check for. If None, then all devices in dataframe df
    
    return:
    pandas dataframe containing anomalous records: 4 fields are in the list
            ['anomaly_type','device','normalized_severity_score','when_captured']
    overlays: dict of holoviews plot objects 
    """
    anomaly_dfs = []
    overlays = {}
    
    if 'negativeFields' in anomaly_types:
        anomaly_dfs.append(negativeFields(fields, df))
    if 'dataContinuityCheck' in anomaly_types:
        anomaly_dfs.append(dataContinuityCheck(df, day_gap, devices))
    if 'PMorderingCheck' in anomaly_types:
        anomaly_dfs.append(PMorderingCheck(df))
    if 'rollingMedianDev' in anomaly_types:
        anomaly_dfs_rmd, overlays_rmd = rollingMedianDev(fields, df, rmd_window, rmd_min_period, rmd_numStd, devices)
        anomaly_dfs.append(anomaly_dfs_rmd)
        overlays['rollingMedianDev'] = overlays_rmd
    if 'rollingStdevDev' in anomaly_types:
        anomaly_dfs_rsd, overlays_rsd = rollingStdevDev(fields, df, rsd_window, rsd_min_period, rsd_numStd, devices)
        anomaly_dfs.append(anomaly_dfs_rsd)
        overlays['rollingStdevDev'] = overlays_rsd
    if 'nightDayDisparity' in anomaly_types:
        anomaly_dfs_ndd, overlays_ndd = nightDayDisparity(fields, df, ndd_min_rec, ndd_day_start, ndd_day_end, devices)
        anomaly_dfs.append(anomaly_dfs_ndd)
        overlays['nightDayDisparity'] = overlays_ndd
    if len(anomaly_dfs) > 1:
        anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
        
    return anomaly_dfs, overlays