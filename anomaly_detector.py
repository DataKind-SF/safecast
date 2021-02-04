import numpy as np 
import pandas as pd
from tqdm import tqdm
import holoviews as hv
hv.extension('bokeh')
import datetime
import argparse

def negativeFields(fields, df):
    """
    Function to filter anomalous records based on negative (therefore meaningless) values of the `field`
    
    args:
    fields: list containing the string name(s) of the field(s) of interest for anomaly detection 
    		fields can be 'pms_pm10_0', 'pms_pm02_5', 'pms_pm01_0', 'lnd_7318u', 'lnd_7318c'
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    
    return:
    anomaly_dfs: pandas dataframe containing anomalous records: 5 fields are in the list,
                ['anomaly_type', 'anomaly_details', 'device','severity_score','when_captured']
    """
    anomaly_dfs = []
    for field in fields:
        anomaly_df = df[df[field] < 0][['when_captured', 'device']]
        anomaly_df['anomaly_details'] = np.repeat(field + ' < 0', anomaly_df.shape[0])
        anomaly_df['severity_score'] = 1
        anomaly_dfs.append(anomaly_df)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    anomaly_dfs['anomaly_type'] = 'Negative fields'
    
    return anomaly_dfs

def PMorderingCheck(df):
    """
    Function to check PM1.0<PM2.5<PM10.0 (by definition)
    If any record violates above definition, report as anomaly
    
    args:
    df: pandas dataframe containing 'device', 'pms_pm01_0', 'pms_pm02_5', 'pms_pm10_0', and 'when_captured'
    
    return:
    anomaly_dfs: pandas dataframe with containing anomalous records: 5 fields are in the list,
                ['anomaly_type', 'anomaly_details', device','severity_score','when_captured']
    """
    anomaly_dfs = []

    anomaly_df1 = df[df['pms_pm02_5'] < df['pms_pm01_0']][['when_captured', 'device']]
    anomaly_df1['anomaly_details'] = np.repeat('pms_pm02_5 < pms_pm01_0', anomaly_df1.shape[0])
    anomaly_df1['severity_score'] = 1

    anomaly_df2 = df[df['pms_pm10_0'] < df['pms_pm01_0']][['when_captured', 'device']]
    anomaly_df2['anomaly_details'] = np.repeat('pms_pm10_0 < pms_pm01_0', anomaly_df2.shape[0])
    anomaly_df2['severity_score'] = 1

    anomaly_df3 = df[df['pms_pm10_0'] < df['pms_pm02_5']][['when_captured', 'device']]
    anomaly_df3['anomaly_details'] = np.repeat('pms_pm10_0 < pms_pm02_5', anomaly_df3.shape[0])
    anomaly_df3['severity_score'] = 1

    anomaly_dfs.append(anomaly_df1)
    anomaly_dfs.append(anomaly_df2)
    anomaly_dfs.append(anomaly_df3)

    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    anomaly_dfs['anomaly_type'] = 'PM ordering'
    
    return anomaly_dfs.reset_index(drop=True)

def dataContinuityCheck(df, day_gap=2):
    """
    Function to check if two consecutive records for a device differ by more than `day_gap`
    If there are consecutive records that differ by > `day_gap` number of days, report them as anomalies
    
    args:
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    day_gap(double): maximum allowed gap (in units of days) between two consecutive records for a device
    
    return:
    anomaly_dfs: pandas dataframe with containing anomalous records: 5 fields are in the list,
                ['anomaly_type', 'anomaly_details', 'device','severity_score','when_captured']
    """
    anomaly_dfs = []
    gf = df.groupby('device')
    for device in df['device'].unique():
        curr_dev = gf.get_group(device)
        curr_dev_noNan = curr_dev.dropna()
        mask = pd.to_datetime(curr_dev_noNan['when_captured']).diff().dt.days >=day_gap
        anomaly_df = curr_dev_noNan[mask][['when_captured', 'device']]
        anomaly_df['anomaly_details'] = np.repeat('more than ' + str(day_gap) +  ' days of gap between consecutive valid records (post cleaning)', anomaly_df.shape[0])
        anomaly_df['severity_score'] = 1
        anomaly_dfs.append(anomaly_df)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0).reset_index(drop=True)
    anomaly_dfs['anomaly_type'] = 'Data continuity'
    
    return anomaly_dfs

def rollingMedianDev(fields, df, window, min_period, numStd):
    """
    Function to filter anomalous records based on `numStd` number of deviations away from rolling Median
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    window: moving window size for the rolling Median and stddev
    min_period: Minimum number of observations in window required to have a value (otherwise result is NAN - Not A Number)
    numStd: tolerance in number of standard deviations away from Median for record to be anomalous
    
    return:
    pandas dataframe containing anomalous records: 5 fields are in the list
            ['anomaly_type', 'anomaly_details', 'device','severity_score','when_captured']
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
            line1Upper = hv.Curve((x, upper), label='Median+ {}.stdev'.format(numStd)).opts(width=700, color='blue')
            line1Lower = hv.Curve((x, lower), label='Median- {}.stdev'.format(numStd)).opts(width=700, color='blue')

            overlay = line1Upper * line1Median * line1 * line1Lower
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies : records where deviation is >= num_std away from Median
            # there can be anomalies with the same timestamp because filtering is done on a per field basis
            temp = df[deviceFilter].copy()
            temp['rollingMedian'] = rollingMedian
            temp['rollingStdev'] = rollingStdev
            temp['severity_score'] = (temp[field]-temp['rollingMedian'])/(numStd*temp['rollingStdev'])
            temp = temp[abs(temp['severity_score']) >= 1]

            # instead of string for anomaly type, we can also make a key:value dictionary 
            # and use that as meta-data for reporting
            temp['anomaly_details'] = np.repeat(field + ' ' + str(numStd) + ' or more away from Median', temp.shape[0])
            anomaly_dfs.append(temp[['anomaly_details', 'device', 'severity_score', 'when_captured']])
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    anomaly_dfs['anomaly_type'] = 'Rolling median outlier'
    
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
    pandas dataframe containing anomalous records: 5 fields are in the list
            ['anomaly_type', 'anomaly_details', 'device','severity_score','when_captured']
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
            line1Upper = hv.Curve((x, upper), label='Roll. Stdev+ {}.stdev'.format(numStd)).opts(width=700, color='blue')
            line1Lower = hv.Curve((x, lower), label='Roll. Stdev- {}.stdev'.format(numStd)).opts(width=700, color='blue')

            overlay = line1Upper * line1Stdev * line1 * line1Lower
            overlay.opts(title="device: "+str(device))
            
            # return the list of anomalies : records where deviation is >= num_std away from Median
            temp = df[deviceFilter].copy()
            temp['rollingStdev'] = rollingStdev
            temp['rollingStdevStdev'] = rollingStdevStdev
            temp['severity_score'] = (temp[field]-temp['rollingStdev'])/(numStd*temp['rollingStdevStdev'])
            temp = temp[abs(temp['severity_score']) >= 1]

            # instead of string for anomaly type, we can also make a key:value dictionary 
            # and use that as meta-data for reporting
            temp['anomaly_details'] = np.repeat(field + ' ' + str(numStd) + ' or more away from local stdev', temp.shape[0])
            anomaly_dfs.append(temp[['anomaly_details', 'device', 'severity_score', 'when_captured']])
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    anomaly_dfs = pd.concat(anomaly_dfs, axis = 0)
    anomaly_dfs['anomaly_type'] = 'Rolling standard deviation outlier'
    
    return anomaly_dfs, overlays

def nightDayDisparity(fields, df, min_rec, day_start, day_end):
    """
    Function to filter anomalous records for nights where the median 
    nighttime field value > median daytime field value for the preceding day
    (1 anomaly per 24 hr period)
    
    args:
    fields:  list containing the string name(s) of the field(s) of interest for anomaly detection  
    Note these can only be AQ fields: 'pms_pm10_0', 'pms_pm01_0', 'pms_pm02_5'
    df: pandas dataframe containing 'device', fields, and 'when_captured'
    min_rec: minimum number of readings for there to be a defined (non-NAN) daytime or nightime median field value
    day_start: start hour of the day (like 6 for 0600hrs)
    day_end: end hour for the day (like 18 for 1800hrs)
    Note: daytime on a date is defined as the hours between `day_start` and `day_end`
    nighttime on a date is defined as the hours between `day_end` of the current_date and 
    the `day_start` of the next_date (if next date records are available in the records)
    
    return:
    pandas dataframe containing anomalous records: 5 fields are in the list
            ['anomaly_type', 'anomaly_details', 'device','severity_score','when_captured']
    overlays: holoviews plot objects with  data segregated by day, night and anomalous positions that need attention
    """
    anomaly_dfs = pd.DataFrame()
    overlays = []
    gf = df.groupby('device')
    # add the hour of day column to the data frame
    df['hod'] = pd.to_datetime(df['when_captured']).dt.hour
    df['date'] = pd.to_datetime(df['when_captured']).dt.date
        
    for device in tqdm(df['device'].unique(), mininterval = 30):
        deviceFilter = df['device'] == device
        overlay_batch = []
        df_dev = gf.get_group(device)
        sf = df_dev.groupby('date')
        dates = np.unique(df_dev['date'])
        # ensure dates are all in increasing order
        assert(np.all(np.diff(dates).astype('timedelta64[h]').astype('float')) >=0) 
        
        if any([field not in ['pms_pm01_0', 'pms_pm02_5', 'pms_pm10_0'] for field in fields]):
            print('This night-day disparity check does not apply to radiation fields:',
                  [x for x in fields if x not in ['pms_pm01_0', 'pms_pm02_5', 'pms_pm10_0']])
            fields = [x for x in fields if x in ['pms_pm01_0', 'pms_pm02_5', 'pms_pm10_0']]
        for field in fields:
            
            # dictionary of [day_median, night_median] value pair for the "field" on that date (with date as the key)
            medians_sf = {}     
            # Note: day and night medians are close together, differ by 1-2 or sometimes 0.
            # report dates and devices as anomalies when the date's night time median is larger than day time median 
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
            temp['severity_score'] = []
            for date, lst in medians_sf.items():
                if lst[1]>lst[0]: # nighttime PM > daytime PM
                	# the the first record to direct user to anomalous night
                    temp['when_captured'].append(sf.get_group(date).iloc[0].when_captured) 
                    # calculate the severity score based on the deviation -- definition here can be revised
                    temp['severity_score'].append((lst[1]-lst[0] + 1)/(lst[0]+1))
            temp['anomaly_details'] = np.repeat('nighttime median ' + field + ' > daytime median', len(temp['when_captured']))
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
            anomaly_dfs['anomaly_type'] = 'Night day disparity'
            overlay_batch.append(overlay)
        overlays.append(overlay_batch)
    
    return anomaly_dfs, overlays

def anomalyDetector(fields, df, anomaly_types=['negativeFields', 'rollingMedianDev', 'rollingStdevDev'], 
                    rmd_window=None, rmd_min_period=None, rmd_numStd=None,
                    rsd_window=None, rsd_min_period=None, rsd_numStd=None, 
                    ndd_min_rec=5, ndd_day_start=6, ndd_day_end=18, 
                    day_gap=2, devices=None):
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
    pandas dataframe containing anomalous records: 5 fields are in the list
            ['anomaly_type', 'anomaly_details', 'device', 'severity_score','when_captured']
    overlays: dict of holoviews plot objects 
    """
    anomaly_dfs = []
    overlays = {}
    
    if devices:
        print('Only analyzing devices:', devices)
        df = df[df['device'].isin(devices)].reset_index(drop = True)
    
    if 'negativeFields' in anomaly_types:
        print('Performing check for negative fields')
        anomaly_dfs.append(negativeFields(fields, df))
    if 'dataContinuityCheck' in anomaly_types:
        print('Performing check for data continuity')
        anomaly_dfs.append(dataContinuityCheck(df, day_gap))
    if 'PMorderingCheck' in anomaly_types:
        print('Performing check for PM ordering')
        anomaly_dfs.append(PMorderingCheck(df))
    if 'rollingMedianDev' in anomaly_types:
        print('Performing check for excessive variance in rolling median')
        anomaly_dfs_rmd, overlays_rmd = rollingMedianDev(fields, df, rmd_window, rmd_min_period, rmd_numStd)
        anomaly_dfs.append(anomaly_dfs_rmd)
        overlays['rollingMedianDev'] = overlays_rmd
    if 'rollingStdevDev' in anomaly_types:
        print('Performing check for excesssive variance in rolling standard deviation')
        anomaly_dfs_rsd, overlays_rsd = rollingStdevDev(fields, df, rsd_window, rsd_min_period, rsd_numStd)
        anomaly_dfs.append(anomaly_dfs_rsd)
        overlays['rollingStdevDev'] = overlays_rsd
    if 'nightDayDisparity' in anomaly_types:
        print('Perform check for night-day disparity')
        anomaly_dfs_ndd, overlays_ndd = nightDayDisparity(fields, df, ndd_min_rec, ndd_day_start, ndd_day_end)
        anomaly_dfs.append(anomaly_dfs_ndd)
        overlays['nightDayDisparity'] = overlays_ndd
    if len(anomaly_dfs) > 1:
        anomaly_dfs = pd.concat(anomaly_dfs, axis = 0).reset_index(drop = True)
        
    return anomaly_dfs, overlays

def normalize_scores(df):
    """
    Normalize severity scores from 0 to 1
    
    args:
    df: pandas dataframe containing 'severity_score' column to be normalized, as well as 'anomaly_type' and 'anomaly_details'
    
    return:
    df: pandas dataframe containing 'normalized_severity_score' column
    """
    df['normalized_severity_score'] = -1
    # Bins based on historical data
    bins_dict = pd.read_json('reference_data/historical_anomaly_thresholds.json').replace({'-np.inf':-np.inf, 
                                                                                           'np.inf':np.inf}).to_dict('list')
    
    # These anomalies have severity 1 on a strict basis
    auto_one_anomaly_types = ['Data continuity', 'PM ordering']
    df['normalized_severity_score'][df['anomaly_type'].isin(auto_one_anomaly_types)] = \
    df['severity_score'][df['anomaly_type'].isin(auto_one_anomaly_types)] # Should already be 1
    
    # These anomalies need to be measured by magnitude, not positive/negative
    abs_val_needed_cols = ['Rolling standard deviation outlier', 'Rolling median outlier']
    df['severity_score'][df['anomaly_type'].isin(abs_val_needed_cols)] = df['severity_score'][df['anomaly_type'].isin(abs_val_needed_cols)].abs()
    
    for anomaly_detail in df['anomaly_details'][~df['anomaly_type'].isin(auto_one_anomaly_types)].unique():
        print('Normalizing scores for', anomaly_detail)
        df['normalized_severity_score'][df['anomaly_details'] == anomaly_detail] = \
        pd.cut(df['severity_score'][df['anomaly_details'] == anomaly_detail], 
               bins=bins_dict[anomaly_detail], labels = [x/10 for x in range(1, 11)])
    
    return df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='*',
                        help='string of devices to check, separated by a space')
    parser.add_argument('--anomaly_types', nargs='*',
                        help='string of anomaly types to check, separated by a space')
    # Example command line input: python anomaly_detector.py --devices 1660294046 3373827677 --anomaly_types negativeFields rollingMedianDev
    args = parser.parse_args()
    
    try:
        df = pd.read_csv('cleaned_data/Solarcast-01_Main_Cleaned_Sorted.csv')
    except FileNotFoundError:
        print('cleaned_data/Solarcast-01_Main_Cleaned_Sorted.csv not found')
    
    if args.anomaly_types is not None:
        anomaly_dfs, _ = anomalyDetector(['pms_pm10_0', 'pms_pm02_5', 'pms_pm01_0', 'lnd_7318u', 'lnd_7318c'], df,
                                         anomaly_types=args.anomaly_types, devices=args.devices,
                                         rmd_window=200, rmd_min_period=50, rmd_numStd=3, rsd_window=200, rsd_min_period=50, rsd_numStd=3)
    else:
        anomaly_dfs, _ = anomalyDetector(['pms_pm10_0', 'pms_pm02_5', 'pms_pm01_0', 'lnd_7318u', 'lnd_7318c'], df, devices=args.devices,
                                         rmd_window=200, rmd_min_period=50, rmd_numStd=3, rsd_window=200, rsd_min_period=50, rsd_numStd=3)
    
    anomaly_dfs = normalize_scores(anomaly_dfs)
    anomaly_dfs.to_csv('final_anomaly_data/anomalies.csv', index = False)