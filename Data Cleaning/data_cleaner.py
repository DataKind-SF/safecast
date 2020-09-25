import numpy as np 
import pandas as pd
import datetime
import argparse

def readCSV(dt):
    """
        Read the CSV file into a dataframe for a YYYY-MM (dt)
        Do preliminary cleaning
        arg: dt -- string with format YYYY-MM
        return df: dataframe containing data from csv
    """
    folder = 'raw_data/'
    filename = 'output-' + str(dt) + '-01T00_00_00+00_00.csv'
    df = pd.read_csv(folder+filename)
    df.when_captured = pd.to_datetime(df.when_captured)

    # Need to change the format of the Time Stamp for all the measurements in the raw data
    df.service_uploaded =  df.service_uploaded.apply(lambda x: \
                            datetime.datetime.strptime(x, '%b %d, %Y @ %H:%M:%S.%f')\
                            .replace(tzinfo=datetime.timezone.utc))
    #### Add a column for the year
    df['year'] = pd.DatetimeIndex(df['when_captured']).year
    
    #### Need to correct for the format of the PM numeric values. 
    df['pms_pm01_0'] = df['pms_pm01_0'].astype(str).str.replace(',', '').astype(float)
    df['pms_pm10_0'] = df['pms_pm10_0'].astype(str).str.replace(',', '').astype(float)
    df['pms_pm02_5'] = df['pms_pm02_5'].astype(str).str.replace(',', '').astype(float)
    
    return df

def findBadData(df):
    
    temp_df = df.groupby(['device','when_captured']).size().to_frame('size').\
                                    reset_index().sort_values('size', ascending=False)
    print("bad device data counts: ")
    badRecords = temp_df[(temp_df['size']>1)]
    print(badRecords)
    
    print("all bad device list: ")
    # Devices that have misbehaved at some point - more than one data values per time stamp
    print(np.unique(temp_df[temp_df['size']>1]['device'].values)) # devices that have misbehaved
    
    return badRecords

def rmInvalidTimeStamps(df):
    """
    remove invalid time stamped records
    """
    
    ## remove records with NULL `when_captured`
    print("Null date records to remove: ", df['when_captured'].isna().sum())
    df = df[df['when_captured'].notna()]
    print("df shape after remove records with NULL `when_captured` : ",df.shape)

    ## remove records where `when_captured` is invalid
    boolean_condition = df['when_captured'] >  pd.to_datetime(2000/1/19, infer_datetime_format=True).tz_localize('UTC')
    print("Valid `when_captured`  entires: ", boolean_condition.sum())
    df = df[df['when_captured'] >  pd.to_datetime(2000/1/19, infer_datetime_format=True).tz_localize('UTC')]
    print("df shape after remove records where `when_captured` is an invalid : ",df.shape)

    ## remove records where gap of `service_uploaded` and `when_captured` > 7 days
    boolean_condition = abs(df['when_captured'].subtract(df['service_uploaded'])).astype('timedelta64[D]') < 7
    boolean_condition.shape
    print("Lag 7 days to remove: ",df.shape[0] - (boolean_condition).sum())
    df = df[boolean_condition]
    print("df shape after records where gap of `service_uploaded` and `when_captured` > 7 days : ",df.shape)
    
    return df

def imputeInaccurateRH(df):
    """ 
    impute data with NaN(missing) for inaccurate values of RH
    """
    
    boolean_condition = (df['env_humid']<0) | (df['env_humid']>100)
    column_name = 'env_humid'
    new_value = np.nan
    df.loc[boolean_condition, column_name] = new_value
    print("Inaccurate RH records imputed: ", boolean_condition.sum())
    
    return df

def dropServiceUploaded(df):
    """
    Inplace dropping of the 'service_uploaded' column
    """
    df.drop('service_uploaded', axis=1, inplace=True)
    
def rmDuplicates(df):
    """
    Inplace dropping of duplicates
    preserve a single copy of duplicative rows
    """
    incoming = df.shape[0]
    df.drop_duplicates(subset=df.columns[0:df.shape[1]], inplace=True, keep='first') # args: subset=[df.columns[0:df.shape[1]]], keep = 'first'
    print("Number of duplicative entries removed : ", -df.shape[0]+incoming)
    
def dataAggWithKey(df):
    """
    Aggregate the df based on key: 'device','when_captured'
    arg: df - incoming dataframe
    return: datframe with COUNTS and COUNT-DISTINCTS for each key
    """
    # STEP 1: Aggregate the dataframe based on key
    
    temp_df = df.groupby(['device','when_captured']).agg(['count','nunique'])
    # temp_df.info()
    num_groups = temp_df.shape[0]
    print("num_groups  is : ", num_groups)

    # STEP 2: Merge Counts and Count-Distincts to check for duplicative records and multiplicities

    even = list(range(0,26,2))
    odd = list(range(1,26,2))
    tmp_df1 = temp_df.iloc[:,even].max(axis=1).to_frame('COUNTS').reset_index()
    tmp_df2 = temp_df.iloc[:,odd].max(axis=1).to_frame('DISTINCTS').reset_index()
    print(tmp_df1.shape, tmp_df2.shape)
    merged = pd.merge(tmp_df1, tmp_df2, left_on = ['device', 'when_captured'], \
                      right_on=['device', 'when_captured'])
    return merged, num_groups

def identifyALLNanRecs(merged):
    """
        Actionable: Records of useless data with all NaNs
        args: incoming datframe with COUNTS and COUNT-DISTINCTS for each key
        return : keys dataframe ('device', 'when_captured') to remove later
    """
    bool1 = (merged.COUNTS >1) & (merged.DISTINCTS==0)
    sum1 = bool1.sum()
    print(sum1)
    toDiscard1 = merged.loc[:,['device', 'when_captured']][bool1]
    toDiscard1.shape
    return sum1, toDiscard1

def identifyMultivaluedTimeStamps(merged):
    """
        Actionable: Records that are a mix of duplicates and non-duplicate rows 
        for a given (`device`, `when_captured`) [must be all discarded]
        args: incoming datframe with COUNTS and COUNT-DISTINCTS for each key
        return : keys dataframe ('device', 'when_captured') to remove later
    """
    bool3 = (merged.COUNTS >1) & (merged.DISTINCTS>1)
    sum3 = bool3.sum()
    print(sum3)
    toDiscard3 = merged.loc[:,['device', 'when_captured']][bool3]
    toDiscard3.shape
    return sum3, toDiscard3

def identifyRemainingDupl(merged):
    """
        Actionable: even though duplicates were dropped, there can still be records for which   (merged.COUNTS >1) & (merged.DISTINCTS==1)
        : consider the case where one of the records for the key under consideration has meaningful values
        : but the other record has all NaNs for the same key. Ex. (Oct 18, 2018 @ 10:36:24.000 , 2299238163): row 22618
        Records where all rows are purely duplicates [preserve only 1 later]
        args: incoming datframe with COUNTS and COUNT-DISTINCTS for each key
    """
    bool2 = (merged.COUNTS >1) & (merged.DISTINCTS==1)
    sum2 = bool2.sum()
    print("remaining duplicates check : " ,merged.COUNTS[bool2].sum() - merged.DISTINCTS[bool2].sum())
    toDiscard2 = merged.loc[:,['device', 'when_captured']][bool2]
    toDiscard2.shape
    return sum2, toDiscard2

def goodTimeStamps(merged):
    """
        Records that are good
    """
    bool4 = (merged.COUNTS ==1) & (merged.DISTINCTS==1)
    sum4 = bool4.sum()
    print('good records : ', sum4)
    return sum4

def writeDF(dt, dframe, descrpt):
    """
        write multivalued timestamps' keys to a csv
        args: dframe to write
        descrpt: string with descripttion to append to file
    """
    # dframe.info()
    print("written records shape : ", dframe.shape)
    dframe.to_csv(str(dt) + '-01_' + str(descrpt) + '.csv')
    
def filterRows(toDiscard1, toDiscard2, toDiscard3, df):
    """
        Inplace discarding of rows based on allNaN record keys (in df : toDiscard1)
        and rows based on MultivaluedTimeStamps keys (in df : toDiscard3)
        from original dataframe: df
        args:
            toDiscard1: allNaN record keys
            toDiscard3: MultivaluedTimeStamps keys
            df: original dataframe
    """
    # STEP 1 : 
    # all tuples of keys to be discarded
<<<<<<< HEAD
    discard = pd.concat([toDiscard1, toDiscard3], ignore_index=True)
=======
    discard = pd.concat([toDiscard1, toDiscard2, toDiscard3], ignore_index=True)
>>>>>>> data_cleaning_script
    discard['KEY_Dev_WhenCapt'] = list(zip(discard.device, discard.when_captured))
    print(df.shape, discard.shape)

    # STEP 2 :
    # tuples of all keys in the dataframe
    df['KEY_Dev_WhenCapt'] = list(zip(df.device, df.when_captured))
    df.shape

    # STEP 3 : 
    # discard the rows
    rows_to_discard = df['KEY_Dev_WhenCapt'].isin(discard['KEY_Dev_WhenCapt'])
    print("these many rows to discard: ", rows_to_discard.sum())

    incoming = df.shape[0]
    df = df[~rows_to_discard]
    print(incoming - df.shape[0])
    
    return df

def cleanSolarCastData(dt):
    """
<<<<<<< HEAD
        Function to clean all the data with the helper functions above
=======
        Master Function to clean all the data with the helper functions in `Data_Cleansing_Single_file`
>>>>>>> data_cleaning_script
        arg: dt: The function returns the cleaned data frame for the YYYY-MM corresponding to "dt"
        return : df: cleaned dataframe
    """

    df = readCSV(dt)
    findBadData(df)

    df = rmInvalidTimeStamps(df)
    print("new df: ", df.shape)

    df = imputeInaccurateRH(df)
    print("new df: ", df.shape)

    dropServiceUploaded(df)
    print("new df after dropping service_uploaded col: ", df.shape)

    rmDuplicates(df)
    print("new df after removing duplicates: ", df.shape)

    merged,num_groups = dataAggWithKey(df)
    print("merged: ", merged.shape)
    print("num_groups : ", num_groups)

    sum1, toDiscard1 = identifyALLNanRecs(merged)
    sum3, toDiscard3 = identifyMultivaluedTimeStamps(merged)
    sum2, toDiscard2 = identifyRemainingDupl(merged)
    sum4 = goodTimeStamps(merged)
    print("toDiscard1 shape: ",toDiscard1.shape)
    print("toDiscard2 shape: ",toDiscard2.shape)
    print("toDiscard3 shape: ",toDiscard3.shape)

    # sanityCheck(): ensure you have all records covered by 1 of the 4 conditions
    assert(num_groups == sum1+sum2+sum3+sum4)

    writeDF(dt, toDiscard3, 'MultivaluedTimeStamps')

    df = filterRows(toDiscard1, toDiscard2, toDiscard3, df)
    print("final df shape: ", df.shape)

    ### Now check to make sure no garbage data is left

    badRecordsLeft = findBadData(df)
    if not badRecordsLeft.empty:
        print("Still bad records remaining:", badRecordsLeft)
        assert(badRecordsLeft.empty)

    return df

def cleanAndWriteDF(dt):
    
    
    df = cleanSolarCastData(dt)

    print(df.shape)

    # Check how many devices there are in the dataset
    devices = np.unique(df.device.values)
    print(len(devices))
    print(devices)

    # *Sort the time series -- it's unsorted.*
    df.sort_values(by=['when_captured'], inplace=True)

    # Write the files
    descrpt = 'cleaned'
    writeDF(dt, df, descrpt)
    
    return df
    
def readCleanedDF(dt, descrpt):
    """
        Read the cleaned & pre-sorted CSV file into a dataframe for a YYYY-MM (dt)
        Do preliminary cleaning
        arg: dt -- string with format YYYY-MM
        return df: dataframe containing data from csv
    """
    folder = './'
    filename = str(dt) + '-01_' + str(descrpt) + '.csv'
    df = pd.read_csv(folder+filename)
    return df

def cleanAndWriteMainDF(start_yyyymm, end_yyyymm):
    """
        Cleans each month's data and saves it; also concatenate all the data into a single DataFrame,
        sort, and then save
        arg: start_yyyymm -- string with format YYYY-MM; earliest month for which data is available
             end_yyyymm -- string with format YYYY-MM; latest month for which data is available
    """
    dfList = []
    for dt in pd.date_range(start_yyyymm, end_yyyymm, freq='MS').strftime("%Y-%m").tolist():
        print("========================")
        print("========================", dt, "========================")
        print("========================")
        df = cleanAndWriteDF(dt)
        dfList.append(df)
    
    mainDF = pd.concat(dfList, ignore_index=True)
    mainDF.when_captured = pd.to_datetime(mainDF.when_captured)
    mainDF.sort_values(by=['when_captured'], inplace=True)
    mainDF['lnd_7318u'] = mainDF['lnd_7318u'].astype(str).str.replace(',', '').astype(float)
    
    writeDF('Solarcast', mainDF, 'Main_Cleaned_Sorted')

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('start_yyyymm', 
                        help='string with format YYYY-MM; earliest month for which data is available')
    parser.add_argument('end_yyyymm',
                        help='string with format YYYY-MM; latest month for which data is available')
    args = parser.parse_args()
    
    cleanAndWriteMainDF(args.start_yyyymm, args.end_yyyymm)