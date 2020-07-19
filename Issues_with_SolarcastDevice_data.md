Data is retreived from Script written by Mat Schaffer here:  https://github.com/Safecast/ingest/pull/95
Link to downloaded Solarcast Data: https://drive.google.com/drive/folders/13tPy3ZxMDfLYYJkplxY6AH76TJ4zGOEz

# Known Issues with data
1. RH (`env_humid`) : humidity sensors have bugs, there are values <0 and >100. Replace these values by NaNs
2. Devices have more than one reading per time stamp -- why? Here's the data logging process (From Ray Ozzie at Safecast). Point (F) might explain the issue with using `service_uploaded` field -- but issue persists with `when_captured` field also
    1. the device takes a measurement of N sensors (its air sensors, its Geigers, etc) at time == T1.  It packages the captured time and sensor measure,ents into 1 composite JSON object stamped with T1 as “captured time”.
    2. the device stores this json object in flash memory
    3. the device takes another measurement of its sensors at time T2.
    4. the device stores the JSON object with T2 as captured time in flash memory
    5. eventually, after many such json objects are buffered, a “call” is made on cellular.  This “call” is called a “session”.
    6. all the buffered JSON objects are transferrred to the service in less than a second. As a side effect of this transfer, the server stamps each json object with “service uploaded” time.  So all of these json measurements captured over quite some time will all have this session time.
    7. the service forwards them to Ingest, where they are stored in the database.
    
    **Solution: For now, just discard those time-stamp readings. If there is more clarity frok Safecast about why this happens, then take the higheest (max) of the multiple readings for groupings agaings device & `when_captured` time stamp**
    
3. If there is over a week's lag between `service_uploaded` and `when_captured` then that data record should be discarded
4. When `when_captured` is 0, then it should be discarded.
5. Whereever you see NaNs in the dataset, they are actually missing values in csv. So that's how they should be treated. NaNs are an import artefact
6. PM of 0.0 does not mean invalid or erroneous reading. It just means very clean air. 
7. There are those data files that have duplicate records when (`device_urn`, `when_captured`) is used as the key. Be sure to filter duplicate records from datav
