# Solarcast Data Cleansing Protocol (Updated 07/20/20)

Points of contact : Ray Ozzie (backend dev with sensors) and Mat Schaffer (data engineer)
1. For records where env_humid < 0 or env_humid >100 -- flag them as "Invalid Relative Humidity" (and I will replace those values with NaNs on my end)
2. For records where when_captured or service_uploaded is datetime(0,0,0) -- flag them as "invalid time stamp"
3. For records where when_captured == service_uploaded - "invalid log Type 1 "
4. For records where date_diff(service_uploaded, when_captured)>7days  - "invalid log Type 2" (based in @ray's recommendation)
5. For all records with the same key: (device, when_captured)  where multiplicity of these records is >=2 AND all such records are identical in all columns (aka attributes/fields/payloads) -- flag them as "duplicate logs"
6. For all records with the same key: (device, when_captured)  where multiplicity of these records is >=2 AND NOT ALL such records are identical -- flag them as "duplicate records to discard"

N.B.: Implementation of (5) and (6) in SQL will be achieved by comparing COUNT  and COUNT DISTINCT after grouping by the key i mentioned above (edited)

N.B.: If you work with Python instead of SQL, then try using [this page](https://medium.com/jbennetcodes/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e) to get parallels between SQL methods
