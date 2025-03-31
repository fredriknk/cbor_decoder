from influxdb_client import InfluxDBClient
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime as dt
load_dotenv()

# InfluxDB credentials
url = os.getenv("HOST")
token = os.getenv("TOKEN")
org = os.getenv("ORG")

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
#start = "2025-03-26T14:00:00Z"
start = "2025-03-25T13:00:00Z"
stop = dt.now().strftime("%Y-%m-%dT%H:%M:%SZ")

envchamberQuery = f'''
from(bucket: "iot-bucket")
  |> range(start: {start}, stop:{stop})
  |> filter(fn: (r) => r["_measurement"] == "env_chamber" or r["_measurement"] == "environment")
  |> filter(fn: (r) => r["_field"] == "CH4"
       or r["_field"] == "CO2"
       or r["_field"] == "H20"
       or r["_field"] == "N20"
       or r["_field"] == "NH4"
       or r["_field"] == "Temp_chamber"
  )
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> group()  
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "mean")
'''
print(envchamberQuery)
sensorQuery = f"""
from(bucket: "iot-bucket")
  |> range(start: {start}, stop:{stop})
  |> filter(fn: (r) => r["_measurement"] == "environment")
  |> filter(fn: (r) => r["_field"] == "gasResistance" 
        or r["_field"] == "humidity"
        or r["_field"] == "pressure"
        or r["_field"] == "temperature")
  //|> aggregateWindow(every: 10m, fn: mean, createEmpty: false)
  |> yield(name: "mean")
"""
filename = "data/env.parquet"
start = dt.now()
dfenv = query_api.query_data_frame(envchamberQuery)
print(f"Query executed in {dt.now() - start} seconds")
print(dfenv.head())
dfenv.to_parquet(filename, compression="snappy", index=False)


# Assume you removed pivot or only pivoted on _field, not imei
df_all = query_api.query_data_frame(sensorQuery)

# A dictionary to hold dataframes keyed by their IMEI
dfs_by_imei = {}

# Get all unique sensor IMEIs present in the DataFrame
unique_imeis = df_all["imei"].unique()

for imei in unique_imeis:
    # Filter out just this sensor’s rows
    df_sensor = df_all[df_all["imei"] == imei].copy()
    
    # Pivot so each measurement field (e.g., “temperature”) becomes its own column
    df_pivoted = df_sensor.pivot(index="_time", columns="_field", values="_value")
    
    # Optional: remove the multi-index on columns if you pivot with more grouping keys
    df_pivoted.reset_index(inplace=True)
    
    # Save to dictionary, e.g. { "350457793812262": <DataFrame>, ... }
    dfs_by_imei[imei] = df_pivoted
    print(df_pivoted.head())
    # Save each sensor's data to a unique Parquet file
    filename = f"data/sensor_{imei}.parquet"

    df_pivoted.to_parquet(filename, compression="snappy", index=False)

    print(f"Saved sensor {imei} data to {filename}")

print(f"Query executed in {dt.now() - start} seconds")

