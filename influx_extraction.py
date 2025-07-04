from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
import warnings
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# 1) silence the pivot warning
warnings.simplefilter("ignore", MissingPivotFunction)

load_dotenv()

# InfluxDB credentials
url   = os.getenv("HOST")
token = os.getenv("TOKEN")
org   = os.getenv("ORG")

client    = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

def run_query(query: str) -> pd.DataFrame:
    """
    Run a Flux query and always return a single DataFrame.
    If the client returns a list of DataFrames, concatenate them.
    """
    result = query_api.query_data_frame(query)
    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        return pd.concat(result, ignore_index=True)
    return result

def get_start_from_parquet(path: str, time_col: str = "_time", default: str = "2025-04-01T08:00:00Z") -> str:
    """
    If `path` exists, load it, take max(time_col) + 1μs, and format as RFC3339.
    Otherwise return `default`.
    """
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df[time_col] = pd.to_datetime(df[time_col])
        max_ts = df[time_col].max()
        eps = max_ts + timedelta(microseconds=1)
        return eps.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        return default

# common parameters
stop = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
agr  = "30s"

# ——— 1) ENV chamber data ———
env_file   = "data/env.parquet"
start_env  = get_start_from_parquet(env_file)
env_query = f'''
from(bucket: "iot-bucket")
  |> range(start: {start_env}, stop: {stop})
  |> filter(fn: (r) => r["_measurement"] == "env_chamber" or r["_measurement"] == "environment")
  |> filter(fn: (r) => r["_field"] == "CH4"
       or r["_field"] == "CO2"
       or r["_field"] == "H20"
       or r["_field"] == "N20"
       or r["_field"] == "NH4"
       or r["_field"] == "Temp_chamber"
  )
  |> aggregateWindow(every: {agr}, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "mean")
'''
df_new_env = run_query(env_query)

if not df_new_env.empty:
    df_old_env = pd.read_parquet(env_file) if os.path.exists(env_file) else pd.DataFrame()
    df_env_all = pd.concat([df_old_env, df_new_env], ignore_index=True)
    df_env_all.drop_duplicates(subset=["_time"], inplace=True)
    df_env_all.sort_values("_time", inplace=True)
    df_env_all.to_parquet(env_file, compression="snappy", index=False)
    print(f"Appended {len(df_new_env)} new rows to {env_file}")
else:
    print("No new env_chamber data to append.")

# ——— 2) SENSOR data per IMEI ———
# Base query template (we’ll fill in start/stop/imei each time)
sensor_query_tpl = """
from(bucket: "iot-bucket")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r["_measurement"] == "environment")
  |> filter(fn: (r) => r["_field"] == "gasResistance" 
        or r["_field"] == "humidity"
        or r["_field"] == "pressure"
        or r["_field"] == "temperature")
  |> filter(fn: (r) => r["imei"] == "{imei}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "mean")
"""

# Discover which IMEIs have new data
# (we don’t care if this one returns multiple tables; run_query will flatten)
idx_file     = "data/sensor_index.parquet"
start_index  = get_start_from_parquet(idx_file)
discover_q   = f"""
from(bucket: "iot-bucket")
  |> range(start: {start_index}, stop: {stop})
  |> filter(fn: (r) => r["_measurement"] == "environment")
  |> keep(columns: ["imei"])
  |> distinct(column: "imei")
"""
df_imeis = run_query(discover_q)
unique_imeis = df_imeis["imei"].dropna().unique()

for imei in unique_imeis:
    path = f"data/sensor_{imei}.parquet"
    start_imei = get_start_from_parquet(path)
    q = sensor_query_tpl.format(start=start_imei, stop=stop, imei=imei)
    df_new = run_query(q)

    if df_new.empty:
        print(f"No new data for IMEI {imei}")
        continue

    df_old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["_time"], inplace=True)
    df_all.sort_values("_time", inplace=True)
    df_all.to_parquet(path, compression="snappy", index=False)
    print(f"Appended {len(df_new)} rows to {path}")

# Update the index file so next run “discover” starts from now
pd.DataFrame({
    "imei": unique_imeis,
    "_time": datetime.utcnow()
}).to_parquet(idx_file, index=False)
