from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
import warnings
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# 1) silence the pivot warning
#warnings.simplefilter("ignore", MissingPivotFunction)

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

start = "2025-05-01T08:00:00Z"  # Default start time if no parquet file exists
stop =  "2025-07-03T08:00:00Z"

cell_query = f'''
from(bucket: "iot-bucket")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => 
    r["_field"] == "CellID" 
    or r["_field"] == "PLMN" 
    or r["_field"] == "TrackingAreaCode"
    )
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "mean")
'''
#time the query
timenow = datetime.utcnow()
df_new_env = run_query(cell_query)
print(f"Query took {datetime.utcnow() - timenow} seconds")
# Convert _time to datetime
df_new_env["_time"] = pd.to_datetime(df_new_env["_time"], utc=True)
# Ensure the DataFrame is sorted by time
df_new_env = df_new_env.sort_values("_time")

df_unique = (
    df_new_env
      .drop_duplicates(
          subset=["CellID", "PLMN", "TrackingAreaCode"],
          keep="first"          # keep="last" if you’d prefer the most-recent row
      )
      .reset_index(drop=True)   # optional: tidy the index
)

print(f"After de-duping: {df_unique.shape} rows")
print(df_unique.head())

df_cells = df_unique.copy()  # keep the original intact

print("Distinct towers :", len(df_cells))      # ≈ 50
# ------------------------------------------------------------------
# 2) prep the numeric columns OpenCelliD expects
# ------------------------------------------------------------------
def hex2dec(h):        # works for strings like '0331C805' or ints
    return int(str(h), 16)

df_cells["cellid_dec"] = df_cells["CellID"].apply(hex2dec)
df_cells["tac_dec"]    = df_cells["TrackingAreaCode"].apply(hex2dec)
df_cells["mcc"]        = df_cells["PLMN"].str[:3].astype(int)
df_cells["mnc"]        = df_cells["PLMN"].str[3:].astype(int)

# ------------------------------------------------------------------
# 3) query /cell/get for every row and collect results
# ------------------------------------------------------------------
import requests, time, os
from tqdm import tqdm

API_KEY = os.getenv("OPENCELLID_KEY")          # put yours in .env
url      = "https://opencellid.org/cell/get"

def ocid_lookup(row):
    """
    Call OpenCelliD and return (lat, lon) or (None, None)
    """
    params = {
        "key":    API_KEY,
        "mcc":    row.mcc,
        "mnc":    row.mnc,
        "lac":    row.tac_dec,          # TAC goes into 'lac'
        "cellid": row.cellid_dec,       # decimal ECI / CID
        "radio":  "LTE",                # or omit → first match
        "format": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            j = r.json()
            # JSON returns {'stat':'ok','lat':…,'lon':…} on success
            if "lat" in j and "lon" in j:
                return j["lat"], j["lon"]
    except requests.RequestException as e:
        print("✖", e, params)
    return None, None

lats, lons = [], []
"""
for _, r in tqdm(df_cells.iterrows(), total=len(df_cells)):
    lat, lon = ocid_lookup(r)
    lats.append(lat); lons.append(lon)
    time.sleep(0.5)                    # polite: ~5 req/s < 1000/day :contentReference[oaicite:1]{index=1}

df_cells["lat"] = lats
df_cells["lon"] = lons


df_cells["tac_dec"]    = df_cells["TrackingAreaCode"].apply(hex2dec)
df_cells["mcc"]        = df_cells["PLMN"].str[:3].astype(int)
df_cells["mnc"]        = df_cells["PLMN"].str[3:].astype(int)
# ------------------------------------------------------------------
# 4) inspect the result
# ------------------------------------------------------------------
#Save dataframe to parquet
df_cells.to_parquet("data/cell_locations.parquet", index=False, compression="snappy")

print(df_cells[["imei", "CellID","cellid_dec","tac_dec","mcc","mnc","tac_dec","lat", "lon"]])
"""