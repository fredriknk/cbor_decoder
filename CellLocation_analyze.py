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

df_cells= pd.read_parquet("data/cell_locations.parquet")

print(df_cells.head())#[["imei", "CellID","cellid_dec","tac_dec","mcc","mnc","tac_dec","lat", "lon"]])
print(df_cells.keys())
