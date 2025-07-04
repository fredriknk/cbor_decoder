import cbor2
import hashlib
from hashlib import sha256
import uuid
import json

cbor_data = bytes([191, 97, 100, 167, 98, 85, 85, 80, 62, 146, 111, 117, 211, 27, 72, 82, 179, 148, 140, 19, 245, 168, 187, 110, 97, 67, 25, 15, 185, 98, 82, 83, 24, 35, 98, 83, 78, 24, 25, 98, 80, 76, 101, 50, 52, 50, 48, 50, 98, 84, 65, 100, 48, 65, 70, 50, 98, 67, 73, 104, 48, 50, 49, 52, 48, 66, 49, 54, 255])   
cbor_data = bytes([162, 97, 100, 162, 98, 73, 77, 111, 51, 53, 48, 52, 53, 55, 55, 57, 49, 54, 50, 52, 51, 56, 56, 97, 67, 25, 11, 27, 97, 101, 129, 162, 97, 67, 0, 97, 77, 120, 30, 84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 32, 101, 114, 114, 111, 114, 32, 108, 111, 103, 32, 101, 110, 116, 114, 121])
#cbor_data = bytes([161,97,100,167,98,73,77,111,51,53,48,52,53,55,55,57,49,54,50,52,51,56,56,97,67,25,11,25,98,82,83,24,44,98,83,78,24,41,98,80,76,101,50,52,50,48,49,98,84,65,100,56,49,65,69,98,67,73,104,48,51,51,49,67,56,48,53,162,97,100,162,98,73,77,111,51,53,48,52,53,55,55,57,49,54,50,52,51,56,56,97,67,25,11,27,97,101,129,162,97,67,0,97,77,120,30,84,104,105,115,32,105,115,32,97,32,116,101,115,116,32,101,114,114,111,114,32,108,111,103,32,101,110,116,114,121])
decoded_data = cbor2.loads(cbor_data)
print(decoded_data)
mapped_data = {
    "d": "device_info",
    "IM": "IMEI",
    "UU": "UUID",
    "RS": "cell_RSSI",
    "SN": "SignalNoiseRatio",
    "PL": "PLMN",
    "TA": "TrackingAreaCode",
    "CI": "CellID",
    "C": "device_timestamp",
    "m": "measurement",
    "T": "temperature",
    "H": "humidity",
    "P": "pressure",
    "G": "gasResistance",
    "L": "latitude",
    "O": "longitude",
    "b": "battery_info",
    "V": "batteryVoltage",
    "A": "chargingCurrent",
    "t": "batteryTemp",
    "S": "chargingStatus",
    "e": "errorMessages", 
    "M": "errorMessage",

}

def remap_keys(data, mapping):
    if isinstance(data, dict):
        return {mapping.get(k, k): remap_keys(v, mapping) for k, v in data.items()}
    elif isinstance(data, list):
        return [remap_keys(i, mapping) for i in data]
    else:
        return data

# Apply remapping
remapped_data = remap_keys(decoded_data, mapped_data)
# Convert to JSON
json_data = json.dumps(remapped_data, indent=4)
print(json_data)