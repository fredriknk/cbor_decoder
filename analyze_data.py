import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

# InfluxDB credentials
URL   = os.getenv("HOST")
TOKEN = os.getenv("TOKEN")
ORG   = os.getenv("ORG")
BUCKET = os.getenv("BUCKET")

for name, val in [("URL",URL),("TOKEN",TOKEN),("ORG",ORG),("BUCKET",BUCKET)]:
    if not isinstance(val, str) or not val:
        raise RuntimeError(f"{name!r} must be a non-empty string, is {val!r}.")
    
def to_signed_64bit(n):
    n = n & ((1 << 64) - 1)
    return n if n < (1 << 63) else n - (1 << 64)

def train_and_validate(X_train, y_train, X_val, y_val, model, model_name,
                       df_train, df_val, variables, poly=None, plot=True):
    """
    Train on X_train/y_train, validate on X_val/y_val, print metrics, and plot.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse   = mean_squared_error(y_val,   y_val_pred)
    train_r2  = r2_score(y_train, y_train_pred)
    val_r2    = r2_score(y_val,   y_val_pred)

    print(f"--- {model_name} ---")
    print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Val   MSE: {val_mse:.4f}, R²: {val_r2:.4f}\n")

    if plot:
        plt.figure(figsize=(8, 6))
        # Scatter true vs predicted for validation
        plt.scatter(y_val, y_val_pred, alpha=0.6, edgecolors='k', label='Validation')
        # Plot diagonal ideal line
        min_val = min(y_val.min(), y_val_pred.min())
        max_val = max(y_val.max(), y_val_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        plt.xlabel('True CH4')
        plt.ylabel('Predicted CH4')
        plt.title(f'{model_name} Validation: True vs Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return model_name, train_mse, val_mse, train_r2, val_r2


def predict_all(df_train, df_wild, model, model_name, variables, poly=None):
    """Predict on wild data (after train cutoff) and plot CH4 predictions plus subplots of variables"""
    # Prepare data
    df_train = df_train.copy()
    df_wild  = df_wild.copy()
    df_train['_time'] = pd.to_datetime(df_train['_time'])
    df_wild['_time']  = pd.to_datetime(df_wild['_time'])

    if poly is not None:
        X_train = poly.transform(df_train[variables])
        X_wild  = poly.transform(df_wild[variables])
    else:
        X_train = df_train[variables]
        X_wild  = df_wild[variables]

    pred_train = model.predict(X_train)
    pred_wild  = model.predict(X_wild)

    # Create subplots: CH4 preds + each variable
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    # 1) CH4
    axes[0].plot(df_train['_time'], df_train['CH4'],         label='True CH4', alpha=0.5)
    axes[0].plot(df_train['_time'], pred_train,              label='Train Pred CH4')
    axes[0].plot(df_wild['_time'],  pred_wild,               label='Wild Pred CH4')
    axes[0].set_ylabel('CH4')
    axes[0].legend()
    axes[0].set_title(f'{model_name} Predictions')

    # 2) Pressure
    axes[1].plot(df_train['_time'], df_train['pressure'],    label='Pressure')
    axes[1].plot(df_wild['_time'],  df_wild['pressure'],     label='Pressure (wild)', linestyle='--')
    axes[1].set_ylabel('Pressure')
    axes[1].legend()

    # 3) Humidity
    axes[2].plot(df_train['_time'], df_train['humidity'],    label='Humidity')
    axes[2].plot(df_wild['_time'],  df_wild['humidity'],     label='Humidity (wild)', linestyle='--')
    axes[2].set_ylabel('Humidity')
    axes[2].legend()

    # 4) Temperature
    axes[3].plot(df_train['_time'], df_train['temperature'], label='Temperature')
    axes[3].plot(df_wild['_time'],  df_wild['temperature'],  label='Temperature (wild)', linestyle='--')
    axes[3].set_ylabel('Temperature')
    axes[3].set_xlabel('Time')
    axes[3].legend()

     # 5) Gas Resistance
    axes[4].plot(df_train['_time'], df_train['gasResistance'], label='Gas Resistance')
    axes[4].plot(df_wild['_time'],  df_wild['gasResistance'],  label='Gas Resistance (wild)', linestyle='--')
    axes[4].set_ylabel('Gas Resistance')
    axes[4].set_xlabel('Time')
    axes[4].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sensor_file = 'data/sensor_350457793812262.parquet'
    #sensor_file = "data/sensor_350457793812171.parquet"
    #sensor_file = "data/sensor_350457793812080.parquet"

    env_file    = 'data/env.parquet'

    # Load and parse
    df_sens = pd.read_parquet(sensor_file)
    df_env  = pd.read_parquet(env_file)
    df_sens['_time'] = pd.to_datetime(df_sens['_time'])
    df_env['_time']  = pd.to_datetime(df_env['_time'])

    # Strip tz if present
    if df_sens['_time'].dt.tz is not None:
        df_sens['_time'] = df_sens['_time'].dt.tz_convert('UTC').dt.tz_localize(None)
    if df_env['_time'].dt.tz is not None:
        df_env['_time']  = df_env['_time'].dt.tz_convert('UTC').dt.tz_localize(None)

    # Sort
    df_sens.sort_values('_time', inplace=True)
    df_env.sort_values('_time', inplace=True)

    # Merge on nearest within 2m
    df_merged = pd.merge_asof(
        df_sens, df_env, on='_time', direction='nearest', tolerance=pd.Timedelta('2m')
    )

    # drop rows with extreme outliers
    df_merged = df_merged[(df_merged['gasResistance'] > 0) & (df_merged['gasResistance'] < 50000)]

    # Define cutoff
    train_start = pd.Timestamp('2025-04-10 00:00:00')
    train_end = pd.Timestamp('2025-04-30 12:00:00')

    # Labeled vs wild
    df_labeled = df_merged[(df_merged['_time'] < train_end) & (df_merged['_time'] > train_start) ].dropna(subset=['CH4']).copy()
    df_wild    = df_merged[df_merged['_time'] >= train_end].copy()

    # Fill predictors
    vars_orig = ['temperature', 'gasResistance', 'humidity']
    df_labeled[vars_orig] = df_labeled[vars_orig].fillna(method='ffill')
    df_wild[vars_orig]    = df_wild[vars_orig].fillna(method='ffill')

    # Train/validation split
    df_train, df_val = train_test_split(df_labeled, test_size=0.1,
                                        random_state=42, shuffle=True)
    X_train = df_train[vars_orig]; y_train = df_train['CH4']
    X_val   = df_val[vars_orig];   y_val   = df_val['CH4']

    # Fit models
    print('Correlation matrix:')
    print(df_train[vars_orig + ['CH4']].corr(), '\n')
    degrees = 4
    # Polynomial Regression
    poly = PolynomialFeatures(degree=degrees, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_val_p   = poly.transform(X_val)
    pr = LinearRegression(); pr.fit(X_train_p, y_train)

    # 1) Get the expanded feature names
    #    If X_train is a DataFrame, supply its column names; otherwise use a list of strings.
    import json

    model_payload = {
        "intercept": pr.intercept_.item(),
        "coefs":    pr.coef_.tolist(),
        "powers":   poly.powers_.tolist(),
        "feature_names": poly.get_feature_names_out(X_train.columns).tolist()
    }
    print(f"Model intercept: {model_payload}")

    with open("poly_model.json","w") as f:
        json.dump(model_payload, f)
    # Build the model payload
    
    model_name, train_mse, val_mse, train_r2, val_r2=train_and_validate(X_train_p, y_train, X_val_p, y_val, pr,
                       f'Polynomial Regression (deg={degrees})', df_train, df_val, vars_orig, poly=poly,plot=False)
    predict_all(df_labeled, df_wild, pr, f'Polynomial Regression (deg={degrees})', vars_orig, poly=poly)
    
    imei = df_merged['imei'].iloc[0]

    # 1) Serialize to a canonical JSON string
    model_json = json.dumps(model_payload, sort_keys=True)

    # 2) Hash the JSON string
    model_id_hex = hashlib.sha256(model_json.encode("utf-8")).hexdigest()

    # Convert hex to int
    model_id_int = int(model_id_hex[:16], 16)  # This is a 64-bit int

    # Convert to signed 64-bit int
    model_id_int = to_signed_64bit(model_id_int)
    

    print(f"Do you want to push the model for imei{imei} to InfluxDB? (y/n) -- HASH {model_id_int}")
    push_model = input().strip().lower()
    if push_model == 'y':
        # Gather metadata
        n_points   = X_train.shape[0]
        # If your training dates live in a datetime index or column:
        start_date = pd.to_datetime(df_labeled['_time']).min().isoformat()
        end_date   = pd.to_datetime(df_labeled['_time']).max().isoformat()

        # Push the model to InfluxDB
        from influxdb_client import InfluxDBClient, Point, WriteOptions
        import datetime
        

        # Connect
        client    = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        write_api = client.write_api(write_options=WriteOptions(batch_size=1))

        
        # Build the point with extra fields
        point = (
            Point("models")
            .tag("imei", str(imei))
            .field("modelJSON", json.dumps(model_payload))
            .field("num_points",   n_points)
            .field("start_date",   start_date)
            .field("end_date",     end_date)
            .field("model_name",   model_name)
            .field("train_mse",    train_mse)
            .field("val_mse",      val_mse)
            .field("train_r2",     train_r2)
            .field("val_r2",       val_r2)
            .field("hash_int",         model_id_int )
        )
        # Write it
        write_api.write(bucket=BUCKET, record=point)
        print(f"Pushed model + metadata for IMEI {imei}")
    else:
        print("Model not pushed to InfluxDB.")
    