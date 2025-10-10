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


def predict_all(df_train, df_wild, model, model_name, variables, poly=None, axes=None, filename=None):
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
    
    fig, axes = init_plot() if axes is None else (None, axes)
    
    # 0) train ch4
    #axes[0].plot(df_train['_time'], df_train['CH4'],         label='True CH4', alpha=0.5)
    axes[0].plot(df_train['_time'], pred_train,              label=filename)
    axes[0].legend()
    axes[0].set_title(f'{model_name} Training Data')
    
    # 1) CH4
    axes[1].plot(df_wild['_time'],  pred_wild,               label=filename)
    axes[1].set_ylabel('CH4')
    axes[1].legend()
    axes[1].set_title(f'{model_name} Predictions')


     # 5) Gas Resistance
    #axes[2].plot(df_train['_time'], df_train['gasResistance'], label='Gas Resistance')
    axes[2].plot(df_wild['_time'],  df_wild['gasResistance'],  label=filename)
    axes[2].set_ylabel('Gas Resistance')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    if fig is not None:
        show_plots()
        
    return pred_train, pred_wild
    
def init_plot():
    # Create subplots: CH4 preds + each variable
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    return fig, axes

def show_plots():
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

def plot_stdev(axes, stdev_df,nr=None):
    # 3) Mean / Std across sensors per bin (skip NaNs)
    pred_mean = stdev_df.mean(axis=1, skipna=True)
    pred_std  = stdev_df.std(axis=1,  ddof=1, skipna=True)
    pred_n    = stdev_df.count(axis=1)

    # (Optional) keep bins with at least k sensors contributing
    k = max(2, len(train_series) // 2)   # e.g., at least half
    mask = pred_n >= k
    pred_mean = pred_mean[mask]
    pred_std  = pred_std[mask]
    pred_n    = pred_n[mask]

    # 4) Plot on your existing predictions axis
    axes[nr].plot(pred_mean.index, pred_mean.values, linewidth=2.5, label='Ensemble mean (10-min)')
    axes[nr].fill_between(pred_mean.index,
                        (pred_mean - pred_std).values,
                        (pred_mean + pred_std).values,
                        alpha=0.2, label='±1σ (10-min bins)')
    # --- Average std dev annotation ---
    avg_std = float(pred_std.mean())                          # simple mean across bins
    med_std = float(pred_std.median())                        # median, often more robust
    # (optional) sensor-count–weighted average σ:
    wavg_std = float((pred_std * pred_n).sum() / pred_n.sum())

    # Put a small textbox on the plot
    axes[nr].text(
        0.99, 0.02,
        f"Avg σ: {avg_std:.3g}\nMed σ: {med_std:.3g}\nW-Avg σ: {wavg_std:.3g}",
        transform=axes[1].transAxes,
        ha='right', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.8),
    )
    axes[nr].legend()




if __name__ == '__main__':
    sensor_file = 'data/sensor_350457793812262.parquet'
    #sensor_file = "data/sensor_350457793812171.parquet"
    #sensor_file = "data/sensor_350457793812080.parquet"
    
    all_sensor_files = [
        #'data/sensor_350457791624164.parquet', #tester ny
        #'data/sensor_350457791624024.parquet', #tester ny
        #'data/sensor_350457793812171.parquet', #tester gammel
        
        'data/sensor_350457791624008.parquet', #rød nedre
        'data/sensor_350457791624099.parquet', #rød midt
        'data/sensor_350457791624149.parquet', #rød øvre
        
        'data/sensor_350457791624156.parquet', #gul nedre
        'data/sensor_350457791624248.parquet', #gul midt
        'data/sensor_350457791624388.parquet', #gul øvre

        'data/sensor_350457793812080.parquet', #gml lilla øverst        
        'data/sensor_350457793812262.parquet', #gml orange nederst
        ]

    train_series = []  # collect one Series per sensor (index = _time)
    wild_series= []  # collect one Series per sensor (index = _time)
    i = 0
    fig, axes = init_plot()
    for sensor_file in all_sensor_files:
        sensor_name = sensor_file.split('/')[-1].split('.')[0]  
        print(f"Processing {sensor_file}...")
        train_start = pd.Timestamp('2025-07-30 00:00:00')
        train_end = pd.Timestamp('2025-08-07 12:00:00')
        test_start = pd.Timestamp('2025-08-21 12:00:00')
        #test_end = pd.Timestamp('2025-08-14 23:59:59')
        degrees = 4  
        vars_orig = ['temperature', 'gasResistance', 'humidity']

        env_file    = 'data/env.parquet'

        # Load and parse
        df_sens = pd.read_parquet(sensor_file)
        df_env  = pd.read_parquet(env_file)
        df_sens['_time'] = pd.to_datetime(df_sens['_time'])
        df_env['_time']  = pd.to_datetime(df_env['_time'])
        
        #drop all df sens gas_resistance < 0 or > 15000
        df_sens = df_sens[(df_sens['gasResistance'] > 0) & (df_sens['gasResistance'] < 15000)]

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


        # Labeled vs wild
        df_labeled = df_merged[(df_merged['_time'] < train_end) & (df_merged['_time'] > train_start) ].dropna(subset=['CH4']).copy()
        df_wild    = df_merged[df_merged['_time'] >= train_end].copy()

        # Fill predictors
        
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
        train_predict, wild_predict = predict_all(df_labeled, df_wild, pr, f'Polynomial Regression (deg={degrees}), parameters [{vars_orig}]', vars_orig, poly=poly,axes=axes, filename=sensor_name)
        
        freq = '10min'
        
        series_train = pd.Series(train_predict, index=pd.to_datetime(df_labeled['_time']), name=sensor_name)

        # resample THIS sensor to 10-minute bins
        series_train_10min = (
            series_train.sort_index()
            .resample(freq, label='left', closed='left', origin='start_day')
            .mean()
        )
        
        train_series.append(series_train_10min)
        
        series_wild = pd.Series(wild_predict, index=pd.to_datetime(df_wild['_time']), name=sensor_name)

        # resample THIS sensor to 10-minute bins
        series_wild_10min = (
            series_wild.sort_index()
            .resample(freq, label='left', closed='left', origin='start_day')
            .mean()
        )
        wild_series.append(series_wild_10min)

        # keep a simple list of Series, not lists
        

        imei = df_merged['imei'].iloc[0]

        # 1) Serialize to a canonical JSON string
        model_json = json.dumps(model_payload, sort_keys=True)

        # 2) Hash the JSON string
        model_id_hex = hashlib.sha256(model_json.encode("utf-8")).hexdigest()

        # Convert hex to int
        model_id_int = int(model_id_hex[:16], 16)  # This is a 64-bit int

        # Convert to signed 64-bit int
        model_id_int = to_signed_64bit(model_id_int)
        
        """
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
        """
    

    
    train_df = pd.concat(train_series, axis=1)  # columns = sensors, index = 10-min bins
    train_df.index.name = '_time'
    plot_stdev(axes, train_df,nr=0)
    
    pred_df = pd.concat(wild_series, axis=1)  # columns = sensors, index = 10-min bins
    pred_df.index.name = '_time'
    pred_df = pred_df[pred_df.index >= test_start]
    plot_stdev(axes, pred_df,nr=1)
    
    show_plots()
    