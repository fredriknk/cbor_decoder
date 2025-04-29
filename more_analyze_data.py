import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



def train_and_test(X_train, y_train, X_test, y_test, model, modeltype, df,variables,
                   printcoeff=False, plot=True, polynomtrans=None):
    """
    Trains and tests a model, optionally printing coefficients
    and plotting predictions vs. true values.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    
    print("-----------------------------------------------------")
    print(f"Model Type: {modeltype} | "
          f"Training MSE: {train_mse:.3f} | Testing MSE: {test_mse:.3f} | "
          f"Training R²: {train_r2:.3f} | Testing R²: {test_r2:.3f}")
    
    # Optionally print coefficients if a polynomial or linear model is used
    if printcoeff and poly is not None and variables is not None:
        feature_names = poly.get_feature_names_out(variables)
        coefficients = model.coef_
        intercept = model.intercept_

        print("Model Intercept:", intercept)
        print("Model Coefficients:")
        for feature, coef in zip(feature_names, coefficients):
            print(f"{feature}: {coef}")


    if plot:
        plt.figure(figsize=(8, 6))
        # Plot the predictions vs the true values
        plt.scatter(y_train, y_train_pred, marker='.', alpha=0.1, edgecolors='blue', label='Predictions')
        plt.scatter(y_test, y_test_pred, marker='.', alpha=0.7, edgecolors='red', label='Predictions')
        # Plot a diagonal line for the "perfect prediction"
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{modeltype} Predictions vs. True Values, trained on {len(X_train)} samples')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if polynomtrans is not None:
            plt.plot(df["_time"], model.predict(poly.transform(df[variables])))
        else:
            plt.plot(df["_time"], model.predict(df[variables].values))
        #plt.plot(df["_time"],df["outlier_iso"]*-10)
        #plt.plot(df["_time"], df["humidity"].values)
        #plt.plot(df["_time"], df["temperature"].values)
        #plt.plot(df["_time"], df["gasResistance"].values)
        plt.plot(df["_time"], df["CH4"].values)
        plt.xlabel("Time")
        plt.ylabel("Prediction")
        plt.title("Model Prediction Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()




timediff = pd.Timedelta("0:00:00")
print(f"Timediff: {timediff}")
#load df_sensor from parquet file

#df_sens = pd.read_parquet("data/sensor_350457793812171.parquet")

df_sens = pd.read_parquet("data/sensor_350457793812080.parquet")
#df_sens = pd.read_parquet("data/sensor_350457793812262.parquet")

df_env = pd.read_parquet("data/env.parquet")

#print(df_sens.head())
#print(df_env.head())

# Make sure both are sorted by time
df_sens = df_sens.sort_values("_time")
df_env = df_env.sort_values("_time")

# Optional: drop columns like "result" or "table" from df_env
df_env = df_env.drop(columns=["result", "table"], errors="ignore")

df_env["_time"] = df_env["_time"] + pd.Timedelta(timediff)

# Merge based on closest past value of df_env for each df_sens row
df_sens_merged = pd.merge_asof(
    df_sens,
    df_env,
    on="_time",
    direction="nearest",  # or "nearest" / "forward"
    tolerance=pd.Timedelta("1m")  # optional: only join if within 10 minutes
)



# Calculate the differences from the last row

# Result is a DataFrame with df_env columns joined into df_sens
#print(df_sens_merged.head())

# Exclude the '_time' column from diff calculation

for col in df_sens_merged.columns:
    if col != '_time':
        df_sens_merged[f'diff_{col}'] = df_sens_merged[col].diff()

"""
from sklearn.ensemble import IsolationForest

#outlier cleaning:


# Drop any NaN rows for simplicity




iso = IsolationForest(contamination=0.05)  # You can tweak the contamination
numerical_data = df_clean[["diff_gasResistance", "diff_humidity", "diff_temperature"]].select_dtypes(include='number')
outlier_preds = iso.fit_predict(numerical_data)
df_clean['outlier_iso'] = outlier_preds  # -1 = outlier, 1 = inlier
"""


variables = ["temperature","humidity","gasResistance"]
             #"diff_gasResistance", "diff_humidity", "diff_temperature"]
df_clean = df_sens_merged.dropna(subset=variables + ["CH4"])
df_clean["_time"] = pd.to_datetime(df_clean["_time"])


#df_clean = df_clean[df_clean["_time"] > "2025-04-02 08:00:00"]
#df_clean = df_clean[df_clean["_time"] > "2025-04-09 06:00:00"]


def selection (df, variable, min_value, max_value):
    """
    Selects the specified variables from the DataFrame.
    """
    return df[(df[variable] > min_value) & (df[variable] < max_value)]

temp = 10
#df_test = selection(df_clean, "humidity", temp, temp+10)
df_train = df_clean
#Remove data where diff_CH4 is larger than 10
#df_test = df_test[df_test["diff_CH4"].abs() < 30]

print(f"Df after picking temp {df_train}, shape: {df_train.shape}")

if False:
    test = "temperature"
    #plot gas resistance as scatter plot vs CH4
    #color the dots based on humidity
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train["gasResistance"], df_train["CH4"], c=df_train[test], cmap='viridis', alpha=0.5)
    #Add a color bar
    plt.colorbar(label=test)
    plt.xlabel("Gas Resistance")
    plt.ylabel("CH4")
    plt.title("Gas Resistance vs CH4")
    plt.show()

if False:
    df_train = selection(df_clean, "humidity", 1, 90)
    #hexbin
    plt.figure(figsize=(10, 6)) 
    plt.hexbin(
        df_train["gasResistance"],
        df_train["CH4"],
        C=df_train["temperature"],  # Use temperature as the third variable
        reduce_C_function=np.mean,  # Average temperature in each bin
        gridsize=50,
        cmap='viridis',
        mincnt=1
    )
    plt.colorbar(label="Avg Temperature")
    plt.xlabel("Gas Resistance")
    plt.ylabel("CH4")
    plt.title("Avg Temperature in Gas Resistance vs CH4 Space")
    plt.show()

if False :
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # Extract values
    x = df_train["gasResistance"].values
    y = df_train["CH4"].values
    z = df_train["temperature"].values

    # Create a grid within the bounds of the data
    x_min, x_max = np.percentile(x, [1, 99])
    y_min, y_max = np.percentile(y, [1, 99])

    xi = np.linspace(x_min, x_max, 50)
    yi = np.linspace(y_min, y_max, 50)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    zi = griddata((x, y), z, (xi, yi), method='linear')  # use 'linear' for stability

    # Mask invalid (NaN) values
    masked_zi = np.ma.masked_invalid(zi)

    # Plot
    plt.figure(figsize=(6, 6))
    contour = plt.contourf(xi, yi, masked_zi, levels=100, cmap='viridis')
    plt.colorbar(contour, label="Avg Temperature")
    plt.xlabel("Gas Resistance")
    plt.ylabel("CH4")
    plt.title("Interpolated Avg Temperature in Gas Resistance vs CH4 Space")
    plt.tight_layout()
    plt.show()

#4d interpolatetion
if False:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator

    # Extract values
    resistance = df_train["gasResistance"].values
    temperature = df_train["temperature"].values
    humidity = df_train["humidity"].values
    ch4 = df_train["CH4"].values

    # Stack the independent variables into a single (N, 3) array
    points = np.column_stack((resistance, temperature, humidity))

    # Create the interpolator: input (resistance, temp, humidity), output CH4
    interp_func = LinearNDInterpolator(points, ch4)

    # Now you can interpolate CH4 at arbitrary points, for example:
    # Example interpolation at specific resistance, temperature, humidity values:
    example_resistance = 12616  # Replace with actual value
    example_temperature = 24  # Replace with actual value
    example_humidity = 23     # Replace with actual value

    interpolated_ch4 = interp_func(example_resistance, example_temperature, example_humidity)
    print(f"Interpolated CH4: {interpolated_ch4:.2f}")

    # To visualize the interpolation at fixed humidity:
    humidity_fixed = 23  # Example fixed humidity level
    res_range = np.linspace(np.percentile(resistance, 1), np.percentile(resistance, 99), 50)
    temp_range = np.linspace(np.percentile(temperature, 1), np.percentile(temperature, 99), 50)

    res_grid, temp_grid = np.meshgrid(res_range, temp_range)
    ch4_grid = interp_func(res_grid, temp_grid, humidity_fixed)

    # Mask invalid values for plotting
    masked_ch4 = np.ma.masked_invalid(ch4_grid)

    # Plot interpolated CH4 at fixed humidity
    plt.figure(figsize=(6, 6))
    contour = plt.contourf(res_grid, temp_grid, masked_ch4, levels=100, cmap='viridis')
    plt.colorbar(contour, label="Interpolated CH4")
    plt.xlabel("Gas Resistance")
    plt.ylabel("Temperature")
    plt.title(f"Interpolated CH4 at Humidity = {humidity_fixed}%")
    plt.tight_layout()
    plt.show()

if True:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import RBFInterpolator

    start = "2025-04-01 08:00:00"
    test_start ="2025-04-01 08:00:00"
    stop = "2025-04-23 02:00:00"

    df_train = df_clean[(df_clean["_time"] > test_start) & (df_clean["_time"] < stop)]

    # include data larger than test_start and smaller than start, or larger than stop
    df_test = df_clean[((df_clean["_time"] < test_start) & (df_clean["_time"] > start)) | (df_clean["_time"] > stop)]

    # Extract training data
    res_train = df_train["gasResistance"].values
    temp_train = df_train["temperature"].values
    hum_train = df_train["humidity"].values
    ch4_train = df_train["CH4"].values
    # Prepare input points for training
    X_train = np.column_stack((res_train, temp_train, hum_train))

    # Train RBF interpolator
    interp_rbf = RBFInterpolator(X_train, ch4_train, neighbors=100, smoothing=0.1)
    #
    #Resample the data to 1 minute intervals
    #df_test = df_test.set_index("_time").resample("1T").mean().reset_index()


    # Extract test data
    res_test = df_test["gasResistance"].values
    temp_test = df_test["temperature"].values
    hum_test = df_test["humidity"].values
    
    ch4_test_actual = df_test["CH4"].values

    # Prepare test input points for prediction
    X_test = np.column_stack((res_test, temp_test, hum_test))

    # Predict CH4 using the trained interpolator
    ch4_test_predicted = interp_rbf(X_test)

    # Plot predicted vs actual CH4
    plt.figure(figsize=(12, 6))
    plt.plot(df_test["_time"], ch4_test_actual, label="Measured CH4", color="blue")
    plt.plot(df_test["_time"], ch4_test_predicted, label="Interpolated CH4", color="red", linestyle="--", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("CH4 Concentration")
    plt.title("Comparison of Actual vs Interpolated CH4 Concentrations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()