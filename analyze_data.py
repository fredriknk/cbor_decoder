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



def train_and_test(X_train, y_train, X_test, y_test, model, modeltype, df,df_test,df_train, variables,
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
    print(f"Training MSE: {train_mse:.3f} | Testing MSE: {test_mse:.3f} | "
          f"Training R²: {train_r2:.3f} | Testing R²: {test_r2:.3f} |"
          f"Model Type: {modeltype} | ")
    
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
        """
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
        """

        if polynomtrans is not None:
            plt.plot(df_train["_time"], model.predict(poly.transform(df_train[variables])), label="Train")
            plt.plot(df_test["_time"], model.predict(poly.transform(df_test[variables])), label="Test")
        else:
            plt.plot(df_train["_time"], model.predict(df_train[variables].values), label="Train")
            plt.plot(df_test["_time"], model.predict(df_test[variables].values), label="Test")
        #plt.plot(df["_time"],df["outlier_iso"]*-10)
        #plt.plot(df["_time"], df["humidity"].values)
        plt.plot(df["_time"], df["temperature"].values, label="Temperature")
        #plt.plot(df["_time"], df["NH4"].values*1)
        plt.plot(df["_time"], 100*df["gasResistance"].values/10000, label="Gas Resistance")
        plt.plot(df["_time"], df["CH4"].values)
        plt.xlabel("Time")
        plt.ylabel("Prediction")
        plt.legend()
        plt.title("Model Prediction Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()




#timediff = pd.Timedelta("0:01:00")
plotting = True
for i in [0]:#range(0, 15):
    timediff = pd.Timedelta(f"0:0{i}:00")
    
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

    #resample df_sens_merged to 1 minute intervals
    #df_sens_merged = df_sens_merged.set_index("_time").resample("2T").mean().reset_index()




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


    variables = ["temperature","gasResistance","humidity"]#,
                #"diff_humidity", "diff_temperature"]
    df_clean = df_sens_merged.dropna(subset=variables + ["CH4"])
    df_clean["_time"] = pd.to_datetime(df_clean["_time"])



    #df_temp_hum = df_clean[21 > df_clean["temperature"] > 20]

    #print(f"Df after dropping NaN: {df_clean}")

    # Train-test split (recommended so you have a set for scoring)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    start = "2025-04-09 04:00:00"
    test_start ="2025-04-09 04:00:00"
    stop = "2025-04-26 14:00:00"

    #start = "2025-04-05 08:00:00"
    #test_start ="2025-04-05 08:00:00"
    #stop = "2025-04-28 13:00:00"


    df_train = df_clean[(df_clean["_time"] > test_start) & (df_clean["_time"] < stop)]

    # include data larger than test_start and smaller than start, or larger than stop
    df_test = df_clean[((df_clean["_time"] < test_start) & (df_clean["_time"] > start)) | (df_clean["_time"] > stop)]

    #print(f"Train set: {df_train.head()} Test set: {df_test.head()}")

    X_train=df_train[variables]
    y_train=df_train["CH4"]

    X_test=df_test[variables]
    y_test=df_test["CH4"]


    numeric_cols = df_clean.select_dtypes(include=[float, int]).columns
    corr_matrix = df_clean[variables + ["CH4"]].corr()

    print("Correlation matrix:")
    print(corr_matrix)
    """
    #Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    modeltype = "Linreg "
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables)


    # RIDGE
    model = Ridge(alpha=1.0)  # alpha is the regularization strength
    model.fit(X_train, y_train)
    modeltype = "Ridge "
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables)

    # LASSO
    model = Lasso(alpha=0.1)  # alpha is the regularization strength
    model.fit(X_train, y_train)
    modeltype = "Lasso "
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables)
    """
    #Polynomial Regression
    for degree in [5,6,7,8]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        modeltype = f"Poly Reg (degree={degree})"

        train_and_test( X_train_poly, y_train, X_test_poly, y_test, model, modeltype, df_clean,df_test,df_train,variables, polynomtrans=True,plot=plotting)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score



    model = RandomForestRegressor(n_estimators=50, max_depth=15,min_samples_split=15, random_state=42)
    model.fit(X_train, y_train)

    modeltype = "Random Forrest"
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables,plot=plotting)

    depths = [tree.get_depth() for tree in model.estimators_]
    print("Average tree depth:", sum(depths) / len(depths))
    print("Maximum tree depth:", max(depths))

    import pickle

    size_mB = len(pickle.dumps(model)) / 1024e3
    print(f"Random Forest size: {size_mB:.2f} mB")

    
    # pip install xgboost
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    modeltype = "XGBoost"
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables,plot=plotting)
    

    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1)  # rbf is non-linear
    model.fit(X_train, y_train)
    modeltype = "SVR model"
    train_and_test(X_train, y_train,X_test,y_test, model, modeltype,df_clean,df_test,df_train,variables,plot=plotting)
    """
