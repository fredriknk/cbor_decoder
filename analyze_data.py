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

def train_and_test(X_train, y_train, X_test, y_test, model, modeltype,
                   printcoeff=False, plot=True, poly=None, variables=None):
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
        plt.scatter(y_test, y_test_pred, marker='o', alpha=0.7, edgecolors='black', label='Predictions')
        # Plot a diagonal line for the "perfect prediction"
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{modeltype} Predictions vs. True Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("-----------------------------------------------------")
    print(f"Model Type: {modeltype} | "
          f"Training MSE: {train_mse:.3f} | Testing MSE: {test_mse:.3f} | "
          f"Training R²: {train_r2:.3f} | Testing R²: {test_r2:.3f}")


#df_sens = pd.read_parquet("data/sensor_350457793812080.parquet")

#load df_sensor from parquet file
df_sens = pd.read_parquet("data/sensor_350457793812262.parquet")
#df_sens = pd.read_parquet("data/sensor_350457793812171.parquet")


df_env = pd.read_parquet("data/env.parquet")

#print(df_sens.head())
#print(df_env.head())

# Make sure both are sorted by time
df_sens = df_sens.sort_values("_time")
df_env = df_env.sort_values("_time")

# Optional: drop columns like "result" or "table" from df_env
df_env = df_env.drop(columns=["result", "table"], errors="ignore")

# Merge based on closest past value of df_env for each df_sens row
df_sens_merged = pd.merge_asof(
    df_sens,
    df_env,
    on="_time",
    direction="nearest",  # or "nearest" / "forward"
    tolerance=pd.Timedelta("2m")  # optional: only join if within 10 minutes
)

# Result is a DataFrame with df_env columns joined into df_sens
print(df_sens_merged.head())

# Drop any NaN rows for simplicity
variables = ["humidity","temperature", "gasResistance"]#, "H2O"]

df_clean = df_sens_merged.dropna(subset=variables + ["CH4"])

print(f"Df after dropping NaN: {df_clean}")
X = df_clean[variables]
y = df_clean["CH4"]    # target

# Train-test split (recommended so you have a set for scoring)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#plotting
#sns.pairplot(df_clean)
#plt.show()

print(df_clean.head())


numeric_cols = df_clean.select_dtypes(include=[float, int]).columns
corr_matrix = df_clean[numeric_cols].corr()

print("Correlation matrix:")
print(corr_matrix)

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
modeltype = "Linreg "
train_and_test(X_train, y_train,X_test,y_test, model, modeltype)


# RIDGE
model = Ridge(alpha=1.0)  # alpha is the regularization strength
model.fit(X_train, y_train)
modeltype = "Ridge "
train_and_test(X_train, y_train,X_test,y_test, model, modeltype)

# LASSO
model = Lasso(alpha=0.1)  # alpha is the regularization strength
model.fit(X_train, y_train)
modeltype = "Lasso "
train_and_test(X_train, y_train,X_test,y_test, model, modeltype)

#Polynomial Regression
for degree in range(1, 5):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    modeltype = f"Poly Reg (degree={degree})"

    train_and_test(X_train_poly, y_train, X_test_poly, y_test, model, modeltype)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

modeltype = "Random Forrest"
train_and_test(X_train, y_train,X_test,y_test, model, modeltype)





# pip install xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

modeltype = "XGBoost"
train_and_test(X_train, y_train,X_test,y_test, model, modeltype)


#from sklearn.svm import SVR
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#model = SVR(kernel="rbf", C=1.0, epsilon=0.1)  # rbf is non-linear
#model.fit(X_train, y_train)
#modeltype = "SVR model"
#train_and_test(X_train, y_train,X_test,y_test, model, modeltype)

