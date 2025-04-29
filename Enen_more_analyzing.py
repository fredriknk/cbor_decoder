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

# Merge based on closest past value of df_env for each df_sens row
df_sens_merged = pd.merge_asof(
    df_sens,
    df_env,
    on="_time",
    direction="nearest",  # or "nearest" / "forward"
    tolerance=pd.Timedelta("1m")  # optional: only join if within 10 minutes
)

variables = ["temperature","gasResistance","humidity"]#,
            #"diff_humidity", "diff_temperature"]
df_clean = df_sens_merged.dropna(subset=variables + ["CH4"])
df_clean = df_clean.copy()          # break the link to the original
df_clean["_time"] = pd.to_datetime(df_clean["_time"])


start = "2025-04-09 04:00:00"
test_start ="2025-04-09 04:00:00"
stop = "2025-04-26 14:00:00"

#start = "2025-04-05 08:00:00"
#test_start ="2025-04-05 08:00:00"
#stop = "2025-04-28 13:00:00"
#only keep variables of interest
df_clean = df_clean[variables + ["CH4", "_time"]]

df_clean2 = df_clean.copy()          # break the link to the original

import numpy as np
import pandas as pd

# ────────── configuration ──────────
BASE_COLS   = ["temperature", "gasResistance", "humidity"]
WIN_LIST    = [10,20,30,40]          # 10 samples ≃ 5 min.  add 20, 40 … if desired
EWM_SPANS   = [10,20,30,40]          # exponential-weighted mean/slope
DROP_NA     = True          # drop first max(window) rows

def build_hysteresis_features(df: pd.DataFrame,
                              cols=BASE_COLS,
                              wins=WIN_LIST,
                              spans=EWM_SPANS,
                              drop_na=DROP_NA) -> pd.DataFrame:
    """Return a new dataframe with raw, diff, rolling-mean, mean-slope,
       rolling-std and EWM level/slope for each base column."""
    out = df.copy()

    for c in cols:
        out[f"{c}_d1"] = out[c].diff()

        for w in wins:
            rm = out[c].rolling(w, min_periods=1)
            out[f"{c}_mean{w}"]  = rm.mean()
            out[f"{c}_std{w}"]   = rm.std()
            out[f"{c}_mslope{w}"] = (rm.mean() - rm.mean().shift(w)).div(w)

        for s in spans:
            ewm = out[c].ewm(span=s, adjust=False)
            out[f"{c}_ewm{s}"]   = ewm.mean()
            out[f"{c}_ewslope{s}"] = ewm.mean().diff()

    return out.dropna() if drop_na else out

def ct_feature_names(ct, original_cols):
    """
    Return output column names for *any* fitted ColumnTransformer,
    even when some inner steps (FunctionTransformer, etc.) do not
    implement get_feature_names_out().
    """
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue

        # --- resolve the column selector to actual labels ---
        if isinstance(cols, slice):
            sel = original_cols[cols]
        elif cols == slice(None):
            sel = original_cols
        elif isinstance(cols, (list, tuple, np.ndarray)):
            sel = [original_cols[i] if isinstance(i, int) else i for i in cols]
        else:   # boolean mask
            sel = [c for c, flag in zip(original_cols, cols) if flag]

        # --- try to get names from the transformer itself ---
        if hasattr(trans, "get_feature_names_out"):
            try:
                trans_names = trans.get_feature_names_out(sel)
            except TypeError:
                trans_names = trans.get_feature_names_out()
        elif hasattr(trans, "get_feature_names"):
            trans_names = trans.get_feature_names()
        else:
            # fall back: keep the input names
            trans_names = sel

        names.extend(trans_names)
    return np.asarray(names)

df_feat = build_hysteresis_features(df_clean2, cols=BASE_COLS,
                                         wins=WIN_LIST,
                                         spans=EWM_SPANS,
                                         drop_na=DROP_NA)

# your existing time cuts
df_train = df_feat[(df_feat["_time"] > test_start) & (df_feat["_time"] < stop)]
df_test  = df_feat[((df_feat["_time"] < test_start) & (df_feat["_time"] > start)) |
                   (df_feat["_time"] > stop)]

# list every feature column except the target + timestamp
FEATURES = [c for c in df_feat.columns
            if c not in ["_time", "CH4"]]

X_train, y_train = df_train[FEATURES], df_train["CH4"]
X_test,  y_test  = df_test[FEATURES],  df_test["CH4"]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# ----------  helper transformers ----------
def safe_log1p(arr):
    """Clip to just above -1 so log1p is defined, then log1p."""
    clipped = np.where(arr <= -1 + 1e-9, -1 + 1e-9, arr)
    return np.log1p(clipped)

num_cols = FEATURES
log_gas_cols = [c for c in num_cols if c.startswith("gasResistance")]
numeric_log   = log_gas_cols                               # only always-positive-ish columns
numeric_other = [c for c in FEATURES if c not in numeric_log]

preprocess = ColumnTransformer(
    transformers=[
        ("log_gas",
         Pipeline([
             ("log", FunctionTransformer(
                 safe_log1p, validate=False,
                 feature_names_out="one-to-one")),
             ("imp", SimpleImputer(strategy="median")),
             ("sc",  StandardScaler()),
         ]),
         numeric_log),
        ("num",
         Pipeline([
             ("imp", SimpleImputer(strategy="median")),
             ("sc",  StandardScaler()),
         ]),
         numeric_other),
    ],
    remainder="drop",
)

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

models = {
    "GradientBoosting": Pipeline([
        ("prep", preprocess),
        ("gb", GradientBoostingRegressor(
            n_estimators=800, learning_rate=0.02,
            max_depth=3, subsample=0.8, random_state=0)),
    ])
}

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

tscv = TimeSeriesSplit(n_splits=5)

fitted_models = {}                       # <-- NEW

rows = []
for name, pipe in models.items():
    cv_rmse = -cross_val_score(pipe, X_train, y_train,
                               scoring="neg_root_mean_squared_error",
                               cv=tscv, n_jobs=-1).mean()
    pipe.fit(X_train, y_train)

    # ⬇️ save the trained pipeline
    fitted_models[name] = pipe          # <-- NEW

    y_pred = pipe.predict(X_test)
    rows.append({
        "Model":     name,
        "CV-RMSE":   cv_rmse,
        "Test-RMSE": rmse(y_test, y_pred),
        "Test-MAE":  mean_absolute_error(y_test, y_pred),
        "Test-R²":   r2_score(y_test, y_pred),
    })

results = pd.DataFrame(rows).sort_values("Test-RMSE")
print(results)

# pick the winner
best_gb = fitted_models["GradientBoosting"]   # ← now defined


import matplotlib.pyplot as plt

# 1️⃣  Predict on the test set
y_pred = best_gb.predict(X_test)

# 2️⃣  Line plot: actual vs. predicted
plt.figure(figsize=(10, 4))
plt.plot(df_test["_time"], y_test, label="Actual")
plt.plot(df_test["_time"], y_pred, label="Predicted")
plt.title("CH₄ – Actual vs. Gradient-Boosting Prediction (test period)")
plt.xlabel("Timestamp")
plt.ylabel("CH₄ concentration (ppm)")
plt.legend()
plt.tight_layout()
plt.show()


import shap

prep = best_gb["prep"]         # ColumnTransformer
gb   = best_gb["gb"]           # fitted GradientBoostingRegressor

X_trans = prep.transform(X_train)      # the matrix the tree saw

explainer   = shap.Explainer(gb)       # let SHAP choose the right backend
shap_values = explainer(X_trans, check_additivity=False)

X_sample = shap.sample(X_trans, 5000, random_state=0)
shap_values = explainer(X_sample, check_additivity=False)


feat_names = ct_feature_names(prep, FEATURES)

import numpy as np, pandas as pd

mean_abs = np.abs(shap_values.values).mean(axis=0)
imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
total = imp.sum()
keep = imp[imp >= 0.01 * total].index.tolist()
print("\nTop 30 features by mean |SHAP|:")
print(imp.head(30).to_string(float_format="%.4f"))
print("\nFeatures with |SHAP| >= 1prs of total:")

print(keep)
