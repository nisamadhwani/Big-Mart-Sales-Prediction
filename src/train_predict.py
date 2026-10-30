"""Big Mart Sales Prediction - reproducible training and test prediction."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "data" / "Train.csv")
test = pd.read_csv(ROOT / "data" / "Test.csv")
X = train.drop(columns=["Item_Outlet_Sales"])
y = train["Item_Outlet_Sales"]
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)
model = HistGradientBoostingRegressor(
    max_iter=180, learning_rate=0.08, l2_regularization=1.0, random_state=42
)
model.fit(X_train_t, y_train)
valid_pred = model.predict(X_valid_t)
print(f"Validation R2: {r2_score(y_valid, valid_pred):.4f}")
print(f"Validation MAE: {mean_absolute_error(y_valid, valid_pred):.2f}")
print(f"Validation RMSE: {mean_squared_error(y_valid, valid_pred)**0.5:.2f}")
# Refit on all labelled rows before predicting the supplied competition/test file.
all_X_t = preprocessor.fit_transform(X)
test_t = preprocessor.transform(test)
model.fit(all_X_t, y)
pred = model.predict(test_t)
out = ROOT / "results" / "Test_Predictions.csv"
pd.DataFrame({"Item_Outlet_Sales_Predicted": pred}).to_csv(out, index=False)
print(f"Saved test predictions to: {out}")
