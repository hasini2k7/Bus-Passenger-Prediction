
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load the dataset
df = pd.read_csv("data/passenger_data.csv")

# Step 2: Prepare date and time
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["hour"] = pd.to_datetime(
    df["time"], format="%H:%M", errors="coerce"
).dt.hour

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Step 3: Remove invalid rows
df = df.dropna(
    subset=[
        "date", "hour", "day_of_week",
        "month", "route", "weather", "passenger_count"
    ]
)

# Step 4: Select input and output columns
features = ["hour", "day_of_week", "month", "route", "weather"]
X = df[features]
y = df["passenger_count"]

if len(df) < 5:
    raise ValueError("Please provide at least 5 valid records.")

# Step 5: Process text columns
preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        ["route", "weather"]
    )
], remainder="passthrough")

# Step 6: Create the prediction model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Step 7: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("Model training completed!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Step 10: Save the model
joblib.dump(model, "bus_passenger_model_new.pkl")
print("New compatible model saved successfully!")
