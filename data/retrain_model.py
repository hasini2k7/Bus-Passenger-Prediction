
import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Project folder is one level above the data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset and saved model locations
DATA_FILE = os.path.join(BASE_DIR, "updated_passenger_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "bus_passenger_model.pkl")

# Load the dataset
data = pd.read_csv(DATA_FILE)

# Check required columns
required = [
    "date",
    "time",
    "route",
    "weather",
    "passenger_count"
]

if not all(col in data.columns for col in required):
    raise ValueError(
        "CSV must contain: date, time, route, weather, passenger_count"
    )

# Prepare features
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["hour"] = pd.to_datetime(
    data["time"], format="%H:%M", errors="coerce"
).dt.hour
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["passenger_count"] = pd.to_numeric(
    data["passenger_count"], errors="coerce"
)

# Remove invalid records
data = data.dropna(
    subset=[
        "hour",
        "day_of_week",
        "month",
        "route",
        "weather",
        "passenger_count"
    ]
)

if len(data) < 5:
    raise ValueError(
        "At least 5 valid passenger records are required."
    )

# Separate input features and target
X = data[
    ["hour", "day_of_week", "month", "route", "weather"]
]
y = data["passenger_count"]

# Encode route and weather
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["route", "weather"]
        ),
        (
            "numeric",
            "passthrough",
            ["hour", "day_of_week", "month"]
        )
    ]
)

# Build the model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# Train using the dataset
model.fit(X, y)

# Save the trained model
joblib.dump(model, MODEL_FILE)

print("Retraining completed successfully!")
print(f"Training records used: {len(data)}")
print(f"Model saved at: {MODEL_FILE}")
