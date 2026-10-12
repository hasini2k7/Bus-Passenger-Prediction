import pandas as pd

# Read the passenger data
data = pd.read_csv("data/passenger_data.csv")

print("Passenger data loaded successfully!")
print("Total records:", len(data))

print("\nColumn names:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

print("\nData check completed!")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load passenger data
data = pd.read_csv("data/passenger_data.csv")

# Step 2: Prepare date and time
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["hour"] = pd.to_datetime(
    data["time"], format="%H:%M", errors="coerce"
).dt.hour

data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month

# Step 3: Remove invalid records
data = data.dropna(
    subset=[
        "date", "hour", "day_of_week",
        "month", "route", "weather", "passenger_count"
    ]
)

# Step 4: Select input features and target
features = ["hour", "day_of_week", "month", "route", "weather"]
X = data[features]
y = data["passenger_count"]

if len(data) < 5:
    raise ValueError("Please add at least 5 valid records.")

# Step 5: Split data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Load the trained model
model = joblib.load("bus_passenger_model.pkl")

# Step 7: Make predictions on test records
predictions = model.predict(X_test)

# Step 8: Calculate evaluation scores
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("Model evaluation completed!")
print("Test records:", len(X_test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("Lower MAE and RMSE are generally better.")
