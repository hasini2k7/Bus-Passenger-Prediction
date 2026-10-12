
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date, time, datetime
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Bus Passenger Prediction",
    page_icon="🚌"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "bus_passenger_model_new.pkl")
LOG_FILE = os.path.join(BASE_DIR, "prediction_logs.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def load_logs():
    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)

        # Add this column to older log files if needed
        if "actual_passengers" not in logs.columns:
            logs["actual_passengers"] = pd.NA
            logs.to_csv(LOG_FILE, index=False)

        return logs

    return pd.DataFrame(columns=[
        "prediction_time",
        "travel_date",
        "bus_time",
        "route",
        "weather",
        "predicted_passengers",
        "actual_passengers"
    ])


st.title("🚌 Bus Passenger Count Prediction")
st.write("Predict, record actual passengers, and monitor accuracy.")

try:
    model = load_model()

    st.subheader("1. Predict Passenger Count")

    selected_date = st.date_input(
        "Select Travel Date",
        value=date.today()
    )

    selected_time = st.time_input(
        "Select Bus Time",
        value=time(8, 0)
    )

    route = st.selectbox(
        "Select Bus Route",
        ["Route_A", "Route_B", "Route_C"]
    )

    weather = st.selectbox(
        "Select Weather",
        ["Clear", "Cloudy", "Rainy"]
    )

    if st.button("Predict Passenger Count"):
        input_data = pd.DataFrame([{
            "hour": selected_time.hour,
            "day_of_week": selected_date.weekday(),
            "month": selected_date.month,
            "route": route,
            "weather": weather
        }])

        prediction = model.predict(input_data)[0]
        passenger_count = max(0, round(prediction))

        log_entry = pd.DataFrame([{
            "prediction_time": datetime.now().isoformat(
                timespec="seconds"
            ),
            "travel_date": selected_date.isoformat(),
            "bus_time": selected_time.strftime("%H:%M"),
            "route": route,
            "weather": weather,
            "predicted_passengers": passenger_count,
            "actual_passengers": pd.NA
        }])

        logs = load_logs()
        logs = pd.concat([logs, log_entry], ignore_index=True)
        logs.to_csv(LOG_FILE, index=False)

        st.success(
            f"Predicted Passenger Count: {passenger_count}"
        )
        st.info("Prediction saved. Enter the actual count after the trip.")

    st.divider()
    st.subheader("2. Record Actual Passenger Count")

    logs = load_logs()

    if not logs.empty:
        # Only show predictions that still need actual counts
        logs["actual_passengers"] = pd.to_numeric(
            logs["actual_passengers"], errors="coerce"
        )

        pending = logs[logs["actual_passengers"].isna()]

        if not pending.empty:
            selected_index = st.selectbox(
                "Select the prediction for the completed trip",
                pending.index.tolist(),
                format_func=lambda i: (
                    f"Record {i + 1}: "
                    f"{logs.loc[i, 'travel_date']} "
                    f"{logs.loc[i, 'bus_time']} - "
                    f"{logs.loc[i, 'route']} "
                    f"(predicted: "
                    f"{logs.loc[i, 'predicted_passengers']})"
                )
            )

            actual_count = st.number_input(
                "Enter actual passenger count",
                min_value=0,
                step=1,
                value=0
            )

            if st.button("Save Actual Passenger Count"):
                logs.loc[
                    selected_index, "actual_passengers"
                ] = actual_count

                logs.to_csv(LOG_FILE, index=False)
                st.success("Actual passenger count saved!")
                st.rerun()

        else:
            st.info(
                "All recorded predictions have actual counts."
            )

        st.divider()
        st.subheader("3. Prediction Monitoring")

        st.dataframe(logs, use_container_width=True)

        completed = logs.dropna(
            subset=["actual_passengers"]
        ).copy()

        if not completed.empty:
            completed["predicted_passengers"] = pd.to_numeric(
                completed["predicted_passengers"]
            )
            completed["actual_passengers"] = pd.to_numeric(
                completed["actual_passengers"]
            )

            mae = mean_absolute_error(
                completed["actual_passengers"],
                completed["predicted_passengers"]
            )

            st.metric(
                "Mean Absolute Error (MAE)",
                f"{mae:.2f} passengers"
            )

            st.caption(
                "MAE shows the average absolute difference "
                "between predicted and actual passenger counts. "
                "Lower is better."
            )

            st.write(
                "Trips with actual counts:",
                len(completed)
            )
        else:
            st.info(
                "Accuracy will appear after you record at least "
                "one actual passenger count."
            )

        st.write("Total predictions:", len(logs))

    else:
        st.info("Make a prediction first to start monitoring.")

except FileNotFoundError:
    st.error(
        "Model file not found. Run python train_model.py first."
    )
except Exception as error:
    st.error(f"An error occurred: {error}")