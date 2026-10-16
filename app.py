import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HVAC Failure Predictor", page_icon="❄️")

# load everything once
rf = joblib.load("rf_model.pkl")
lr = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")
config = joblib.load("config.pkl")

st.title("❄️ HVAC Failure Predictor")
st.write("Enter the sensor readings and the model will tell you if the unit looks like it's heading for trouble.")

model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 24.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 51.0)
    vibration = st.number_input("Vibration (mm/s)", 0.0, 20.0, 0.65)
with col2:
    pressure = st.number_input("Pressure (kPa)", 80.0, 160.0, 101.0)
    energy = st.number_input("Energy (kW)", 0.0, 80.0, 18.0)
    runtime = st.number_input("Runtime (hours)", 0.0, 5000.0, 1600.0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "vibration_mm_s": vibration,
        "pressure_kpa": pressure,
        "energy_kw": energy,
        "runtime_hours": runtime,
    }])[config["features"]]          # same column order as training

    if model_choice == "Random Forest":
        proba = rf.predict_proba(row)[0, 1]
        threshold = config["rf_threshold"]
    else:
        proba = lr.predict_proba(scaler.transform(row))[0, 1]
        threshold = config["lr_threshold"]

    st.metric("Failure probability", f"{proba:.1%}")
    st.progress(min(float(proba), 1.0))

    if proba > threshold:
        st.error("⚠️ High risk of failure. Schedule an inspection.")
    else:
        st.success("✅ Looks normal.")

    st.caption(f"Decision threshold used: {threshold}")