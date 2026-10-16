import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")  # hide harmless "feature names" warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Vercel looks for this exact variable name: "app"
app = Flask(__name__)

# ---- load everything once, when the server starts ----
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
lr = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
config = joblib.load(os.path.join(MODEL_DIR, "config.pkl"))

FEATURES = config["features"]  # exact column order used in training

# (label shown on the page, form field name, min, max, default)
FIELDS = [
    ("Temperature (°C)", "temperature_c", 0, 60, 24.0),
    ("Humidity (%)", "humidity_pct", 0, 100, 51.0),
    ("Vibration (mm/s)", "vibration_mm_s", 0, 20, 0.65),
    ("Pressure (kPa)", "pressure_kpa", 80, 160, 101.0),
    ("Energy (kW)", "energy_kw", 0, 80, 18.0),
    ("Runtime (hours)", "runtime_hours", 0, 5000, 1600.0),
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        fields=FIELDS,
        rf_threshold=config["rf_threshold"],
        lr_threshold=config["lr_threshold"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # 1) read + validate the six numbers
    values = {}
    for label, name, low, high, _ in FIELDS:
        try:
            v = float(data[name])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"Please enter a valid number for {label}."), 400
        if not (low <= v <= high):
            return jsonify(error=f"{label} must be between {low} and {high}."), 400
        values[name] = v

    # 2) build the row in the SAME order as training
    row = np.array([[values[f] for f in FEATURES]], dtype=float)

    # 3) pick the model
    model_choice = data.get("model", "rf")
    if model_choice == "lr":
        proba = float(lr.predict_proba(scaler.transform(row))[0, 1])  # LR needs scaling
        threshold = config["lr_threshold"]
        model_name = "Logistic Regression"
    else:
        proba = float(rf.predict_proba(row)[0, 1])  # RF takes raw values
        threshold = config["rf_threshold"]
        model_name = "Random Forest"

    # 4) turn the probability into a verdict
    if proba > threshold:
        level, message = "high", "High risk of failure. Schedule an inspection."
    elif proba > threshold * 0.75:
        level, message = "medium", "Borderline. Keep an eye on this unit."
    else:
        level, message = "low", "Looks normal."

    return jsonify(
        model=model_name,
        probability=round(proba, 4),
        threshold=threshold,
        level=level,
        message=message,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
