from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

# Load the Pay2Gether model from this project folder.
model_path = Path(__file__).resolve().parent / "fintech.joblib"
model = joblib.load(model_path)


def predict_budget(number_of_members, trip_length_days, planned_budget_per_person):
    data = pd.DataFrame(
        {
            "number_of_members": [number_of_members],
            "trip_length_days": [trip_length_days],
            "planned_budget_per_person": [planned_budget_per_person],
        }
    )

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return prediction, probability


# HTML front end
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            number_of_members = int(request.form["number_of_members"])
            trip_length_days = int(request.form["trip_length_days"])
            planned_budget_per_person = float(
                request.form["planned_budget_per_person"]
            )

            prediction, probability = predict_budget(
                number_of_members,
                trip_length_days,
                planned_budget_per_person,
            )

            if prediction == 1:
                message = "The group is likely to exceed its planned budget."
            else:
                message = "The group is likely to stay within its planned budget."

            result = {
                "message": message,
                "probability": round(probability * 100, 1),
            }

        except (ValueError, KeyError):
            error = "Please enter valid numbers in all fields."

    return render_template("index.html", result=result, error=error)


# API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    try:
        prediction, probability = predict_budget(
            int(data["number_of_members"]),
            int(data["trip_length_days"]),
            float(data["planned_budget_per_person"]),
        )

        return jsonify(
            {
                "prediction": prediction,
                "probability_of_exceeding_budget": round(probability, 4),
            }
        )

    except (ValueError, KeyError, TypeError):
        return jsonify({"error": "Invalid or missing input variables."}), 400