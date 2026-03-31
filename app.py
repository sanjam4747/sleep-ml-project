from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load artifacts produced by training pipeline.
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_COLUMNS = [
    "WorkoutTime",
    "ReadingTime",
    "PhoneTime",
    "WorkHours",
    "CaffeineIntake",
    "RelaxationTime",
]

@app.route("/")
def home():
    return "Sleep Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # simple recommendation logic
    if prediction < 6:
        advice = "Your sleep is low. Reduce screen time and improve routine."
    elif prediction > 9:
        advice = "You might be oversleeping. Maintain a balanced schedule."
    else:
        advice = "Healthy sleep pattern."

    return jsonify({
        "predicted_sleep": float(prediction),
        "recommendation": advice
    })

if __name__ == "__main__":
    app.run(debug=True)