from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load Trained Models
migration_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl')
location_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl')
time_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl')

rf_final = migration_model['rf_final']
location_rf = location_model['location_model']
month_model = time_model['month_model']
hour_model = time_model['hour_model']

# Load Label Encoders
le_species = migration_model['label_encoders']['COMMON NAME']
le_locality = migration_model['label_encoders']['LOCALITY']

# Helper Functions
def encode_inputs(species, locality):
    species_encoded = le_species.transform([species])[0] if species in le_species.classes_ else None
    locality_encoded = le_locality.transform([locality])[0] if locality in le_locality.classes_ else None
    return species_encoded, locality_encoded

def convert_day_of_week(day_name):
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    return days.get(day_name, None)

def convert_time_of_day(time_period):
    time_ranges = {'morning': 7, 'afternoon': 13, 'evening': 18, 'night': 22}
    return time_ranges.get(time_period.lower(), None)

# ✅ Endpoint 1: Migration Presence Prediction
@app.route("/predict_migration_presence", methods=["GET"])
def predict_migration_presence():
    try:
        species = request.args.get("species")
        year = int(request.args.get("year"))
        month = int(request.args.get("month"))
        day_of_week = convert_day_of_week(request.args.get("day_of_week"))
        hour = int(request.args.get("hour"))
        locality = request.args.get("locality")

        if day_of_week is None:
            return jsonify({"error": "Invalid day of the week"}), 400

        species_encoded, locality_encoded = encode_inputs(species, locality)
        if species_encoded is None or locality_encoded is None:
            return jsonify({"error": "Invalid species or locality"}), 400

        input_data = pd.DataFrame([[year, month, day_of_week, hour, locality_encoded, species_encoded]],
                                  columns=['Year', 'Month', 'Day_of_Week', 'Hour', 'LOCALITY_ENCODED', 'COMMON NAME_ENCODED'])

        prob = rf_final.predict_proba(input_data)[:, 1][0]
        presence = "Yes" if prob >= 0.5 else "No"

        return jsonify({
            "species": species,
            "locality": locality,
            "presence": presence,
            "confidence": f"{prob:.2%}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Endpoint 2: Best Locations for Bird Species
@app.route("/predict_bird_location", methods=["GET"])
def predict_bird_location():
    try:
        species = request.args.get("species")
        year = int(request.args.get("year"))
        month = int(request.args.get("month"))
        day_of_week = convert_day_of_week(request.args.get("day_of_week"))
        time_period = request.args.get("time_period")

        if day_of_week is None:
            return jsonify({"error": "Invalid day of the week"}), 400

        species_encoded, _ = encode_inputs(species, None)
        if species_encoded is None:
            return jsonify({"error": "Invalid species"}), 400

        time_hour = convert_time_of_day(time_period)
        if time_hour is None:
            return jsonify({"error": "Invalid time period (choose morning, afternoon, evening, or night)"}), 400

        input_data = pd.DataFrame([[year, month, day_of_week, time_hour, species_encoded]],
                                  columns=['Year', 'Month', 'Day_of_Week', 'Hour', 'COMMON NAME_ENCODED'])

        probabilities = location_rf.predict_proba(input_data)[:, 1]
        location_predictions = sorted(zip(le_locality.classes_, probabilities), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            "species": species,
            "time_period": time_period,
            "best_locations": [{ "location": loc, "probability": f"{prob:.1%}"} for loc, prob in location_predictions]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Endpoint 3: Best Time for Birdwatching
@app.route("/predict_best_time", methods=["GET"])
def predict_best_time():
    try:
        species = request.args.get("species")
        locality = request.args.get("locality")
        year = int(request.args.get("year"))
        day_of_week = convert_day_of_week(request.args.get("day_of_week"))
        season = request.args.get("season")
        time_period = request.args.get("time_period")

        if day_of_week is None:
            return jsonify({"error": "Invalid day of the week"}), 400

        species_encoded, locality_encoded = encode_inputs(species, locality)
        if species_encoded is None or locality_encoded is None:
            return jsonify({"error": "Invalid species or locality"}), 400

        features = {'OBSERVATION': 1, 'Year': year, 'Day_of_Week': day_of_week,
                    'Is_Summer': 0, 'Is_Winter': 0, 'Is_Spring': 0, 'Is_Autumn': 0,
                    'Is_Morning': 0, 'Is_Afternoon': 0, 'Is_Evening': 0, 'Is_Night': 0}

        if season in features:
            features[season] = 1
        if time_period in features:
            features[time_period] = 1

        input_data = pd.DataFrame([[1, year, day_of_week, species_encoded, locality_encoded] + list(features.values())[3:]],
                                  columns=['OBSERVATION', 'Year', 'Day_of_Week', 'COMMON NAME_ENCODED', 'LOCALITY_ENCODED'] + list(features.keys())[3:])

        best_month = month_model.predict(input_data)[0]
        best_hour = hour_model.predict(input_data)[0]

        return jsonify({
            "species": species,
            "locality": locality,
            "best_month": int(best_month),
            "best_hour": int(best_hour)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
