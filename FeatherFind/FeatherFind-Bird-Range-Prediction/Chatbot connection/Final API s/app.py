from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import datetime
import os
import requests
import logging
from difflib import get_close_matches
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Shared Configuration
# ========================
VALID_BIRD_NAMES = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]
VALID_LOCALITIES = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park",
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake"
]

BIRD_ALIASES = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "blue bird": "Blue-tailed Bee-eater",
    "bee eater": "Blue-tailed Bee-eater",
    "red bird": "Red-vented Bulbul",
    "bulbul": "Red-vented Bulbul",
    "white bird": "White-throated Kingfisher",
    "kingfisher": "White-throated Kingfisher"
}

# ========================
# Model Loading Functions
# ========================
def download_model(url, save_path):
    if not os.path.exists(save_path):
        logger.info(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return joblib.load(save_path)

# Load all models
location_model_data = download_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl",
    "location_model.pkl"
)

presence_model_data = download_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl",
    "presence_model.pkl"
)

time_model_data = download_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl",
    "time_model.pkl"
)

# ========================
# Shared Utility Functions
# ========================
def correct_bird_name(name):
    name = name.lower()
    if name in BIRD_ALIASES:
        return BIRD_ALIASES[name]
    matches = get_close_matches(name, [b.lower() for b in VALID_BIRD_NAMES], n=1, cutoff=0.3)
    return matches[0].title() if matches else "Unknown Bird"

def parse_date(query):
    today = datetime.date.today()
    date_mappings = {
        "tomorrow": today + datetime.timedelta(days=1),
        "day after tomorrow": today + datetime.timedelta(days=2),
        "next week": today + datetime.timedelta(weeks=1)
    }
    for key in date_mappings:
        if key in query:
            return date_mappings[key]
    match = re.search(r"in (\d+) days", query)
    return today + datetime.timedelta(days=int(match.group(1))) if match else today

def extract_datetime_features(query):
    query = query.lower()
    date_obj = parse_date(query)
    time_match = re.search(r'\b(\d{1,2})(?::\d{2})?\s?(am|pm)?\b', query)
    
    hour = datetime.datetime.now().hour
    if time_match:
        hour = int(time_match.group(1))
        if time_match.group(2) == 'pm' and hour < 12:
            hour += 12
        elif time_match.group(2) == 'am' and hour == 12:
            hour = 0

    return {
        "year": date_obj.year,
        "month": date_obj.month,
        "day": date_obj.day,
        "day_of_week": date_obj.weekday(),
        "day_name": date_obj.strftime("%A"),
        "hour": hour,
        "time_of_day": get_time_of_day(hour)
    }

def get_time_of_day(hour):
    if 6 <= hour < 11: return "morning"
    if 11 <= hour < 16: return "afternoon"
    if 16 <= hour < 20: return "evening"
    return "night"

# ========================
# Route Implementations
# ========================
@app.route('/predict_location', methods=['POST'])
def predict_location():
    try:
        data = request.get_json()
        query = data.get("query", "").lower()
        features = extract_datetime_features(query)
        
        # Bird name handling
        bird_name = next((b for b in VALID_BIRD_NAMES if b.lower() in query), "Unknown Bird")
        if bird_name == "Unknown Bird":
            return jsonify({"error": "Please specify a valid bird species", "options": VALID_BIRD_NAMES}), 400

        # Model prediction
        location_model = location_model_data['location_model']
        label_encoders = location_model_data['label_encoders']
        bird_encoded = label_encoders['COMMON NAME'].transform([bird_name])[0]
        
        predictions = []
        for loc in location_model_data['predefined_locations']:
            input_df = pd.DataFrame([[
                features['year'],
                features['month'],
                features['day_of_week'],
                features['hour'],
                loc['LATITUDE'],
                1,  # Assuming COUNTRY_CODE
                loc['LONGITUDE'],
                bird_encoded
            ]], columns=location_model_data['selected_features'])
            
            pred = location_model.predict(input_df)[0]
            predictions.append(label_encoders['LOCALITY'].inverse_transform([pred])[0])

        return jsonify({
            "bird": bird_name,
            "date": f"{features['day_name']}, {features['month']}/{features['day']}/{features['year']}",
            "locations": list(set(predictions))
        }), 200

    except Exception as e:
        logger.error(f"Location prediction error: {str(e)}")
        return jsonify({"error": "Location prediction failed"}), 500

@app.route('/predict_presence', methods=['POST'])
def predict_presence():
    try:
        data = request.get_json()
        query = data.get("query", "").lower()
        features = extract_datetime_features(query)
        
        # Extract location and bird
        location = next((loc for loc in VALID_LOCALITIES if loc.lower() in query), "Unknown Location")
        bird_name = next((b for b in VALID_BIRD_NAMES if b.lower() in query), "Unknown Bird")
        
        if location == "Unknown Location":
            return jsonify({"error": "Please specify a valid location", "options": VALID_LOCALITIES}), 400
        if bird_name == "Unknown Bird":
            return jsonify({"error": "Please specify a valid bird species", "options": VALID_BIRD_NAMES}), 400

        # Model prediction
        model = presence_model_data['rf_final']
        label_encoders = presence_model_data['label_encoders']
        
        loc_encoded = label_encoders['LOCALITY'].transform([location])[0]
        bird_encoded = label_encoders['COMMON NAME'].transform([bird_name])[0]
        
        input_df = pd.DataFrame([[
            features['year'],
            features['month'],
            features['day_of_week'],
            features['hour'],
            loc_encoded,
            bird_encoded
        ]], columns=presence_model_data['selected_features'])
        
        probability = model.predict_proba(input_df)[0][1]
        return jsonify({
            "present": probability > 0.5,
            "confidence": float(probability),
            "location": location,
            "bird": bird_name,
            "time": features['time_of_day']
        }), 200

    except Exception as e:
        logger.error(f"Presence prediction error: {str(e)}")
        return jsonify({"error": "Presence prediction failed"}), 500

@app.route('/predict_best_time', methods=['POST'])
def predict_best_time():
    try:
        data = request.get_json()
        query = data.get("query", "").lower()
        features = extract_datetime_features(query)
        
        # Extract location and bird
        location = next((loc for loc in VALID_LOCALITIES if loc.lower() in query), "Unknown Location")
        bird_name = next((b for b in VALID_BIRD_NAMES if b.lower() in query), "Unknown Bird")
        
        if location == "Unknown Location":
            return jsonify({"error": "Please specify a valid location", "options": VALID_LOCALITIES}), 400
        if bird_name == "Unknown Bird":
            return jsonify({"error": "Please specify a valid bird species", "options": VALID_BIRD_NAMES}), 400

        # Model prediction
        time_model = time_model_data['hour_model']
        month_model = time_model_data['month_model']
        label_encoders = time_model_data['label_encoders']
        
        loc_encoded = label_encoders['LOCALITY'].transform([location])[0]
        bird_encoded = label_encoders['COMMON NAME'].transform([bird_name])[0]
        
        input_df = pd.DataFrame([[
            1,  # Dummy feature
            features['year'],
            features['day_of_week'],
            loc_encoded,
            bird_encoded,
            0, 0, 0, 0,  # Season flags
            0, 0, 0, 0   # Time period flags
        ]], columns=time_model_data['selected_features'])
        
        best_hour = int(round(time_model.predict(input_df)[0]))
        best_month = int(round(month_model.predict(input_df)[0]))
        
        return jsonify({
            "best_time": f"{best_hour % 12 or 12} {'AM' if best_hour < 12 else 'PM'}",
            "best_month": datetime.date(1900, best_month, 1).strftime("%B"),
            "location": location,
            "bird": bird_name
        }), 200

    except Exception as e:
        logger.error(f"Time prediction error: {str(e)}")
        return jsonify({"error": "Time prediction failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)