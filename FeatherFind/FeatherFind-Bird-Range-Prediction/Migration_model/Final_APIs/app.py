from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import datetime
from difflib import get_close_matches
import requests
import io
from flask_cors import CORS
import logging
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Model Loading
# ========================
def load_model(url, save_path):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return joblib.load(f)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    with open(save_path, "rb") as f:
        return joblib.load(f)

# Load all models
location_model_data = load_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/location_prediction_model.pkl",
    "location_model.pkl"
)
presence_model_data = load_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl",
    "presence_model.pkl"
)
time_model_data = load_model(
    "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl",
    "time_model.pkl"
)

# Model components
location_model = location_model_data['location_model']
location_features = location_model_data['selected_features']
location_encoders = location_model_data['label_encoders']

presence_model = presence_model_data['rf_final']
presence_encoders = presence_model_data['label_encoders']
presence_features = presence_model_data['selected_features']

month_model = time_model_data['month_model']
hour_model = time_model_data['hour_model']
time_encoders = time_model_data['label_encoders']
time_features = time_model_data['selected_features']

# ========================
# Shared Utilities
# ========================
VALID_BIRDS = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]
LOCALITIES = [
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

def correct_bird_name(name):
    name = name.lower()
    if name in BIRD_ALIASES:
        return BIRD_ALIASES[name]
    matches = get_close_matches(name, [b.lower() for b in VALID_BIRDS], n=1, cutoff=0.3)
    return matches[0].title() if matches else "Unknown Bird"

def parse_date(query):
    today = datetime.date.today()
    # Date patterns
    patterns = {
        'tomorrow': today + datetime.timedelta(days=1),
        'day after tomorrow': today + datetime.timedelta(days=2),
        'next week': today + datetime.timedelta(weeks=1)
    }
    
    for term, date in patterns.items():
        if term in query.lower():
            return date
    
    # Numeric date pattern
    match = re.search(r'in (\d+) days', query, re.IGNORECASE)
    if match:
        return today + datetime.timedelta(days=int(match.group(1)))
    
    # Specific date parsing
    try:
        parsed_date = parser.parse(query, fuzzy=True)
        return parsed_date.date()
    except:
        return today

# ========================
# Prediction Endpoints
# ========================
@app.route('/predict_location', methods=['POST'])
def predict_location():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Feature extraction
        date = parse_date(query)
        bird = next((b for b in VALID_BIRDS if b.lower() in query.lower()), "Unknown Bird")
        if bird == "Unknown Bird":
            return jsonify({"error": "Bird species not recognized"}), 400

        # Prediction logic
        results = []
        for loc in LOCALITIES:
            input_data = pd.DataFrame([[
                date.year, date.month, date.weekday(),
                datetime.datetime.now().hour,
                location_encoders['LOCALITY'].transform([loc])[0],
                1,  # Assuming constant value from original code
                location_encoders['COMMON NAME'].transform([bird])[0]
            ]], columns=location_features)
            
            prediction = location_model.predict(input_data)[0]
            results.append(location_encoders['LOCALITY'].inverse_transform([prediction])[0])

        return jsonify({
            "bird": bird,
            "locations": list(set(results)),
            "date": date.strftime("%Y-%m-%d")
        }), 200

    except Exception as e:
        logger.error(f"Location prediction error: {str(e)}")
        return jsonify({"error": "Location prediction failed"}), 500

@app.route('/predict_presence', methods=['POST'])
def predict_presence():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Feature extraction
        date = parse_date(query)
        bird = correct_bird_name(query)
        locality = next((l for l in LOCALITIES if l.lower() in query.lower()), None)
        
        if not locality or bird == "Unknown Bird":
            return jsonify({"error": "Missing required parameters"}), 400

        # Prepare input
        input_data = pd.DataFrame([[
            date.year,
            date.month,
            date.weekday(),
            datetime.datetime.now().hour,
            presence_encoders['LOCALITY'].transform([locality])[0],
            presence_encoders['COMMON NAME'].transform([bird])[0]
        ]], columns=presence_features)
        
        probability = presence_model.predict_proba(input_data)[0][1]
        return jsonify({
            "present": probability >= 0.5,
            "confidence": round(float(probability), 2),
            "bird": bird,
            "location": locality,
            "date": date.strftime("%Y-%m-%d")
        }), 200

    except Exception as e:
        logger.error(f"Presence prediction error: {str(e)}")
        return jsonify({"error": "Presence prediction failed"}), 500

@app.route('/predict_best_time', methods=['POST'])
def predict_best_time():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Feature extraction
        date = parse_date(query)
        bird = correct_bird_name(query)
        locality = next((l for l in LOCALITIES if l.lower() in query.lower()), None)
        
        if not locality or bird == "Unknown Bird":
            return jsonify({"error": "Missing required parameters"}), 400

        # Prepare input
        input_data = pd.DataFrame([[
            1,  # Assuming constant value from original code
            date.year,
            date.weekday(),
            time_encoders['LOCALITY'].transform([locality])[0],
            time_encoders['COMMON NAME'].transform([bird])[0],
            0, 0, 0, 0, 0, 0, 0, 0  # Placeholder for seasonal features
        ]], columns=time_features)
        
        best_month = month_model.predict(input_data)[0]
        best_hour = hour_model.predict(input_data)[0]
        
        return jsonify({
            "best_month": datetime.date(2000, int(best_month), 1).strftime("%B"),
            "best_hour": f"{int(best_hour % 12)} {'AM' if best_hour < 12 else 'PM'}",
            "bird": bird,
            "location": locality
        }), 200

    except Exception as e:
        logger.error(f"Time prediction error: {str(e)}")
        return jsonify({"error": "Time prediction failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)