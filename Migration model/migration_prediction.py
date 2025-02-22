import joblib
import numpy as np
import pandas as pd
import difflib

# Load Models
migration_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl')
location_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl')
time_model_data = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl')

month_model = time_model_data['month_model']
hour_model = time_model_data['hour_model']
county_encoder = time_model_data['county_encoder']
locality_encoder = time_model_data['locality_encoder']
selected_features = time_model_data['selected_features']

# Define expected inputs for each model
migration_features = ['Year', 'Day', 'Day_of_Week', 'Hour', 'COMMON NAME_', 'COUNTY_', 'LOCALITY_']
location_features = ['Year', 'Month', 'Day', 'Day_of_Week', 'Hour', 'LATITUDE', 'LONGITUDE', 'COMMON NAME_']
time_features = ['OBSERVATION', 'Year', 'Day_of_Week', 'LOCALITY_encoded', 'Is_Summer', 'Is_Winter', 'Is_Spring', 
                 'Is_Autumn', 'Is_Morning', 'Is_Afternoon', 'Is_Evening', 'Is_Night', 'COMMON NAME_']

# Define keyword mappings for error handling
keyword_mapping = {
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "day_of_week": "Day_of_Week",
    "hour": "Hour",
    "lat": "LATITUDE",
    "long": "LONGITUDE",
    "species": "COMMON NAME_",
    "county": "COUNTY_",
    "locality": "LOCALITY_",
    "obs": "OBSERVATION"
}

# Function to correct user input keys
def correct_keys(user_inputs):
    corrected_inputs = {}
    for key, value in user_inputs.items():
        closest_match = difflib.get_close_matches(key.lower(), keyword_mapping.keys(), n=1)
        if closest_match:
            corrected_inputs[keyword_mapping[closest_match[0]]] = value
        else:
            corrected_inputs[key] = value
    return corrected_inputs

# Function to preprocess user input for the models
def preprocess_input(user_inputs, required_features):
    """
    Ensure the user input matches the required feature format.
    Handles missing values, applies encoding, and ensures all necessary features exist.
    """
    user_inputs = correct_keys(user_inputs)
    processed_input = {}

    for feature in required_features:
        if feature.startswith("COMMON NAME_"):
            if 'COMMON NAME_' in user_inputs:
                processed_input[feature] = 1 if feature.endswith(user_inputs['COMMON NAME_']) else 0
            else:
                processed_input[feature] = 0
        elif feature.startswith("COUNTY_"):
            processed_input[feature] = 1 if user_inputs.get("COUNTY_", "").lower() in feature.lower() else 0
        elif feature.startswith("LOCALITY_"):
            processed_input[feature] = 1 if user_inputs.get("LOCALITY_", "").lower() in feature.lower() else 0
        else:
            processed_input[feature] = user_inputs.get(feature, 0)  # Default missing values to 0

    return np.array([list(processed_input.values())])  # Convert to NumPy array

# Function to generate meaningful chatbot responses
def generate_response(predictions, user_inputs):
    """
    Translates model predictions into human-friendly text.
    """
    responses = []

    if 'COMMON NAME_' in user_inputs:
        species = user_inputs['COMMON NAME_']
    else:
        species = "this species"

    if 'LOCALITY_' in user_inputs:
        location = user_inputs['LOCALITY_']
    else:
        location = "the given area"

    if 'Year' in user_inputs and 'Month' in user_inputs and 'Day' in user_inputs:
        date_info = f"on {user_inputs['Year']}-{user_inputs['Month']}-{user_inputs['Day']}"
    else:
        date_info = "on the given date"

    if predictions.get('migration'):
        if predictions['migration'] == 1:
            responses.append(f"✅ The {species} is likely to be present {date_info}.")
        else:
            responses.append(f"❌ The {species} is unlikely to be seen {date_info}.")
    
    if predictions.get('location'):
        responses.append(f"📍 The best locations to spot {species} are: {', '.join(predictions['location'])}.")
    
    if predictions.get('time'):
        responses.append(f"⏰ The best time to observe {species} is at {predictions['time']} hours.")
    
    return " ".join(responses)

# Function to make predictions using the three models
def predict_migration(user_inputs):
    """
    Main function to predict bird migration presence, locations, and best times.
    """
    predictions = {}

    # Step 1: Migration Model (Model A) - Predict presence
    migration_input = preprocess_input(user_inputs, migration_features)
    migration_prob = migration_model.predict_proba(migration_input)[:, 1][0]
    
    threshold = 0.274  # Optimal probability threshold
    predictions['migration'] = 1 if migration_prob >= threshold else 0

    if predictions['migration'] == 1:
        # Step 2: Location Model (Model B) - Predict locations
        location_input = preprocess_input(user_inputs, location_features)
        location_pred = location_model.predict(location_input)

        top_locations = [f"Location_{i}" for i, val in enumerate(location_pred[0]) if val == 1]
        predictions['location'] = top_locations[:3]  # Return top 3 locations

        # Step 3: Time Model (Model C) - Predict best time
        if 'LOCALITY_' in user_inputs:
            time_input = preprocess_input(user_inputs, time_features)
            best_hour = hour_model.predict(time_input)[0]
            predictions['time'] = best_hour

    return generate_response(predictions, user_inputs)

# Example user input
user_query = {
    "species": "Red-vented Bulbul",
    "Year": 2025,
    "Month": 3,
    "Day": 15,
    "Day_of_Week": 2,
    "Hour": 10,
    "LOCALITY_": "Bundala National Park"
}

# Run the prediction
response = predict_migration(user_query)
print(response)
