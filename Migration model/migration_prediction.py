import pandas as pd
import joblib

# ✅ Load Trained Models
migration_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl")
location_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl")
time_models = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl")

def prepare_input_for_model(user_input, model_type):
    """
    Prepares user input based on model type by selecting only required features.
    """
    input_df = pd.DataFrame([user_input])

    # ✅ Define expected feature sets
    if model_type == "migration":
        required_features = migration_model.feature_names_in_
    elif model_type == "location":
        required_features = location_model.feature_names_in_
    elif model_type == "time":
        return user_input  # No preprocessing needed for time models

    # ✅ Ensure only trained features are passed to model
    input_df = input_df[required_features]

    return input_df

# ✅ Migration Prediction
def predict_migration(user_input):
    """
    Predicts migration probability given user input.
    """
    input_features = prepare_input_for_model(user_input, "migration")
    probability = migration_model.predict_proba(input_features)[:, 1]  # Get probability
    return {"migration_probability": probability[0]}

# ✅ Location Prediction
def predict_location(user_input):
    """
    Predicts possible bird species for given location and time.
    """
    input_features = prepare_input_for_model(user_input, "location")
    probabilities = location_model.predict_proba(input_features)[:, 1]
    
    species_list = location_model.classes_  # Get species names
    sorted_species = sorted(zip(species_list, probabilities), key=lambda x: x[1], reverse=True)

    top_species = [f"{species} (Probability: {prob:.2f})" for species, prob in sorted_species[:5]]

    return {"top_species_predictions": top_species}

# ✅ Time Prediction
def predict_best_time(user_input):
    """
    Predicts best observation time for a given location.
    """
    location_encoded = time_models['locality_encoder'].transform([user_input["location"]])[0]
    
    best_month = time_models['month_model'].predict([[location_encoded]])[0]
    best_day = time_models['day_model'].predict([[location_encoded]])[0]
    best_hour = time_models['hour_model'].predict([[location_encoded]])[0]

    return {"best_time": {"month": best_month, "day": best_day, "hour": best_hour}}

# ✅ Unified Prediction Function for Chatbot
def process_user_query(user_input):
    """
    Determines the type of user query and returns relevant predictions.
    """
    response = {}

    # User asks for migration probability
    if "Year" in user_input and "LATITUDE" in user_input and "LONGITUDE" in user_input:
        response.update(predict_migration(user_input))

    # User asks for bird species at a location
    if "COMMON NAME_Blue-tailed Bee-eater" in user_input or "COMMON NAME_Red-vented Bulbul" in user_input:
        response.update(predict_location(user_input))

    # User asks for best observation time
    if "location" in user_input:
        response.update(predict_best_time(user_input))

    return response

# ✅ Example Input Testing
if __name__ == "__main__":
    test_input = {
        "Year": 2025, "Month": 3, "Day": 12, "Day_of_Week": 2, "Hour": 7,
        "LATITUDE": 6.287, "LONGITUDE": 81.261,
        "COUNTY_Angunakolapelessa": 0, "COUNTY_Hambantota": 1,
        "LOCALITY_Yala": 1, "COMMON NAME_Blue-tailed Bee-eater": 1
    }

    result = process_user_query(test_input)
    print("Prediction Result:", result)
