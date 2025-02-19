import joblib
import pandas as pd
import numpy as np
from fuzzywuzzy import process

# Load Models
migration_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl")
location_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl")
time_model_data = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl")

# Extract models from time prediction model
month_model = time_model_data['month_model']
day_model = time_model_data['day_model']
hour_model = time_model_data['hour_model']
county_encoder = time_model_data['county_encoder']
locality_encoder = time_model_data['locality_encoder']

# Load Datasets
df_migration = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\migration_data.csv")
df_location = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\location_data.csv")
df_time = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\time_data.csv")

# Extract feature names
species_list = [col.replace("COMMON NAME_", "") for col in df_migration.columns if col.startswith("COMMON NAME_")]
county_list = [col.replace("COUNTY_", "") for col in df_migration.columns if col.startswith("COUNTY_")]
locality_list = [col.replace("LOCALITY_", "") for col in df_migration.columns if col.startswith("LOCALITY_")]

# Expected feature names
migration_expected_features = getattr(migration_model, "feature_names_in_", df_migration.columns)
location_expected_features = getattr(location_model, "feature_names_in_", df_location.columns)

# Fuzzy Matching Helper Function
def fuzzy_match(user_input, choices):
    if not user_input:
        return None
    match, score = process.extractOne(user_input, choices)
    return match if score > 80 else None  # Accept matches above 80% confidence

def interpret_migration_prediction(user_input):
    """
    Processes user queries and determines the type of prediction required.

    Args:
    - user_input (dict): User query parameters.

    Returns:
    - str: Meaningful natural language response.
    """
    # Extract user inputs
    species = user_input.get("species", None)
    location = user_input.get("location", None)
    county = user_input.get("county", None)
    date = user_input.get("date", None)
    time = user_input.get("time", None)
    latitude = user_input.get("LATITUDE", None)
    longitude = user_input.get("LONGITUDE", None)

    # ✅ Handle Typos using Fuzzy Matching
    if species:
        species = fuzzy_match(species, species_list)
    if location:
        location = fuzzy_match(location, locality_list)
    if county:
        county = fuzzy_match(county, county_list)

    # ✅ Convert Date & Time
    if date and time:
        datetime_query = pd.to_datetime(f"{date} {time}", errors="coerce")
        year, month, day, hour, day_of_week = (
            datetime_query.year, datetime_query.month, datetime_query.day, datetime_query.hour, datetime_query.dayofweek
        )
    else:
        return "Please provide a valid date and time."

    # ✅ Establish Inter-Model Dependencies
    if species:
        print(f"Checking presence of species: {species}")

        if f"COMMON NAME_{species}" not in df_migration.columns:
            return f"Sorry, we don't have migration data for {species}."

        # Prepare input data for migration model
        migration_input = {feature: 0 for feature in migration_expected_features}
        migration_input.update({
            "Year": year,
            "Day": day,
            "Day_of_Week": day_of_week,
            "Hour": hour,
        })
        migration_input[f"COMMON NAME_{species}"] = 1

        # Convert to DataFrame and Predict
        migration_df = pd.DataFrame([migration_input])[migration_expected_features]
        species_presence = migration_model.predict(migration_df)[0]

        # If species is not predicted to be present, stop further predictions
        if species_presence == 0:
            return f"The {species} is **not expected to be present** on {date} at {time}. Try a different species or time."

    # ✅ If species is present, predict location
    if species and species_presence == 1:
        print(f"Predicting locations for species: {species}")

        # Prepare input for location model
        location_input = {feature: 0 for feature in location_expected_features}
        location_input.update({
            "Year": year,
            "Month": month,
            "Day": day,
            "Day_of_Week": day_of_week,
            "Hour": hour,
        })
        if latitude is not None:
            location_input["LATITUDE"] = latitude
        if longitude is not None:
            location_input["LONGITUDE"] = longitude
        location_input[f"COMMON NAME_{species}"] = 1

        # Convert to DataFrame and Predict Locations
        location_df = pd.DataFrame([location_input])[location_expected_features]
        location_predictions = location_model.predict(location_df)

        best_location = fuzzy_match(location_predictions[0], locality_list)  # Get closest matching location

        if best_location:
            print(f"Predicted best location: {best_location}")
        else:
            return f"No suitable location found for {species}."

    # ✅ If species & location are found, predict best time
    if species and best_location:
        print(f"Predicting best time for {species} at {best_location}")

        # Encode location or county
        input_data = county_encoder.transform([[county]]) if county else locality_encoder.transform([[best_location]])
        predicted_month = month_model.predict(input_data)
        predicted_day = day_model.predict(input_data)
        predicted_hour = hour_model.predict(input_data)

        # Generate Response
        response = f"📌 The **{species}** is **expected to be present** on {date} at {time}.\n"
        response += f"📍 **Best Location:** {best_location}\n"
        response += f"⏳ **Best Time to Spot:**\n"
        response += f"- **Month:** {predicted_month[0]}\n"
        response += f"- **Day:** {predicted_day[0]}\n"
        response += f"- **Hour:** {predicted_hour[0]}\n"

        return response

    return "Invalid query. Please provide species, location, or date/time."

