import joblib
import pandas as pd
import numpy as np
from fuzzywuzzy import process

# Load Models
migration_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl")
location_model = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl")
time_model_data = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl")

# Extract time models and encoders
month_model = time_model_data['month_model']
day_model = time_model_data['day_model']
hour_model = time_model_data['hour_model']
county_encoder = time_model_data['county_encoder']
locality_encoder = time_model_data['locality_encoder']

# Load Data (used for interpretation)
df_migration = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\migration_data.csv")
df_location = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\location_data.csv")
df_time = pd.read_csv(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\time_data.csv")

# Extract feature names from dataset
species_list = [col.replace("COMMON NAME_", "") for col in df_migration.columns if col.startswith("COMMON NAME_")]
county_list = [col.replace("COUNTY_", "") for col in df_migration.columns if col.startswith("COUNTY_")]
locality_list = [col.replace("LOCALITY_", "") for col in df_migration.columns if col.startswith("LOCALITY_")]

# Get expected feature names from trained models
migration_expected_features = getattr(migration_model, "feature_names_in_", df_migration.columns)
location_expected_features = getattr(location_model, "feature_names_in_", df_location.columns)

# Helper function for fuzzy matching (handling typos and similar names)
def fuzzy_match(user_input, choices):
    match, score = process.extractOne(user_input, choices)
    return match if score > 80 else None

def interpret_migration_prediction(user_input):
    """
    Interprets user queries and retrieves meaningful migration predictions.

    Args:
    - user_input (dict): Contains user query parameters (species, location, date, time, etc.).

    Returns:
    - str: Natural language response for chatbot.
    """
    # Extract user input
    species = user_input.get("species", None)
    location = user_input.get("location", None)
    county = user_input.get("county", None)
    date = user_input.get("date", None)
    time = user_input.get("time", None)
    latitude = user_input.get("LATITUDE", None)
    longitude = user_input.get("LONGITUDE", None)

    # ✅ Handle Misspellings using Fuzzy Matching
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

    # ✅ Determine Query Type
    query_type = None
    if species and date and time:
        query_type = "location"
    elif location and date and time:
        query_type = "species"
    elif date and time:
        query_type = "full"
    elif species and location:
        query_type = "time"

    # ✅ Handle Query Types

    ### **1️⃣ Species-based Location Prediction**
    if query_type == "location":
        print(f"Predicting locations for species: {species} on {date} at {time}")

        if f"COMMON NAME_{species}" not in df_migration.columns:
            return f"Sorry, migration data for {species} is not available."

        # ✅ Ensure input data has all required features
        input_data = {feature: 0 for feature in location_expected_features}
        input_data.update({
            "Year": year,
            "Month": month,
            "Day": day,
            "Day_of_Week": day_of_week,
            "Hour": hour,
        })
        if latitude is not None:
            input_data["LATITUDE"] = latitude
        if longitude is not None:
            input_data["LONGITUDE"] = longitude

        # ✅ Mark the species presence
        input_data[f"COMMON NAME_{species}"] = 1

        # ✅ Convert to DataFrame
        input_df = pd.DataFrame([input_data])[location_expected_features]

        # ✅ Predict possible locations
        location_probs = location_model.predict_proba(input_df)[:, 1]
        location_scores = dict(zip(locality_list, location_probs))
        top_locations = sorted(location_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        response = f"The *{species}* is most likely to be found at:\n"
        for loc, prob in top_locations:
            response += f"- **{loc}**\n"
        
        return response

    ### **2️⃣ Location-based Species Prediction**
    elif query_type == "species":
        print(f"Predicting species for location: {location} on {date} at {time}")

        if f"LOCALITY_{location}" not in df_migration.columns:
            return f"Sorry, we don't have data for {location}."

        # ✅ Ensure input data has all required features
        input_data = {feature: 0 for feature in migration_expected_features}
        input_data.update({
            "Year": year,
            "Month": month,
            "Day": day,
            "Day_of_Week": day_of_week,
            "Hour": hour,
        })
        input_data[f"LOCALITY_{location}"] = 1

        # ✅ Convert to DataFrame
        input_df = pd.DataFrame([input_data])[migration_expected_features]

        # ✅ Predict possible species
        species_probs = migration_model.predict_proba(input_df)[:, 1]
        species_scores = dict(zip(species_list, species_probs))
        top_species = sorted(species_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        response = f"At **{location}** on **{date} at {time}**, you are likely to see:\n"
        for spec, prob in top_species:
            response += f"- **{spec}**\n"

        return response

    ### **3️⃣ Full Prediction (Species & Locations)**
    elif query_type == "full":
        print(f"Predicting both species and locations for {date} at {time}")

        top_locations = df_migration.groupby("LOCALITY")["OBSERVATION"].mean().sort_values(ascending=False).head(3)
        top_species = df_migration.groupby("COMMON NAME")["OBSERVATION"].mean().sort_values(ascending=False).head(3)

        response = f"On **{date} at {time}**, the best locations for birdwatching are:\n"
        for loc in top_locations.index:
            response += f"- **{loc}**\n"

        response += "\nYou may spot these birds:\n"
        for spec in top_species.index:
            response += f"- **{spec}**\n"

        return response

    ### **4️⃣ Time-Based Prediction**
    elif query_type == "time":
        print(f"Predicting best times for {species} at {location}")

        input_data = county_encoder.transform([[county]]) if county else locality_encoder.transform([[location]])
        predicted_month = month_model.predict(input_data)
        predicted_day = day_model.predict(input_data)
        predicted_hour = hour_model.predict(input_data)

        response = f"The best time to see **{species}** at **{location}** is:\n"
        response += f"- **Month:** {predicted_month[0]}\n"
        response += f"- **Day:** {predicted_day[0]}\n"
        response += f"- **Hour:** {predicted_hour[0]}\n"

        return response

    return "Invalid query. Please provide species, location, or date/time."
