import joblib
import numpy as np
import pandas as pd

# Load Migration Prediction Model (Model A)
migration_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl')

# Load Location Prediction Model (Model B)
location_model = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl')

# Load Time Prediction Model (Model C) (Contains multiple models)
time_models = joblib.load(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl')

# Extract components from Time Model
month_model = time_models['month_model']
hour_model = time_models['hour_model']
county_encoder = time_models['county_encoder']
locality_encoder = time_models['locality_encoder']
selected_features_time = time_models['selected_features']

print("✅ All models loaded successfully!")


def predict_bird_migration(user_input, df):
    """
    Predicts bird migration presence, possible locations, and best observation times using an interdependent model pipeline.

    Args:
    - user_input (dict): Contains species, location, date, time.
    - df (DataFrame): The processed dataset with historical bird observations.

    Returns:
    - dict: Structured prediction results with explanations.
    """
    # Extract user inputs
    species = user_input.get("species", None)
    location = user_input.get("location", None)
    date = user_input.get("date", None)
    time = user_input.get("time", None)

    # Convert date and time
    if date and time:
        datetime_query = pd.to_datetime(f"{date} {time}", errors="coerce")
        year, month, day, hour, day_of_week = (
            datetime_query.year, datetime_query.month, datetime_query.day, datetime_query.hour, datetime_query.dayofweek
        )
    else:
        return {"error": "Please provide a valid date and time."}

    # **Step 1: Predict Species Presence (Model A)**
    print(f"🔍 Checking if {species} is likely to be present on {date} at {time}...")
    
    # Create feature input for Model A
    feature_migration = {
        "Year": year, "Day": day, "Day_of_Week": day_of_week, "Hour": hour,
        "COUNTY_ENCODED": county_encoder.transform([user_input.get("county", "Unknown")])[0] if "county" in user_input else 0,
        "LOCALITY_ENCODED": locality_encoder.transform([user_input.get("locality", "Unknown")])[0] if "locality" in user_input else 0,
    }
    
    # One-hot encode species
    species_column = f"COMMON NAME_{species}"
    for col in ["COMMON NAME_Blue-tailed Bee-eater", "COMMON NAME_Red-vented Bulbul", "COMMON NAME_White-throated Kingfisher"]:
        feature_migration[col] = 1 if col == species_column else 0

    # Convert to DataFrame
    feature_df = pd.DataFrame([feature_migration])
    
    # Make migration prediction
    migration_prob = migration_model.predict_proba(feature_df)[:, 1][0]
    
    if migration_prob < 0.5:
        return {"result": f"❌ {species} is unlikely to be observed at this time."}
    
    print(f"✅ {species} is likely present! (Probability: {migration_prob:.2f})")
    
    # **Step 2: Predict Possible Locations (Model B)**
    print(f"📍 Predicting possible locations for {species}...")
    
    # Create feature input for Model B
    feature_location = feature_migration.copy()
    feature_location.update({
        "Month": month,
        "LATITUDE": user_input.get("latitude", 0),
        "LONGITUDE": user_input.get("longitude", 0),
    })
    
    # Convert to DataFrame
    feature_df_location = pd.DataFrame([feature_location])
    
    # Predict locations
    location_prob = location_model.predict_proba(feature_df_location)[:, 1]
    
    # Select top-N locations
    df["Predicted_Location_Probability"] = location_prob
    top_locations = df.groupby("LOCALITY").mean()["Predicted_Location_Probability"].sort_values(ascending=False).head(3)
    
    print(f"🏞 Best Locations: {list(top_locations.index)}")

    # **Step 3: Predict Best Time (Model C)**
    print(f"⏳ Predicting the best time to observe {species} in {list(top_locations.index)[0]}...")
    
    # Create feature input for Model C
    feature_time = {
        "OBSERVATION": 1,
        "Year": year,
        "Day_of_Week": day_of_week,
        "LOCALITY_encoded": locality_encoder.transform([list(top_locations.index)[0]])[0] if list(top_locations.index) else 0,
        "Is_Summer": int(month in [6, 7, 8]),
        "Is_Winter": int(month in [12, 1, 2]),
        "Is_Spring": int(month in [3, 4, 5]),
        "Is_Autumn": int(month in [9, 10, 11]),
        "Is_Morning": int(hour in range(6, 12)),
        "Is_Afternoon": int(hour in range(12, 18)),
        "Is_Evening": int(hour in range(18, 24)),
        "Is_Night": int(hour in range(0, 6)),
    }
    
    # Predict Month, Day, Hour
    best_month = month_model.predict(pd.DataFrame([feature_time]))[0]
    best_hour = hour_model.predict(pd.DataFrame([feature_time]))[0]
    
    print(f"📆 Best Observation Month: {best_month}, Best Hour: {best_hour}")

    # **Final Structured Response**
    response = {
        "Species": species,
        "Observation Probability": f"{migration_prob*100:.1f}%",
        "Best Locations": {loc: f"{prob*100:.1f}%" for loc, prob in top_locations.items()},
        "Best Observation Month": best_month,
        "Best Observation Hour": best_hour,
    }
    
    return response
