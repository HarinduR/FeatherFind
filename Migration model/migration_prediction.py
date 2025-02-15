import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/migration_prediction_model.pkl")

def interpret_migration_prediction(user_input, df):
    """
    Interprets user queries and retrieves meaningful migration predictions.

    Args:
    - user_input (dict): Contains user query parameters (species, location, date, time, etc.).
    - df (DataFrame): Processed dataset with historical bird observations.

    Returns:
    - str: Natural language response for chatbot.
    """

    # Extract user input
    species = user_input.get("species", None)
    location = user_input.get("location", None)
    date = user_input.get("date", None)
    time = user_input.get("time", None)

    # Convert date and time to proper format
    if date and time:
        datetime_query = pd.to_datetime(f"{date} {time}", errors="coerce")
        year, month, day, hour, day_of_week = (
            datetime_query.year, datetime_query.month, datetime_query.day, datetime_query.hour, datetime_query.dayofweek
        )
    else:
        return "Please provide a valid date and time."

    # Determine Query Type
    if species and date and time:
        query_type = "location"
    elif location and date and time:
        query_type = "species"
    elif date and time:
        query_type = "full"
    else:
        return "Invalid query. Please provide species or location with date and time."

    # Prepare features for model prediction
    feature_dict = {
        "Year": year, "Month": month, "Day": day, "Day_of_Week": day_of_week, "Hour": hour
    }

    # Handle Location-Based Prediction
    if query_type == "location":
        print(f"Predicting locations for species: {species} on {date} at {time}")

        # Convert species to one-hot encoding
        species_column = f"COMMON NAME_{species}"
        if species_column not in df.columns:
            return f"Sorry, migration data for {species} is not available."

        # Filter dataset for the given species
        species_data = df[df[species_column] == 1]

        # Predict migration likelihood for different locations
        probabilities = model.predict_proba(species_data.drop(columns=["OBSERVATION"]))[:, 1]

        # Attach probabilities to locations
        species_data["Migration Probability"] = probabilities
        top_locations = species_data.groupby(["LOCALITY"]).mean()["Migration Probability"].sort_values(ascending=False).head(3)

        # Convert to Natural Language Response
        response = f"The *{species}* is most likely to be found at:\n"
        for loc, prob in top_locations.items():
            response += f"- **{loc}** with a {prob*100:.1f}% chance\n"

        return response

    # Handle Species-Based Prediction
    elif query_type == "species":
        print(f"Predicting species for location: {location} on {date} at {time}")

        # Filter dataset for the given location
        location_data = df[df["LOCALITY_" + location] == 1]

        # Predict species likelihood
        probabilities = model.predict_proba(location_data.drop(columns=["OBSERVATION"]))[:, 1]

        # Attach probabilities to species
        species_probs = {col.replace("COMMON NAME_", ""): prob for col, prob in zip(df.columns, probabilities) if "COMMON NAME_" in col}

        # Sort species by probability
        sorted_species = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        # Convert to Natural Language Response
        response = f"At **{location}** on **{date} at {time}**, you are likely to see:\n"
        for spec, prob in sorted_species:
            response += f"- **{spec}** with a {prob*100:.1f}% chance\n"

        return response

    # Handle Full Prediction
    elif query_type == "full":
        print(f"Predicting both species and locations for {date} at {time}")

        # Predict locations
        top_locations = df.groupby("LOCALITY")["OBSERVATION"].mean().sort_values(ascending=False).head(3)

        # Predict species
        species_probs = {col.replace("COMMON NAME_", ""): df[col].mean() for col in df.columns if "COMMON NAME_" in col}
        sorted_species = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        # Convert to Natural Language Response
        response = f"On **{date} at {time}**, the best locations for birdwatching are:\n"
        for loc, prob in top_locations.items():
            response += f"- **{loc}** ({prob*100:.1f}% probability)\n"

        response += "\nYou may spot these birds:\n"
        for spec, prob in sorted_species:
            response += f"- **{spec}** ({prob*100:.1f}% probability)\n"

        return response

    return "Error in processing query."
