import joblib
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from rapidfuzz import process

# ✅ Load the trained model and encoders
model_path = r'C:\Users\Deshan\Documents\IIT LECS\Year 2 Sem 1\DSGP\Git hub\FeatherFind\Migration model\models\migration_prediction_model.pkl'
model_data = joblib.load(model_path)

rf_model = model_data['rf_final']
selected_features = model_data['selected_features']
label_encoders = model_data['label_encoders']

# ✅ Locality and Bird Name Handling
valid_localities = [
    "Buckingham Place Hotel Tangalle", "Bundala NP General", "Bundala National Park", 
    "Kalametiya", "Tissa Lake", "Yala National Park General", "Debarawewa Lake"
]

valid_bird_names = ["Blue-tailed Bee-eater", "Red-vented Bulbul", "White-throated Kingfisher"]

bird_aliases = {
    "blue tailed bird": "Blue-tailed Bee-eater",
    "kingfisher": "White-throated Kingfisher",
    "bulbul": "Red-vented Bulbul"
}

# ✅ Function: Handle Bird Name Variations
def correct_bird_name(name):
    name = name.lower()
    if name in bird_aliases:
        return bird_aliases[name]

    matches = get_close_matches(name, [b.lower() for b in valid_bird_names], n=1, cutoff=0.3)
    if matches:
        return next(b for b in valid_bird_names if b.lower() == matches[0])  
    return "Unknown Bird"

# ✅ Function: Handle Locality Variations
def correct_locality(user_input):
    """
    Uses keyword-based mapping to handle user input more flexibly.
    """
    user_input = user_input.lower()

    # ✅ 1. Check for exact match first
    for loc in valid_localities:
        if user_input == loc.lower():
            return loc

    # ✅ 2. Check if user input is a **partial match** (keyword-based)
    for loc in valid_localities:
        if user_input in loc.lower():
            return loc  # Return the first match

    # ✅ 3. Handle some **manual keyword mappings** for better accuracy
    manual_mappings = {
        "bundala": "Bundala NP General",
        "yala": "Yala National Park General",
        "tissa": "Tissa Lake",
        "debara": "Debarawewa Lake",
        "kalametiya": "Kalametiya Bird Sanctuary"
    }
    
    for keyword, mapped_location in manual_mappings.items():
        if keyword in user_input:
            return mapped_location

    return "Unknown Location"  # No match found

# ✅ Function: Convert Day Name to Integer
def day_name_to_int(day_name):
    days_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
        "friday": 4, "saturday": 5, "sunday": 6
    }
    return days_map.get(day_name.lower(), None)

# ✅ Function: Convert Time of Day to Hour Range
def time_of_day_to_hour(time_str):
    time_ranges = {
        "morning": (6, 10), "afternoon": (11, 15),
        "evening": (16, 19), "night": (20, 23)
    }
    return time_ranges.get(time_str.lower(), None)

# ✅ Function: Extract Features from Query
def extract_query_features(query):
    query = query.lower()
    
    # Extract Year
    year_match = re.search(r'\b(20[0-9]{2})\b', query)
    year = int(year_match.group()) if year_match else None

    # Extract Month
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', query)
    month = months_map.get(month_match.group()) if month_match else None

    # Extract Day of Week
    day_name_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query)
    day_of_week = day_name_to_int(day_name_match.group()) if day_name_match else None

    # Extract Time (Hour)
    time_match = re.search(r'\b([0-9]{1,2}):?([0-9]{2})?\s?(am|pm)?\b', query)
    if time_match:
        hour = int(time_match.group(1))
        if time_match.group(3) == "pm" and hour < 12:
            hour += 12
    else:
        time_match = re.search(r'\b(morning|afternoon|evening|night)\b', query)
        hour_range = time_of_day_to_hour(time_match.group()) if time_match else None
        hour = hour_range[0] if hour_range else None

    # ✅ Extract Locality Using Improved Matching
    locality_match = re.search(r'\b[a-zA-Z\s]+\b', query)
    locality = correct_locality(locality_match.group()) if locality_match else "Unknown Location"

    # Extract Bird Name
    bird_name_match = re.search(r'\b(?:' + '|'.join([b.lower().replace("-", ".*") for b in valid_bird_names]) + r')\b', query)
    bird_name = correct_bird_name(bird_name_match.group()) if bird_name_match else "Unknown Bird"

    # ✅ Return an Error Dictionary Instead of `None`
    if None in [year, month, day_of_week, hour] or locality == "Unknown Location" or bird_name == "Unknown Bird":
        missing = [k for k, v in zip(["Year", "Month", "Day_of_Week", "Hour", "Locality", "Bird Name"],
                                     [year, month, day_of_week, hour, locality, bird_name]) if v is None or v.startswith("Unknown")]
        return {"error": f"Missing or incorrect inputs: {', '.join(missing)}"}

    return {
        "year": year,
        "month": month,
        "day_of_week": day_of_week,
        "hour": hour,
        "locality": locality,
        "bird_name": bird_name
    }

def generate_meaningful_query(result, original_query):
    """
    Converts the structured prediction result into a meaningful sentence.
    Ensures the time of day matches the original query.
    """
    if "error" in result:
        return result["error"]

    presence_text = "is likely to be present" if result["predicted_presence"] == 1 else "is unlikely to be present"
    
    # Convert day index back to string
    days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = days_map[result["features_used"]["day_of_week"]]

    # Extract original time phrase from query (morning, afternoon, evening, night)
    time_of_day_match = re.search(r'\b(morning|afternoon|evening|night)\b', original_query.lower())
    if time_of_day_match:
        time_of_day = time_of_day_match.group()  # Get the exact phrase
    else:
        # Convert hour into a more natural phrase if no explicit time was mentioned
        hour = result["features_used"]["hour"]
        if 6 <= hour <= 10:
            time_of_day = "in the morning"
        elif 11 <= hour <= 15:
            time_of_day = "in the afternoon"
        elif 16 <= hour <= 19:
            time_of_day = "in the evening"
        elif 20 <= hour <= 23:
            time_of_day = "at night"
        else:
            time_of_day = f"at {hour}:00"  # Show exact time if out of range

    probability = result["probability"] * 100  # Convert to percentage

    # Construct a meaningful sentence
    query_sentence = (
        f"The {result['features_used']['bird_name']} {presence_text} "
        f"at {result['features_used']['locality']} on {day_name}, {result['features_used']['month']}/"
        f"{result['features_used']['year']} {time_of_day}. (Confidence: {probability:.1f}%)"
    )

    return query_sentence

# ✅ Function: Predict Bird Presence
def predict_bird_presence(query):
    features = extract_query_features(query)
    
    # ✅ Ensure `features` is a dictionary
    if not isinstance(features, dict):
        return {"error": "Failed to extract features. Please check your query format."}

    # ✅ If the extracted features contain errors, return immediately
    if "error" in features:
        return features

    # ✅ Handle Unknown Locality
    if features["locality"] == "Unknown Location":
        return {"error": "The locality entered is not recognized. Please provide a more specific location."}
    
    # ✅ Handle Unknown Bird Name
    if features["bird_name"] == "Unknown Bird":
        return {"error": "The bird name entered is not recognized. Please check the spelling or try another bird."}

    try:
        # Encode locality and bird name
        locality_encoded = label_encoders['LOCALITY'].transform([features["locality"]])[0]
        bird_name_encoded = label_encoders['COMMON NAME'].transform([features["bird_name"]])[0]

        # Prepare input data
        input_data = pd.DataFrame([[features["year"], features["month"], features["day_of_week"],
                                     features["hour"], locality_encoded, bird_name_encoded]],
                                  columns=selected_features)

        # Make prediction
        probability = rf_model.predict_proba(input_data)[:, 1][0]
        prediction = int(probability >= 0.5)

        # Structure the result
        result = {
            "query": query,
            "predicted_presence": prediction,
            "probability": round(probability, 3),
            "features_used": features
        }

        return generate_meaningful_query(result, query)  # Pass original query

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
