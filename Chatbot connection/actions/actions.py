import requests
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# ✅ API Endpoints
RANGE_PREDICTION_API = "http://127.0.0.1:5002/predict_presence"
LOCATION_API = "http://127.0.0.1:5002/predict_location"
TIME_PREDICTION_API = "http://127.0.0.1:5002/predict_best_time"

# ✅ Function: Call Range Prediction API
def handle_range_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(RANGE_PREDICTION_API, json=payload, headers=headers)
        json_response = response.json()
        
        HAMBANTHOTA_LOCATIONS = [
            "Bundala National Park", "Kalametiya", "Tissa Lake", "Yala National Park General",
            "Debarawewa Lake", "Bundala NP General", "Bundala Freshwater Marsh", "Yoda Lake",
            "Kalametiya Bird Sanctuary", "Thangalle Marsh", "Hibiscus Garden Hotel Tissamaharama",
            "Senasuma Wetland", "Pannegamuwa Lake", "Buckingham Place Hotel Tangalle",
            "Weliaragoda Wetland", "Pallemalala Wewa", "Wirawila", "Bandagiriya Southern Province",
            "Palatupana", "Palatupana Wetland", "Gal Wala Home Walasmulla Southern",
            "Kalamatiya Sanctuary", "Palatupana Southern Province", "Ampitiya Lake Beliatta Southern Province",
            "Yoda Kandiya Tank", "Godakalapuwa Ruhuna NP", "Lake View Cottage Tissamaharama",
            "Sithulpawwa", "Road Weligatta Southern Province", "Karagan Lewaya Hambanthota"
        ]


        message = json_response.get("message", "")

        # ✅ Combine with locations if available
        if "you can use these locations" in json_response or "you_can_use_these_locations" in json_response or "location_error" in message.lower():
            locations_text = "\n".join(HAMBANTHOTA_LOCATIONS)
            full_message = f"{message}\n\nValid Locations:\n{locations_text}"
            dispatcher.utter_message(text=full_message)
            return


        if "valid_bird_names" in json_response or "bird species" in message.lower():
            VALID_BIRD_SPECIES = [
                "Blue-tailed Bee-eater",
                "Red-vented Bulbul",
                "White-throated Kingfisher"
            ]
            birds_text = "\n".join(VALID_BIRD_SPECIES)
            dispatcher.utter_message(
                text=f"{message}\n\nValid Bird Species:\n{birds_text}"
            )
            return


        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the range prediction API.")



# ✅ Function: Call Location Prediction API
def handle_location_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(LOCATION_API, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(f"❌ API Error: {response.status_code} - {response.text}")
            dispatcher.utter_message(text="There was an error processing your location prediction request.")
            return

        json_response = response.json()
        
        message = json_response.get("message", "")


        # ✅ Handle bird species not found
        if "valid_bird_names" in json_response or "bird species" in message.lower():
            VALID_BIRD_SPECIES = [
                "Blue-tailed Bee-eater",
                "Red-vented Bulbul",
                "White-throated Kingfisher"
            ]
            birds_text = "\n".join(VALID_BIRD_SPECIES)
            dispatcher.utter_message(
                text=f"{message}\n\nValid Bird Species:\n{birds_text}"
            )
            return

        # ✅ General response
        dispatcher.utter_message(
            text=json_response.get("Response for you", "I couldn't generate a response.")
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(
            text="There was an error connecting to the location prediction API."
        )


# ✅ Function: Call Time Prediction API
def handle_time_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(TIME_PREDICTION_API, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(f"❌ API Error: {response.status_code} - {response.text}")
            dispatcher.utter_message(text="There was an error processing your request.")
            return

        json_response = response.json()

        # ✅ Handle location validation message + list
        HAMBANTHOTA_LOCATIONS = [
            "Bundala National Park", "Kalametiya", "Tissa Lake", "Yala National Park General",
            "Debarawewa Lake", "Bundala NP General", "Bundala Freshwater Marsh", "Yoda Lake",
            "Kalametiya Bird Sanctuary", "Thangalle Marsh", "Hibiscus Garden Hotel Tissamaharama",
            "Senasuma Wetland", "Pannegamuwa Lake", "Buckingham Place Hotel Tangalle",
            "Weliaragoda Wetland", "Pallemalala Wewa", "Wirawila", "Bandagiriya Southern Province",
            "Palatupana", "Palatupana Wetland", "Gal Wala Home Walasmulla Southern",
            "Kalamatiya Sanctuary", "Palatupana Southern Province", "Ampitiya Lake Beliatta Southern Province",
            "Yoda Kandiya Tank", "Godakalapuwa Ruhuna NP", "Lake View Cottage Tissamaharama",
            "Sithulpawwa", "Road Weligatta Southern Province", "Karagan Lewaya Hambanthota"
        ]


        message = json_response.get("message", "")

        # ✅ Combine with locations if available
        if "you can use these locations" in json_response or "you_can_use_these_locations" in json_response or "location_error" in message.lower():
            locations_text = "\n".join(HAMBANTHOTA_LOCATIONS)
            full_message = f"{message}\n\nValid Locations:\n{locations_text}"
            dispatcher.utter_message(text=full_message)
            return


        if "valid_bird_names" in json_response or "bird species" in message.lower():
            VALID_BIRD_SPECIES = [
                "Blue-tailed Bee-eater",
                "Red-vented Bulbul",
                "White-throated Kingfisher"
            ]
            birds_text = "\n".join(VALID_BIRD_SPECIES)
            dispatcher.utter_message(
                text=f"{message}\n\nValid Bird Species:\n{birds_text}"
            )
            return

        # ✅ Normal case: show prediction
        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the time prediction API.")


# ✅ Keyword Routing Logic
def determine_api_from_query(query: str) -> str:
    query = query.lower()

    time_keywords = ["when", "best time", "morning", "afternoon", "evening", "night", "what time", "hour", "summer", "winter", "spring", "autmn"]
    
    location_keywords = ["where", "locations", "spots", "places", "areas",
                         "district", "blue bird", "Blue tailed bird", "when", "When", "Where" ,"Where can I find", "find",]
    
    presence_keywords = ["present", "see", "appear", "found", "visible", "will I", "can i"]

    if any(word in query for word in time_keywords):
        return "time"
    elif any(word in query for word in location_keywords):
        return "location"
    elif any(word in query for word in presence_keywords):
        return "presence"
    else:
        return "presence"  # Default fallback

# ✅ Rasa Action Class
class ActionHandleBirdPrediction(Action):
    def name(self) -> str:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 Received user query: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Can you rephrase it?")
            return []

        selected_api = determine_api_from_query(user_query)
        logger.info(f"📡 Routing query to: {selected_api.upper()} API")

        if selected_api == "time":
            handle_time_prediction(user_query, dispatcher)
        elif selected_api == "location":
            handle_location_prediction(user_query, dispatcher)
        else:
            handle_range_prediction(user_query, dispatcher)

        return []
