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

        # ✅ Show full message from API (with inline locations already included)
        if "message" in json_response:
            dispatcher.utter_message(text=json_response["message"])
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
        json_response = response.json()

        # ✅ Show full message from API (with inline locations already included)
        if "message" in json_response:
            dispatcher.utter_message(text=json_response["message"])
            return

        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the range prediction API.")


# ✅ Function: Call Time Prediction API
def handle_time_prediction(query, dispatcher):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(TIME_PREDICTION_API, json=payload, headers=headers)

        try:
            json_response = response.json()
        except ValueError:
            dispatcher.utter_message(text="⚠️ Sorry, something went wrong while processing the response.")
            return

        # ✅ If API returned a message (even on error)
        if "message" in json_response:
            dispatcher.utter_message(text=json_response["message"])
            return

        # ✅ Else show the normal successful response
        dispatcher.utter_message(text=json_response.get("Response", "I couldn't generate a response."))

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API call error: {e}")
        dispatcher.utter_message(text="There was an error connecting to the time prediction API.")



# ✅ Keyword Routing Logic
def determine_api_from_query(query: str) -> str:
    query = query.lower()

    time_keywords = ["what", "time", "good time", "when", "best time", "what time", "hour", "summer", "winter", "spring", "autumn"]
    
    location_keywords = ["where", "locations", "spots", "places", "areas",
                         "district", "blue bird", "Blue tailed bird", "when", "When", "Where" ,"Where can I find", "find",]
    
    presence_keywords = ["present", "see", "appear", "found", "visible", "will I", "can i" ,"can"]

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
