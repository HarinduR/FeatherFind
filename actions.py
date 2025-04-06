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

        # ✅ Match API key: "you can use these locations"
        if "you can use these locations" in json_response:
            locations_list = json_response["you can use these locations"]
            locations_text = "\n".join(locations_list)
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nYou can use these locations:\n{locations_text}"
            )
            return

        if "valid_bird_names" in json_response:
            valid_birds_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}"
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

        # ✅ Handle bird species not found
        if "valid_bird_names" in json_response:
            valid_birds_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nValid Bird Species:\n{valid_birds_text}"
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
        if "you can use these locations" in json_response:
            locations_text = "\n".join(json_response["you can use these locations"])
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nYou can use these locations:\n{locations_text}"
            )
            return

        # ✅ Handle bird name validation message
        if "valid_bird_names" in json_response:
            bird_names_text = "\n".join(json_response["valid_bird_names"])
            dispatcher.utter_message(
                text=f"{json_response['message']}\n\nValid Bird Species:\n{bird_names_text}"
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
