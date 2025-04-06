from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from custom_components.bert_intent_classifier import BertIntentClassifier
import requests  # For making HTTP requests to the Flask API
import logging
from typing import Text, Dict, Any, List


# Initialize Logger
logger = logging.getLogger(__name__)

class ActionClassifyIntent(Action):
    def name(self) -> str:
        return "action_classify_intent"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text")
        print(f"🔍 User Query: {user_query}")

        # ✅ Use BERT model for intent classification
        intent_classifier = BertIntentClassifier()
        predicted_intent, confidence = intent_classifier.predict_intent(user_query)
        print(f"✅ Predicted Intent: {predicted_intent} (Confidence: {confidence:.2f})")

        # ✅ Handle Low Confidence
        if confidence < 0.7:
            dispatcher.utter_message(text="I'm not sure about your request. Let me analyze it further.")
            print("⚠️ Confidence too low. Falling back to Rasa NLU model.")
            return []  # Let Rasa NLU handle it

        # ✅ Route query based on intent
        if predicted_intent == "bird_info_generate":
            dispatcher.utter_message(text="info generator.")

        elif predicted_intent == "image_classification":
            dispatcher.utter_message(text="Please upload an image for bird identification.")

        elif predicted_intent == "range_prediction":
            dispatcher.utter_message(text="I can predict bird migration patterns. Which bird are you interested in?")

        elif predicted_intent == "keyword_finder":
            return [ActionKeywordFinder().run(dispatcher, tracker, domain)]

        elif predicted_intent == "fallback":
            dispatcher.utter_message(text="I couldn't understand that. Can you rephrase?")
        
        else:
            dispatcher.utter_message(text="I'm not sure how to help with that. Can you rephrase your question?")
            print(f"❌ Unknown Intent: {predicted_intent}")

        return []

class ActionKeywordFinder(Action):
    def name(self) -> str:
        return "action_keyword_finder"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text")
        flask_api_url = "http://127.0.0.1:5001/query_bird"  # Ensure Flask API is running

        payload = {"text": user_query}
        try:
            response = requests.post(flask_api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                bird_list = [list(item.values())[0] for item in data.get('results', [])]  # Extract only the names
                
                if not bird_list:
                    response_text = "Sorry! The system was unable to identify the bird :("
                elif len(bird_list) > 1:
                    response_text = f"The birds that match your description are: {', '.join(bird_list)}"
                else:
                    response_text = f"The bird you are describing is likely to be {bird_list[0]}"

            else:
                response_text = "Failed to connect to the bird query service."
        except Exception as e:
            logger.error(f"Error calling Flask API for query_bird: {str(e)}")
            response_text = "An error occurred while processing your request."
        
        dispatcher.utter_message(text=response_text)
        return []

class ActionGetBirdInfo(Action):
    def name(self) -> Text:
        return "action_get_bird_info"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Extract user query from the tracker
        user_query = tracker.latest_message.get("text")
        logger.info(f"🔍 User Query: {user_query}")

        # Define the Flask API endpoint
        flask_api_url = "http://127.0.0.1:5003/get_bird_info"

        # Prepare the payload for the API request
        payload = {"query": user_query}

        try:
            # Send POST request to the Flask API
            response = requests.post(flask_api_url, json=payload)
            response.raise_for_status()  # Raise an exception for non-200 status codes

            # Parse the API response
            api_response = response.json()
            final_response = api_response.get("response", "No response from the API.")

            # Send the response back to the user
            dispatcher.utter_message(text=final_response)

        except requests.exceptions.RequestException as e:
            # Handle API request errors
            logger.error(f"❌ Error calling Flask API: {e}")
            dispatcher.utter_message(text="There was an error processing your request. Please try again later.")

        return []

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
            dispatcher.utter_message(text="⚠ Sorry, something went wrong while processing the response.")
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
