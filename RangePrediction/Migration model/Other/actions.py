import requests
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# API endpoint for prediction
RANGE_PREDICTION_API = "http://127.0.0.1:5000/predict"

class ActionHandleBirdPrediction(Action):
    def name(self) -> str:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # ✅ Capture full user query correctly
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 User Query Sent to API: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Could you rephrase it?")
            return []

        # ✅ Construct Proper API Request
        request_payload = {"query": user_query}
        headers = {"Content-Type": "application/json"}

        try:
            # ✅ Send API Request
            response = requests.post(RANGE_PREDICTION_API, json=request_payload, headers=headers)

            if response.status_code == 200:
                json_response = response.json()
                meaningful_sentence = json_response.get("meaningful_sentence", "I couldn't generate a response.")
                probability = json_response.get("probability", "Unknown")

                # ✅ Log API Response for Debugging
                logger.info(f"✅ API Response: {json_response}")

                # ✅ Return prediction result to user
                dispatcher.utter_message(text=f"{meaningful_sentence} (Confidence: {probability:.1f}%)")
            else:
                logger.error(f"❌ API request failed: {response.status_code}, Response: {response.text}")
                dispatcher.utter_message(text=f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API call error: {e}")
            dispatcher.utter_message(text="There was an error connecting to the prediction API.")

        return []
