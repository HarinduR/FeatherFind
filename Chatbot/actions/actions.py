from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from custom_components.bert_intent_classifier import BertIntentClassifier  # Use correct class name

import requests
import logging

# Initialize Logger
logger = logging.getLogger(__name__)

# Define API endpoints for each component
BIRD_INFO_API = "http://localhost:5000/bird_info"
RANGE_PREDICTION_API = "http://localhost:5000/range_prediction"
IMAGE_CLASSIFICATION_API = "http://localhost:5000/image_classification"
KEYWORD_FINDER_API = "http://localhost:5000/keyword_finder"

class ActionClassifyIntent(Action):
    def name(self) -> str:
        return "action_classify_intent"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text")
        logger.info(f"🔍 User Query: {user_query}")

        # ✅ Identify Intent using Rasa's NLU
        intent = tracker.latest_message.get("intent", {}).get("name", "fallback")
        confidence = tracker.latest_message.get("intent", {}).get("confidence", 0.0)
        logger.info(f"✅ Predicted Intent: {intent} (Confidence: {confidence:.2f})")

        # ✅ Handle Low Confidence
        if confidence < 0.7:
            dispatcher.utter_message(text="I'm not sure about your request. Let me analyze it further.")
            return []

        # ✅ Route query based on intent
        api_url = None
        response_message = "I couldn't process your request."

        if intent == "bird_info_generate":
            api_url = BIRD_INFO_API
        elif intent == "image_classification":
            api_url = IMAGE_CLASSIFICATION_API
            dispatcher.utter_message(text="Please upload an image for bird identification.")
        elif intent == "range_prediction":
            api_url = RANGE_PREDICTION_API
            dispatcher.utter_message(text="I can predict bird migration patterns. Which bird are you interested in?")
        elif intent == "keyword_finder":
            api_url = KEYWORD_FINDER_API
            dispatcher.utter_message(text="I can analyze bird-related keywords. Please provide the text.")
        elif intent == "fallback":
            dispatcher.utter_message(text="I couldn't understand that. Can you rephrase?")
            return []

        if api_url:
            try:
                response = requests.post(api_url, json={"query": user_query})
                if response.status_code == 200:
                    response_message = response.json().get("result", response_message)
                else:
                    logger.error(f"❌ API request failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ API call error: {e}")
                response_message = "I encountered an error while fetching the data."

        dispatcher.utter_message(text=response_message)
        return []
