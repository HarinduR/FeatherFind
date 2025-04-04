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

'''
class ActionHandleBirdPrediction(Action):
    def name(self) -> str:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text", "").strip()
        logger.info(f"🔍 User Query Sent to API: {user_query}")

        if not user_query:
            dispatcher.utter_message(text="I couldn't understand your request. Could you rephrase it?")
            return []

        request_payload = {"query": user_query}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post("http://127.0.0.1:5002/predict", json=request_payload, headers=headers)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            json_response = response.json()

            meaningful_sentence = json_response.get("meaningful_sentence", "I couldn't generate a response.")
            probability = float(json_response.get("probability", 0.0))  # Ensure probability is a float

            logger.info(f"✅ API Response: {json_response}")
            dispatcher.utter_message(text=f"{meaningful_sentence} (Confidence: {probability:.1f}%)")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API call error: {e}")
            dispatcher.utter_message(text="There was an error connecting to the prediction API.")

        return []
'''   
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
    
import requests
import logging
from typing import Dict, Text, Any, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

# Consolidated API configuration
FLASK_API_BASE = "http://127.0.0.1:5002"  # Update with your actual URL if different

API_ENDPOINTS = {
    "presence": f"{FLASK_API_BASE}/predict_presence",
    "location": f"{FLASK_API_BASE}/predict_location",
    "time": f"{FLASK_API_BASE}/predict_best_time"
}

class ActionBirdPrediction(Action):
    def name(self) -> Text:
        return "action_range_prediction"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text", "").strip()
        if not user_query:
            dispatcher.utter_message(text="Please provide a valid bird observation query.")
            return []

        # Determine which API endpoint to use
        endpoint_type = self._determine_endpoint(user_query)
        endpoint_url = API_ENDPOINTS.get(endpoint_type)

        try:
            response = requests.post(
                endpoint_url,
                json={"query": user_query},
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                dispatcher.utter_message(text="Sorry, I'm having trouble processing your request right now.")
                return []

            api_response = response.json()

            # Handle different response types
            if "error" in api_response:
                self._handle_api_errors(api_response, dispatcher)
            else:
                self._handle_success_response(endpoint_type, api_response, dispatcher)

        except requests.exceptions.RequestException as e:
            logger.error(f"API Connection Error: {str(e)}")
            dispatcher.utter_message(text="I'm having trouble connecting to the bird prediction service. Please try again later.")

        return []

    def _determine_endpoint(self, query: Text) -> Text:
        """Determine which prediction endpoint to use based on query content"""
        query = query.lower()
        
        if any(kw in query for kw in ["where", "location", "spot", "place", "find"]):
            return "location"
        if any(kw in query for kw in ["when", "time", "hour", "best time", "season"]):
            return "time"
        return "presence"

    # Updated error handling in actions.py
    def _handle_api_errors(self, response: Dict, dispatcher: CollectingDispatcher):
        """Handle different error scenarios from the API"""
        if "valid_localities" in response:
            locations = response["valid_localities"]
            buttons = [{"title": loc, "payload": loc} for loc in locations]
            dispatcher.utter_message(
                text=f"🚨 {response.get('message', 'Please select a valid location from these options:')}",
                buttons=buttons
            )
        elif "valid_bird_names" in response:
            birds = response["valid_bird_names"]
            buttons = [{"title": bird, "payload": bird} for bird in birds]
            dispatcher.utter_message(
                text=f"🐦 {response.get('message', 'Please choose a bird species from these options:')}",
                buttons=buttons
            )
        else:
            error_msg = response.get("error", "An unknown error occurred. Please try again.")
            dispatcher.utter_message(text=f"⚠️ {error_msg}")

    # Updated success response handling
    def _handle_success_response(self, endpoint_type: Text, response: Dict, dispatcher: CollectingDispatcher):
        """Format successful API responses for user"""
        try:
            if endpoint_type == "location":
                locations = ", ".join(response.get("locations", []))
                dispatcher.utter_message(
                    text=f"📍 Here are the best locations to observe {response.get('bird', 'the bird')}:\n{locations}"
                )
                
            elif endpoint_type == "time":
                best_time = response.get('best_time', 'unknown time')
                best_month = response.get('best_month', 'unknown month')
                dispatcher.utter_message(
                    text=f"⏰ Best observation time for {response.get('bird', 'the bird')}:\n"
                        f"• Month: {best_month}\n"
                        f"• Time: {best_time}"
                )
                
            else:  # presence
                bird = response.get('bird', 'This bird')
                confidence = response.get('confidence', 0)
                status = "highly likely" if confidence > 0.7 else "likely" if confidence > 0.4 else "unlikely"
                
                dispatcher.utter_message(
                    text=f"🔍 {bird} is {status} to be present at {response.get('location', 'this location')} "
                        f"(confidence: {confidence:.0%})"
                )
                
        except KeyError as e:
            logger.error(f"Missing key in API response: {str(e)}")
            dispatcher.utter_message(text="Sorry, I encountered an issue processing the response. Please try again.")