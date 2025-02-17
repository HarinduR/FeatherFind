from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from custom_components.bert_intent_classifier import BERTIntentClassifier
#from BirdInfoGenerater.src.bird_info_generator import BirdInfoRetriever  # If used
import requests  # For making HTTP requests to the Flask API
import logging

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
            '''retriever = BirdInfoRetriever()
            response = retriever.get_bird_info(user_query)
            dispatcher.utter_message(text=response)'''
            dispatcher.utter_message(text="info generator.")


        elif predicted_intent == "image_classification":
            dispatcher.utter_message(text="Please upload an image for bird identification.")

        elif predicted_intent == "range_prediction":
            dispatcher.utter_message(text="I can predict bird migration patterns. Which bird are you interested in?")

        elif predicted_intent == "keyword_finder":
            # Call the Flask API for bird query
            flask_api_url = "http://127.0.0.1:5000/query_bird"  # Make sure this URL matches your Flask app's host and port
            payload = {"text": user_query}
            try:
                response = requests.post(flask_api_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    response_text = (
                        f"Query: {data.get('query')}\n"
                        f"Features: {data.get('features')}\n"
                        f"Results: {data.get('results')}"
                    )
                else:
                    response_text = "Failed to connect to the bird query service."
            except Exception as e:
                logger.error(f"Error calling Flask API for query_bird: {str(e)}")
                response_text = "An error occurred while processing your request."
            
            dispatcher.utter_message(text=response_text)

        elif predicted_intent == "fallback":
            dispatcher.utter_message(text="I couldn't understand that. Can you rephrase?")
        
        else:
            dispatcher.utter_message(text="I'm not sure how to help with that. Can you rephrase your question?")
            print(f"❌ Unknown Intent: {predicted_intent}")

        return []
