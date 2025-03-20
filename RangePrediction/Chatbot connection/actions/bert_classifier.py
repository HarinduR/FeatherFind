from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import logging
from custom_components.bert_intent_classifier import BertIntentClassifier

logger = logging.getLogger(__name__)

# ✅ Ensure BertIntentClassifier is properly initialized
def get_intent_classifier():
    try:
        return BertIntentClassifier(config=None, model_storage=None, resource=None, execution_context=None)
    except Exception as e:
        logger.error(f"❌ Error initializing BERT classifier: {e}")
        return None

bert_classifier = get_intent_classifier()

class ActionClassifyIntent(Action):
    def name(self):
        return "action_classify_intent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")

        if not user_message:
            dispatcher.utter_message("I couldn't understand your input. Could you try again?")
            return []

        # ✅ Ensure classifier is available
        if bert_classifier is None:
            dispatcher.utter_message("The intent classifier is currently unavailable. Please try again later.")
            return []

        # ✅ Predict intent
        try:
            predicted_intent = bert_classifier.predict_intent(user_message)
            if not predicted_intent:
                predicted_intent = "unknown"

            response = f"The predicted intent is: {predicted_intent}"
            dispatcher.utter_message(response)

            return [SlotSet("intent", predicted_intent)]

        except Exception as e:
            logger.error(f"❌ Error predicting intent: {e}")
            dispatcher.utter_message("There was an error processing your request.")
            return []
