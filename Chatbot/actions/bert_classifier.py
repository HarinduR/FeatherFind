from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from custom_components.bert_intent_classifier import BERTIntentClassifier

# Load BERT model
bert_classifier = BERTIntentClassifier()

class ActionClassifyIntent(Action):
    def name(self):
        return "action_classify_intent"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        predicted_intent = bert_classifier.predict_intent(user_message)

        # Send response
        response = f"The predicted intent is: {predicted_intent}"
        dispatcher.utter_message(response)

        return [SlotSet("intent", predicted_intent)]
