import os
import torch
import logging
from typing import Any, Dict, List, Text

from transformers import BertTokenizer, BertForSequenceClassification
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.classifiers.classifier import IntentClassifier

@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False)
class BertIntentClassifier(GraphComponent, IntentClassifier):

    name = "bert_intent_classifier"
    provides = ["intent"]
    language_list = ["en"]

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        super().__init__()

        self._model_storage = model_storage
        self._resource = resource

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../models/BERT_model5")

        try:
            logging.info("✅ Loading BERT tokenizer & model for intent classification...")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            logging.info("✅ BERT Intent Classifier loaded successfully.")
        except Exception as e:
            logging.error(f" Error loading BERT model: {e}")
            raise RuntimeError("BERT model could not be loaded. Ensure the model files exist at the specified path.")

        self.intent_mapping = {
            0: "affirm", 
            1: "agree", 
            2: "bird_info_generate", 
            3: "bot_challenge", 
            4: "deny", 
            5: "feedback_negative", 
            6: "feedback_positive", 
            7: "general_question", 
            8: "goodbye",
            9: "greet_checking_in", 
            10: "greet_good_afternoon", 
            11: "greet_good_evening", 
            12: "greet_good_morning", 
            13: "greet_good_night",
            14: "greet_hi", 
            15: "help", 
            16: "image_classification", 
            17: "keyword_finder",
            18: "mood_excited", 
            19: "mood_great", 
            20: "mood_unhappy", 
            21: "non_birds", 
            22: "range_prediction",
            23: "repeat_request", 
            24: "thank_you", 
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "BertIntentClassifier":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../models/BERT_model5")

        if os.path.exists(model_path):
            logging.info(f"✅ Reloading BERT model from {model_path}")
            return cls(config, model_storage, resource, execution_context)

        logging.warning(f"Model path {model_path} not found. Ensure the model exists.")
        return cls(config, model_storage, resource, execution_context)

    def process(self, messages: List[Message]) -> List[Message]:
        texts = [message.get("text") for message in messages]
        logging.info(f"✅ Processing batch of {len(texts)} messages.")

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
        except Exception as e:
            logging.error(f" Error during model inference: {e}")
            return messages

        for i, message in enumerate(messages):
            predicted_class = torch.argmax(logits[i]).item()
            confidence = float(torch.nn.functional.softmax(logits, dim=1)[i][predicted_class])

            if confidence < 0.3:
                intent_name = "fallback"
                logging.warning(f"Low confidence ({confidence:.2f}). Using fallback intent.")
            else:
                intent_name = self.intent_mapping.get(predicted_class, "fallback")

            logging.info(f"✅ Predicted intent: {intent_name} (Confidence: {confidence:.2f})")
            message.set("intent", {"name": intent_name, "confidence": confidence})

        return messages

    def persist(self, file_name: str, model_dir: str) -> None:
        save_path = os.path.join(model_dir, file_name)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.info(f"Model persisted to {save_path}.")

