import os
import torch
import logging
from typing import Any, Dict, List, Optional, Text, Type

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
    requires = []
    defaults = {}
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
        self._execution_context = execution_context

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../models/BERT_model4")

        logging.info("🔹 Loading BERT tokenizer & model for intent classification...")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)

        self.intent_mapping = {
            0: "affirm",
            1: "agree",
            2: "bird_info_generate",
            3: "bot_challenge",
            4: "bot_opinion",
            5: "deny",
            6: "fallback",
            7: "feedback_negative",
            8: "feedback_positive",
            # 9: "feedback_suggestions",
            # 10: "fun_fact",
            9: "general_question",
            10: "goodbye",
            11: "greet_checking_in",
            12: "greet_formal",
            13: "greet_good_afternoon",
            14: "greet_good_evening",
            15: "greet_good_morning",
            16: "greet_good_night",
            17: "greet_hi",
            18: "greet_welcome",
            19: "help",
            20: "image_classification",
            21: "keyword_finder",
            22: "mood_excited",
            23: "mood_great",
            24: "mood_unhappy",
            25: "non_birds",
            36: "range_prediction",
            27: "repeat_request",
            28: "thank_you",
            29: "user_preferences"
        }

        logging.info("✅ BERT Intent Classifier loaded successfully.")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "BertIntentClassifier":
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> None:
        pass

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.get("text")
            logging.info(f"🔹 Predicting intent for: {text}")

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            intent_name = self.intent_mapping.get(predicted_class, "fallback")
            confidence = float(torch.nn.functional.softmax(logits, dim=1)[0][predicted_class])

            logging.info(f"✅ Predicted intent: {intent_name} (Confidence: {confidence})")

            message.set("intent", {"name": intent_name, "confidence": confidence})

        return messages

    def persist(self, file_name: str, model_dir: str) -> None:
        pass
