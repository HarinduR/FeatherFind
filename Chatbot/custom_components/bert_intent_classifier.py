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

# ✅ Register as a Rasa NLU Component
@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False)
class BertIntentClassifier(GraphComponent, IntentClassifier):
    """Custom BERT-based Intent Classifier for Rasa."""

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
        """Initialize the BERT Intent Classifier."""
        super().__init__()

        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        # ✅ Define model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../models/BERT_model")

        # ✅ Load Tokenizer & Model
        logging.info("🔹 Loading BERT tokenizer & model for intent classification...")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)

        # ✅ Define intent mapping (Make sure it matches your dataset)
        self.intent_mapping = {
            5: "greet",
            4: "goodbye",
            0: "affirm",
            3: "deny",
            8: "mood_great",
            9: "mood_unhappy",
            2: "bot_challenge",
            1: "bird_info_generate",
            10: "range_prediction",
            6: "image_classification",
            7: "keyword_finder",
            11: "thank_you",
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
        """Creates an instance of this component."""
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> None:
        """Rasa requires a `train` method, but BERT is pre-trained, so this is skipped."""
        pass

    def process(self, messages: List[Message]) -> List[Message]:
        """Predicts intent for a given list of messages."""
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

            # ✅ Set intent in message
            message.set("intent", {"name": intent_name, "confidence": confidence})

        return messages

    def persist(self, file_name: str, model_dir: str) -> None:
        """BERT is already stored, so nothing to persist."""
        pass
