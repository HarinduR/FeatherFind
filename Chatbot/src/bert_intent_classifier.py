# import torch
# import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification
# from rasa.nlu.components import Component
# from rasa.nlu.model import Metadata

# class BERTIntentClassifier(Component):
#     """Custom RASA Intent Classifier using BERT."""

#     name = "bert_intent_classifier"
#     provides = ["intent"]
#     requires = []
#     defaults = {}
#     language_list = ["en"]

#     def __init__(self, component_config=None):
#         super().__init__(component_config)

#         # Load Fine-Tuned BERT Model
#         self.model_path = "bert_intent_model"
#         self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
#         self.model = BertForSequenceClassification.from_pretrained(self.model_path)

#     def process(self, message, **kwargs):
#         """Classify intent using BERT model."""
#         text = message.text
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             predicted_label = torch.argmax(predictions).item()

#         # Map label index to intent name
#         intent_name = self.get_intent_name(predicted_label)

#         message.set("intent", {"name": intent_name, "confidence": predictions[0][predicted_label].item()}, add_to_output=True)

#     def get_intent_name(self, index):
#         """Map predicted label index to intent name."""
#         label_mapping = {
#             0: "greet",
#             1: "goodbye",
#             2: "affirm",
#             3: "deny",
#             4: "bot_challenge",
#             5: "bird_info_generate",
#             6: "keyword_finder",
#             7: "range_prediction",
#             8: "image_classification",
#             9: "mood_great",
#             10: "mood_unhappy",
#         }
#         return label_mapping.get(index, "unknown")

#     @classmethod
#     def load(cls, model_dir, model_metadata: Metadata, cached_component, **kwargs):
#         """Load the trained component."""
#         return cls(model_metadata.get("bert_intent_classifier", {}))


import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from rasa.nlu.components import Component
from rasa.nlu.model import Metadata

class BERTIntentClassifier(Component):
    """Custom RASA Intent Classifier using BERT."""

    name = "bert_intent_classifier"
    provides = ["intent"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        super().__init__(component_config)

        # Load Fine-Tuned BERT Model
        self.model_path = "models/BERT_model"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, from_safetensors=True)
        
        # Load label mapping
        with open(f"{self.model_path}/label_mapping.pkl", "rb") as f:
            self.label_mapping = pickle.load(f)

    def process(self, message, **kwargs):
        """Classify intent using BERT model."""
        text = message.text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions).item()

        # Get intent name
        intent_name = self.label_mapping.get(predicted_label, "unknown")

        message.set("intent", {"name": intent_name, "confidence": predictions[0][predicted_label].item()}, add_to_output=True)

    @classmethod
    def load(cls, model_dir, model_metadata: Metadata, cached_component, **kwargs):
        """Load the trained component."""
        return cls(model_metadata.get("bert_intent_classifier", {}))
