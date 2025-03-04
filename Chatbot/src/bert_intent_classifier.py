# import torch
# import numpy as np
# import pickle
# from transformers import BertTokenizer, BertForSequenceClassification
# from rasa.nlu.components import Component
# from rasa.nlu.model import Metadata

# class BlaERTIntentCssifier(Component):

#     name = "bert_intent_classifier"
#     provides = ["intent"]
#     requires = []
#     defaults = {}
#     language_list = ["en"]

#     def __init__(self, component_config=None):
#         super().__init__(component_config)

#         self.model_path = "models/BERT_model2"
#         self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
#         self.model = BertForSequenceClassification.from_pretrained(self.model_path, from_safetensors=True)
        
#         with open(f"{self.model_path}/label_mapping.pkl", "rb") as f:
#             self.label_mapping = pickle.load(f)

#     def process(self, message, **kwargs):

#         text = message.text
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             predicted_label = torch.argmax(predictions).item()

#         # Get intent name
#         intent_name = self.label_mapping.get(predicted_label, "unknown")

#         message.set("intent", {"name": intent_name, "confidence": predictions[0][predicted_label].item()}, add_to_output=True)

#     @classmethod
#     def load(cls, model_dir, model_metadata: Metadata, cached_component, **kwargs):

#         return cls(model_metadata.get("bert_intent_classifier", {}))
