from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load BERT Model
model_path = "C:\Users\Daham\Documents\GitHub\FeatherFind\Chatbot\models\BERT_model5"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

@app.route("/predict", methods=["POST"])
def predict_intent():
    data = request.json
    text = data.get("text", "")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    intent_index = torch.argmax(outputs.logits).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][intent_index].item()

    return jsonify({"intent": intent_index, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
