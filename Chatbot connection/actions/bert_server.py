from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Fix: Use the correct model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "BERT_model5")  # 🔹 Updated to BERT_model5

# ✅ Ensure the model exists
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    raise FileNotFoundError(f"🚨 Model file not found in {MODEL_DIR}")

print(f"✅ Loading BERT model from {MODEL_DIR}")

# ✅ Load Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ BERT Model Loaded Successfully.")

@app.route("/predict", methods=["POST"])
def predict_intent():
    """
    Receives a text input and returns the predicted intent.
    """
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided for prediction."}), 400

    # ✅ Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # ✅ Extract intent index and confidence
    intent_index = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][intent_index].item()

    return jsonify({"intent": intent_index, "confidence": round(confidence, 3)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
