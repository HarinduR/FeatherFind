from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("new bird_classification_model.h5")
class_names = ["Blue Tailed Bee Eaterrrrrrr", "Red Vented Bulbul", "White Throated Kingfisher"]


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if confidence < 0.6:
        return jsonify({"result": "Not a recognized bird", "confidence": float(confidence)})

    return jsonify({"result": class_names[predicted_class], "confidence": float(confidence)})


if __name__ == "__main__":
    app.run(debug=True)
