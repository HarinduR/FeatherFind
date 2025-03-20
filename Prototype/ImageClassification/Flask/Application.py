from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load both models
try:
    bird_detector_model = tf.keras.models.load_model("../Models/second_model.h5")  # First model: Bird detection
    bird_classifier_model = tf.keras.models.load_model("../Models/check_try_model.h5")  # Second model: Bird classification
    app.logger.info("Models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")
    raise

# Class names for bird species classification
class_names = ["Blue Tailed Bee Eater", "Red Vented Bul Bul", "White Throated Kingfisher", "Unknown"]

def preprocess_image(image):
    # Convert image to RGB if it's in a different format
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize and normalize the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return jsonify({"message": "Flask server is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    app.logger.debug(f"Request files: {request.files}")
    if "image" not in request.files:
        app.logger.error("No file provided in the request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["image"]
    app.logger.info(f"File received: {file.filename}")

    # Open and validate the image
    try:
        image = Image.open(file)
        image.verify()  # Verify that the file is a valid image
        image = Image.open(file)  # Reopen the file after verification
    except Exception as e:
        app.logger.error(f"Invalid image file: {e}")
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess image
    processed_image = preprocess_image(image)

    # Run the first model to check if it's a bird
    bird_prediction = bird_detector_model.predict(processed_image)
    is_bird = np.argmax(bird_prediction, axis=1)[0]

    if is_bird == 1:
        app.logger.info("Image is not a bird")
        return jsonify({"result": "Sorry, Cannot Identify this image"})

    # Run the second model for bird classification
    predictions = bird_classifier_model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if confidence < 0.75:
        app.logger.info("Low confidence in prediction")
        return jsonify({"result": "Not a recognized bird species", "confidence": f"{confidence * 100:.2f}%"})

    app.logger.info(f"Prediction successful: {class_names[predicted_class]}")
    return jsonify({"result": f"I believe this is a {class_names[predicted_class]}", "confidence": f"{confidence * 100:.2f}%"})

if __name__ == "__main__":
    app.run(debug=True)