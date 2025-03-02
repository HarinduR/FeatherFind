

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load both models
bird_detector_model = tf.keras.models.load_model("second_model.h5")  # First model: Bird detection
bird_classifier_model = tf.keras.models.load_model("check_try_model.h5")  # Second model: Bird classification

# Class names for bird species classification
class_names = ["Blue Tailed Bee Eater", "Red Vented Bul Bul", "White Throated Kingfisher", "Unknown"]

def preprocess_image(image):
    # Convert image to RGB if it's in a different format (e.g., PNG with transparency)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize and normalize the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("basicPage.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(file)

    # Preprocess image (including PNG-to-JPG conversion if necessary)
    processed_image = preprocess_image(image)

    # Run the first model to check
    bird_prediction = bird_detector_model.predict(processed_image)
    is_bird = np.argmax(bird_prediction, axis=1)[0]  # 0 = Bird (3birds), 1 = Not a Bird (randomclass2)

    if is_bird == 1:
        return jsonify({"result": "Sorry, Cannot Identify this image"})

    # Run the second model for bird classification
    predictions = bird_classifier_model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if confidence < 0.75:
        return jsonify({"result": "Not a recognized bird species", "confidence": f"{confidence * 100:.2f}%"})

    return jsonify({"result": class_names[predicted_class], "confidence":f"{confidence * 100:.2f}%"})

if __name__ == "__main__":
    app.run(debug=True)

