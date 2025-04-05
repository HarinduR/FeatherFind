from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import datetime
from difflib import get_close_matches
import requests
import io
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
import joblib
import io
import os

MODEL_URL1 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/migration_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH1 = "migration_prediction_model.pkl"

# ✅ Function: Download Large Model in Chunks
def download_model(url, save_path, chunk_size=1024 * 1024):  # 1MB chunks
    if os.path.exists(save_path):  # ✅ Skip download if file exists
        print(f"📁 Using cached model: {save_path}")
        return save_path

    print("📥 Downloading model. Please wait...")
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # ✅ Check for errors
        total_size = int(response.headers.get("content-length", 0))  # ✅ Get file size
        downloaded = 0
        
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    print(f"🔄 Downloaded: {downloaded / total_size:.2%}", end="\r")  # ✅ Show progress

    print("\n✅ Model downloaded successfully.")
    return save_path

# ✅ Download model if not cached
model_file_path = download_model(MODEL_URL1, MODEL_PATH1)

# ✅ Load Model
with open(model_file_path, "rb") as model_file:
    model_data1 = joblib.load(model_file)

rf_model = model_data1['rf_final']
label_encoders1 = model_data1['label_encoders']
selected_features1 = model_data1['selected_features']

print("✅ Model loaded successfully!")



import requests
import joblib
import io
import os

MODEL_URL3 = "https://raw.githubusercontent.com/Deshan-Senanayake/Bird-Range-Prediction/main/Migration%20model/models/time_prediction_model.pkl"

# ✅ Define a local model path
MODEL_PATH3 = "time_prediction_model.pkl"

# ✅ Function: Download Large Model in Chunks
def download_model(url3, save_path3, chunk_size=1024 * 1024):  # 1MB chunks
    if os.path.exists(save_path3):  # ✅ Skip download if file exists
        print(f"📁 Using cached model: {save_path3}")
        return save_path3

    print("📥 Downloading model. Please wait...")
    
    with requests.get(url3, stream=True) as response3:
        response3.raise_for_status()  # ✅ Check for errors
        total_size = int(response3.headers.get("content-length", 0))  # ✅ Get file size
        downloaded = 0
        
        with open(save_path3, "wb") as file:
            for chunk in response3.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    print(f"🔄 Downloaded: {downloaded / total_size:.2%}", end="\r")  # ✅ Show progress

    print("\n✅ Model downloaded successfully.")
    return save_path3

# ✅ Download model if not cached
model_file_path3 = download_model(MODEL_URL3, MODEL_PATH3)

# ✅ Load Model
with open(model_file_path3, "rb") as model_file3:
    model_data3 = joblib.load(model_file3)

month_model = model_data3['month_model']
hour_model = model_data3['hour_model']
selected_features3 = model_data3['selected_features']
label_encoders3 = model_data3['label_encoders']

print("✅ Model loaded successfully!")


