from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and features
model = joblib.load('bird_migration_model.pkl')
feature_columns = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return "Bird Migration Prediction API is running!"

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from request
    input_df = pd.DataFrame(data, index=[0])  # Convert to DataFrame

    # Ensure input matches the training features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        response = {'result': 'Bird presence predicted!'}
    else:
        response = {'result': 'No bird presence predicted.'}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
