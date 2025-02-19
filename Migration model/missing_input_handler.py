import numpy as np
import pandas as pd

import joblib
import pandas as pd
import numpy as np

# Load all models
model_A = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\migration_prediction_model.pkl")
model_B = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\location_prediction_model.pkl")
model_C = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl")

model_C_data = joblib.load(r"C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\models\time_prediction_model.pkl")
month_model = model_C_data['month_model']
hour_model = model_C_data['hour_model']
county_encoder = model_C_data['county_encoder']
locality_encoder = model_C_data['locality_encoder']

required_inputs = {
    "Model A": ['Year', 'Day', 'Day_of_Week', 'Hour', 'COUNTY_', 'LOCALITY_', 'COMMON NAME_'],
    "Model B": ['Year', 'Month', 'Day', 'Day_of_Week', 'Hour', 'LATITUDE', 'LONGITUDE', 'COMMON NAME_'],
    "Model C": ['OBSERVATION', 'Year', 'Day_of_Week', 'LOCALITY_encoded',
                'Is_Summer', 'Is_Winter', 'Is_Spring', 'Is_Autumn',
                'Is_Morning', 'Is_Afternoon', 'Is_Evening', 'Is_Night', 'COMMON NAME_']
}

def preprocess_user_input(user_input, model_name):
    """
    Preprocess user input to handle missing values before feeding it to the model.
    """
    model_features = required_inputs[model_name]
    
    # Step 1: Ensure Bird Name Exists
    if 'COMMON NAME_' not in user_input or not user_input['COMMON NAME_']:
        raise ValueError("Error: Bird species name is required.")

    # Step 2: Handle Missing Values
    default_values = {
        'Year': 2024,  # Default to current year
        'Month': np.nan,  # Cannot be defaulted, needs inference
        'Day': np.nan,  # Needs inference
        'Day_of_Week': np.nan,  # Will infer from 'Year' & 'Day'
        'Hour': np.nan,  # Can be imputed using historical patterns
        'LATITUDE': np.nan,  # May need geo-matching
        'LONGITUDE': np.nan,
        'COUNTY_': 'Unknown',
        'LOCALITY_': 'Unknown',
        'LOCALITY_encoded': 0,  # Default encoding for missing values
        'Is_Summer': 0, 'Is_Winter': 0, 'Is_Spring': 0, 'Is_Autumn': 0,
        'Is_Morning': 0, 'Is_Afternoon': 0, 'Is_Evening': 0, 'Is_Night': 0
    }

    # Step 3: Fill Missing Values
    for feature in model_features:
        if feature not in user_input or pd.isna(user_input[feature]):
            if feature in default_values:
                user_input[feature] = default_values[feature]
            else:
                raise ValueError(f"Missing required input: {feature}")

    # Step 4: Infer Values if Needed
    if pd.isna(user_input['Day_of_Week']) and not pd.isna(user_input['Year']) and not pd.isna(user_input['Day']):
        user_input['Day_of_Week'] = pd.Timestamp(year=user_input['Year'], month=1, day=user_input['Day']).dayofweek

    # Step 5: Encode LOCATION for Model C
    if model_name == "Model C":
        user_input['LOCALITY_encoded'] = locality_encoder.transform([user_input['LOCALITY_']])[0]

    # Step 6: Ensure Format is Correct
    return pd.DataFrame([user_input])
