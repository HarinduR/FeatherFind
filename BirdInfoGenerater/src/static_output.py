import pandas as pd
import os
from fuzzywuzzy import process
import dill
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import contextlib

# Suppress NLTK download messages
with contextlib.redirect_stdout(None):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Define the keyword extraction function
def extract_keywords(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    return keywords

# Function to load the template file
def load_template(template_path):
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            template = file.read()
        return template
    except Exception as e:
        print(f"Error loading template: {e}")
        return None

# Function to generate the bird description using the template
def generate_description(template, bird_data):
    try:
        return template.format(
            Name=bird_data.get("Name", "Unknown"),
            Scientific_Name=bird_data.get("Scientific Name", "Unknown"),
            Conservation_Status=bird_data.get("Conservation Status", "Unknown"),
            Distinctive_Features=bird_data.get("Distinctive Features", "Unknown"),
            Size=bird_data.get("Size", "Unknown") if bird_data.get("Size", "Unknown") != "Unknown" else "an unknown size",
            Habitat=bird_data.get("Habitat", "Unknown") if bird_data.get("Habitat", "Unknown") != "Unknown" else "an unknown habitat",
            Behavior=bird_data.get("Behavior", "Unknown") if bird_data.get("Behavior", "Unknown") != "Unknown" else "behavior details not available",
            Range=bird_data.get("Range", "Unknown") if bird_data.get("Range", "Unknown") != "Unknown" else "range details not available"
        )
    except KeyError as e:
        print(f"Missing key in data: {e}")
        return None

# Function to find the best match using fuzzy matching
def find_best_match(user_input, bird_list):
    best_match, score = process.extractOne(user_input, bird_list)
    if score > 80:  # Confidence threshold
        return best_match
    return None

# Load keyword extraction model
def load_keyword_extractor(model_path):
    try:
        with open(model_path, "rb") as f:
            extract_keywords_loaded = dill.load(f)
        return extract_keywords_loaded
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading keyword extractor: {e}")
        return None

# Search for birds using keywords
def search_by_keywords(user_input, df, extract_keywords):
    keywords = extract_keywords(user_input)
    results = df[df["Distinctive Features"].apply(lambda x: any(kw in str(x).lower() for kw in keywords))]
    return results

# Load the dataset with error handling
dataset_path = r"..\Dataset\dataset.csv"
if os.path.exists(dataset_path):
    try:
        birds_df = pd.read_csv(dataset_path, encoding="utf-8")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit()
else:
    print(f"Dataset file not found at {dataset_path}")
    sys.exit()

# Load the template
template_path = r"..\Templates\template1.txt"
if os.path.exists(template_path):
    template = load_template(template_path)
    if template:
        print("Template loaded successfully.")
    else:
        print("Template loading failed.")
        sys.exit()
else:
    print(f"Template file not found at {template_path}")
    sys.exit()

# Load keyword model
keyword_model_path = r"..\model\keyword_extractor.pkl"
extract_keywords_function = load_keyword_extractor(keyword_model_path)

if not extract_keywords_function:
    print("Error: Could not load keyword extraction model.")
    sys.exit()

# Ensure column "Name" exists and clean it
if "Name" in birds_df.columns:
    birds_df["Name"] = birds_df["Name"].astype(str).str.lower().str.strip()
else:
    print("Column 'Name' not found in dataset.")
    sys.exit()

# User interaction loop
while True:
    user_input = input("\nEnter bird name or description (or type 'exit' to quit): ").strip().lower()
    if user_input == 'exit':
        print("Exiting program.")
        break

    # First, try exact or fuzzy match
    matched_bird = find_best_match(user_input, birds_df["Name"].tolist())

    if matched_bird:
        bird_row = birds_df[birds_df["Name"] == matched_bird]
        if not bird_row.empty:
            bird_data = bird_row.iloc[0].to_dict()
            description = generate_description(template, bird_data)
            print("\n\nGenerated Description:")
            print(description)
        else:
            print(f"No exact match found for '{user_input}', but suggested: {matched_bird}")
    elif extract_keywords_function:
        # Perform keyword-based search
        keyword_results = search_by_keywords(user_input, birds_df, extract_keywords_function)
        if not keyword_results.empty:
            print("\nBirds found based on descriptive keywords:")
            print(keyword_results[["Name", "Distinctive Features"]].to_string(index=False))
        else:
            print("\nError: Can't identify as a bird. Please try again.")
    else:
        print("\nError: Can't identify as a bird. Please try again.")

    # Provide user with available bird names if needed
    print("\nDid you mean any of these birds? ", birds_df["Name"].unique()[:5])
