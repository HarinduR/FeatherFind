import pandas as pd
import os
from fuzzywuzzy import process
import pickle

# Define the extract_keywords function before loading the pickle file
def extract_keywords(text):
    words = text.lower().split()
    return [word for word in words if len(word) > 2]  # Example of keyword extraction

# Function to load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df["Name"] = df["Name"].str.lower().str.strip()  # Normalize bird names
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None

# Function to load the template
def load_template(template_path):
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            template = file.read()
        return template
    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found.")
        return None

def generate_description(template, bird_data):
    try:
        return template.format(
            Name=bird_data.get("Name", "Unknown"),
            Scientific_Name=bird_data.get("Scientific Name", "   "),
            Conservation_Status=bird_data.get("Conservation Status", "Not mentioned"),
            Distinctive_Features=bird_data.get("Distinctive Features", "Not recorded"),
            Size="an unknown size" if bird_data.get("Size", "Unknown").lower() == "unknown" else bird_data["Size"],
            Habitat="don't know exactly." if bird_data.get("Habitat", "Unknown").lower() == "unknown" else bird_data["Habitat"],
            Behavior="behavior details not available" if bird_data.get("Behavior", "Unknown").lower() == "unknown" else bird_data["Behavior"],
            Range="range is impossible to say exactly." if bird_data.get("Range", "Unknown").lower() == "unknown" else bird_data["Range"]
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

# Function to load keyword extractor using pickle
def load_keyword_extractor(model_path):
    try:
        with open(model_path, "rb") as f:
            extract_keywords_loaded = pickle.load(f)
        return extract_keywords_loaded
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading keyword extractor: {e}")
        return None

# Main function
def main():
    dataset_path = "../Dataset/dataset.csv"
    template_path = "../Templates/template1.txt"
    extractor_model_path = "../model/keyword_extractor.pkl"

    # Load dataset and template
    birds_df = load_dataset(dataset_path)
    if birds_df is None:
        return

    template = load_template(template_path)
    if template is None:
        return

    # Load the keyword extraction model
    extract_keywords_function = load_keyword_extractor(extractor_model_path)

    if not extract_keywords_function:
        print("Error: Could not load keyword extraction model.")
        return

    print("Welcome to the Bird Information System!")
    
    while True:
        query = input("\nEnter bird name (or type 'exit' to quit): ").strip().lower()
        if query == 'exit':
            print("Exiting the system.")
            break
        
        # Fuzzy matching to find the best bird name
        matched_bird = find_best_match(query, birds_df["Name"].tolist())

        if matched_bird:
            bird_row = birds_df[birds_df["Name"] == matched_bird]
            if not bird_row.empty:
                bird_data = bird_row.iloc[0].to_dict()
                description = generate_description(template, bird_data)
                print("\nGenerated Description:\n")
                print(description)
            else:
                print(f"\nNo exact match found for '{query}', but suggested: {matched_bird}")
        else:
            print("\nError: Can't identify as a bird. Please try again.")

        # Suggest similar birds
        print("\nDid you mean any of these birds? ", birds_df["Name"].unique()[:3])

if __name__ == "__main__":
    main()
