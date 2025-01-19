import pandas as pd
import pickle
from fuzzywuzzy import process

# Load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df["Name"] = df["Name"].str.lower().str.strip()  # Preprocess names
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None

# Load keyword extraction model
def load_keyword_extractor(model_path):
    try:
        with open(model_path, "rb") as f:
            extract_keywords = pickle.load(f)
        return extract_keywords
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None

# Fuzzy matching function to find the closest bird name
def find_best_match(user_input, bird_list):
    best_match, score = process.extractOne(user_input, bird_list)
    return best_match if score > 80 else None  # Set confidence threshold

# Search bird using keyword-based matching
def search_by_keywords(user_input, df, extract_keywords):
    keywords = extract_keywords(user_input)
    results = df[df["Distinctive Features"].apply(lambda x: any(kw in x.lower() for kw in keywords))]
    return results

# Combine fuzzy matching and keyword search
def search_bird(user_input, df, extract_keywords):
    # Try fuzzy matching first
    matched_bird = find_best_match(user_input, df["Name"].tolist())
    if matched_bird:
        bird_info = df[df["Name"] == matched_bird].iloc[0]
        return f"Found bird: {bird_info['Name']} - {bird_info['Distinctive Features']}"

    # If no exact match, try keyword search
    keyword_results = search_by_keywords(user_input, df, extract_keywords)
    if not keyword_results.empty:
        return keyword_results[["Name", "Distinctive Features"]].to_string(index=False)

    return "No information found for the given input."

def main():
    # Paths to dataset and model
    dataset_path = "../Dataset/preprocessed_dataset.csv"
    extractor_model_path = "../model/keyword_extractor.pkl"

    # Load the dataset and keyword extraction model
    birds_df = load_dataset(dataset_path)
    extract_keywords = load_keyword_extractor(extractor_model_path)

    if birds_df is None or extract_keywords is None:
        print("Error: Unable to load necessary files. Please check the paths.")
        return

    print("Welcome to the Bird Information System")
    while True:
        query = input("Enter bird name or description (or type 'exit' to quit): ").strip().lower()
        if query == 'exit':
            break
        result = search_bird(query, birds_df, extract_keywords)
        print("\n" + result + "\n")

if __name__ == "__main__":
    main()
