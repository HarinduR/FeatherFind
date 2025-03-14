import pandas as pd
import json

# 🔹 File Paths
dataset_file = "../RAG_dataset/dataset.csv"
qa_dataset_file = "../RAG_dataset/qa_dataset.json"

# 🔹 Handle Unknown Values Directly in Dataset
DEFAULT_VALUES = {
    "Size": "a bird of varying sizes",
    "Habitat": "various environments",
    "Distinctive Features": "unique physical traits",
    "Behavior": "different behaviors depending on location",
    "Range": "several regions around the world",
    "Conservation Status": "not well-documented"
}

def clean_text(value, field_name):
    """Replaces 'unknown' or 'information not available' with meaningful defaults."""
    if isinstance(value, str) and value.strip().lower() in ["unknown", "information not available"]:
        return DEFAULT_VALUES.get(field_name, "not well-documented")
    return value.strip()

# 🔹 Load Dataset & Apply Cleaning
df = pd.read_csv(dataset_file)
df.fillna("unknown", inplace=True)

for column in DEFAULT_VALUES.keys():
    df[column] = df[column].apply(lambda x: clean_text(x, column))

# 🔹 Save Cleaned Dataset
cleaned_dataset_file = "../RAG_dataset/cleaned_birds.csv"
df.to_csv(cleaned_dataset_file, index=False, encoding="utf-8")

print(f"✅ Dataset cleaned and saved as {cleaned_dataset_file}")

# 🔹 Define Question Templates
QUESTION_TEMPLATES = {
    "Scientific Name": "What is the scientific name of the {name}?",
    "Conservation Status": "What is the conservation status of the {name}?",
    "Distinctive Features": "What are the distinctive features of the {name}?",
    "Size": "What size is the {name}?",
    "Habitat": "Where does the {name} live?",
    "Behavior": "How does the {name} behave?",
    "Range": "Where can the {name} be found?"
}

# 🔹 Response Templates (for consistency)
RESPONSE_TEMPLATES = {
    "Scientific Name": "The scientific name of the {name} is *{value}*.",
    "Conservation Status": "The {name} is classified as '{value}', meaning it is not currently endangered.",
    "Distinctive Features": "The {name} has the following distinctive features: {value}.",
    "Size": "The {name} is {value} in size.",
    "Habitat": "The {name} is found in the following habitat: {value}.",
    "Behavior": "The behavior of the {name} is described as: {value}.",
    "Range": "The range of the {name} includes: {value}."
}

# 🔹 Generate Full Description for the Bird
def format_bird_description(row):
    """Generate a structured full description for each bird."""
    size = clean_text(row["Size"], "Size")
    habitat = clean_text(row["Habitat"], "Habitat")
    features = clean_text(row["Distinctive Features"], "Distinctive Features")
    behavior = clean_text(row["Behavior"], "Behavior")
    conservation = clean_text(row["Conservation Status"], "Conservation Status")
    range_info = clean_text(row["Range"], "Range")

    description = f"""The {row['Name'].title()} (*{row['Scientific Name']}*) is {size}, commonly found in {habitat}. It is recognized by its {features}. This species is known to be {behavior}. It has a conservation status of "{conservation}" and is primarily found in {range_info}."""
    
    return description.replace("\n    ", " ")  

# 🔹 Create Q&A Pairs
qa_dataset = []

for _, row in df.iterrows():
    name = row["Name"].title()
    scientific_name = row["Scientific Name"]

    # 🔹 Generate Q&A for each feature
    for feature, question_template in QUESTION_TEMPLATES.items():
        question = question_template.format(name=name)
        retrieved_chunk = row[feature]
        enhanced_response = RESPONSE_TEMPLATES[feature].format(name=name, value=retrieved_chunk)

        qa_dataset.append({
            "question": question,
            "retrieved_chunk": retrieved_chunk,
            "enhanced_response": enhanced_response
        })

    # 🔹 Add Full Description
    full_description = format_bird_description(row)
    qa_dataset.append({
        "question": f"Tell me about the {name}.",
        "retrieved_chunk": full_description,
        "enhanced_response": full_description
    })

# 🔹 Save Dataset as JSON
with open(qa_dataset_file, "w", encoding="utf-8") as f:
    json.dump(qa_dataset, f, indent=4)

print(f"✅ Q&A Dataset generated and saved as {qa_dataset_file}")
