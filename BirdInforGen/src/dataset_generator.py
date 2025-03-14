import pandas as pd
import json

# 🔹 File Paths
CSV_FILE = "../RAG_dataset/dataset.csv" 
OUTPUT_JSON = "../RAG_dataset/gpt2_finetune_dataset.json" 

# 🔹 Template for dataset generation
QUESTION_TEMPLATES = {
    "Scientific Name": "What is the scientific name of the {name}?",
    "Conservation Status": "What is the conservation status of the {name}?",
    "Distinctive Features": "What are the distinctive features of the {name}?",
    "Size": "What size is the {name}?",
    "Habitat": "What is the habitat of the {name}?",
    "Behavior": "What is the behavior of the {name}?",
    "Range": "What is the range of the {name}?"
}


FINAL_DESCRIPTION_TEMPLATE = "Tell me about the {name}."

RESPONSE_TEMPLATES = {
    "Scientific Name": "The scientific name of the {name} is *{value}*.",
    "Conservation Status": "The {name} is classified as '{value}', meaning it is not currently endangered.",
    "Distinctive Features": "The {name} is known for {value}.",
    "Size": "The {name} is {value}, making it easy to identify.",
    "Habitat": "The {name} is commonly found in {value}.",
    "Behavior": "The {name} is known to be {value}.",
    "Range": "The {name} is primarily found in {value}.",
}

# 🔹 Load Dataset
df = pd.read_csv(CSV_FILE)

# 🔹 Convert CSV into JSON format for GPT-2 fine-tuning
dataset = []

for _, row in df.iterrows():
    name = row["Name"]
    scientific_name = row["Scientific Name"]
    
    # 🔹 Generate prompts for each feature
    for feature, question_template in QUESTION_TEMPLATES.items():
        question = question_template.format(name=name)
        retrieved_chunk = row[feature]
        enhanced_response = RESPONSE_TEMPLATES[feature].format(name=name, value=retrieved_chunk)

        dataset.append({
            "instruction": f"Extract only the {feature.lower()} details from the following text and answer concisely.",
            "question": question,
            "retrieved_chunk": retrieved_chunk,
            "enhanced_response": enhanced_response
        })

    # 🔹 Generate full description
    full_description = (
        f"The {name} (*{scientific_name}*) is {row['Size']}. "
        f"It is known for {row['Distinctive Features']}. "
        f"It is commonly found in {row['Habitat']}. "
        f"This bird is {row['Behavior']}. "
        f"It has a conservation status of '{row['Conservation Status']}' and is primarily found in {row['Range']}."
    )

    dataset.append({
        "instruction": "Enhance the following bird description with a smooth and engaging narrative.",
        "question": FINAL_DESCRIPTION_TEMPLATE.format(name=name),
        "retrieved_chunk": full_description,
        "enhanced_response": full_description  # This will be fine-tuned to be more engaging later
    })

# 🔹 Save JSON file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"✅ Dataset generated and saved as {OUTPUT_JSON}")
