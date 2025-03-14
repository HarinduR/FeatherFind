import pandas as pd
import json

CSV_FILE = "../RAG_dataset/cleaned.csv" 
OUTPUT_JSON = "../RAG_dataset/gpt2_finetune_dataset.json" 

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

df = pd.read_csv(CSV_FILE)

dataset = []

for _, row in df.iterrows():
    name = row["Name"]
    scientific_name = row["Scientific Name"]
    
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
        "enhanced_response": full_description 
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"✅ Dataset generated and saved as {OUTPUT_JSON}")
