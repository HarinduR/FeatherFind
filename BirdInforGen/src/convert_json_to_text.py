import json

with open("../RAG_dataset/gpt2_finetune_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

OUTPUT_FILE = "gpt2_training_data.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(f"### Instruction:\n{entry['instruction']}\n")
        f.write(f"### Question:\n{entry['question']}\n")
        f.write(f"### Retrieved Chunk:\n{entry['retrieved_chunk']}\n")
        f.write(f"### Enhanced Response:\n{entry['enhanced_response']}\n")
        f.write("\n---\n")

print(f"✅ Training data saved as {OUTPUT_FILE}")
