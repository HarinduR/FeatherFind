import json
from transformers import pipeline

gpt2_pipe = pipeline("text-generation", model="../model/gpt2_finetuned")

with open("../RAG_dataset/qa_dataset.json", "r", encoding="utf-8") as f:
    test_dataset = json.load(f)

correct = 0
total = len(test_dataset)

for entry in test_dataset[:50]: 
    question = entry["question"]
    retrieved_chunk = entry["retrieved_chunk"]
    expected_response = entry["enhanced_response"]

    prompt = f"Instruction: Answer using the given fact only.\nQuestion: {question}\nRetrieved Fact: {retrieved_chunk}\nAnswer:"
    output = gpt2_pipe(prompt, max_new_tokens=50, temperature=0.2, top_p=0.6, return_full_text=False, do_sample=True)

    generated_response = output[0]["generated_text"].strip()

    if expected_response.lower() in generated_response.lower():
        correct += 1

print(f"\n✅ GPT-2 Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
