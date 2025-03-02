import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_gpt2_model(model_path="../model/gpt2-bird-finetuned"):
    print("Loading GPT-2 Model...")
    try:

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        gpt2_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1 
        )

        print("GPT-2 Model Loaded Successfully!")
        return gpt2_pipe
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        return None

def enhance_description(gpt2_pipe, bird_name, static_description):
    if gpt2_pipe is None:
        return f"{static_description}\n\n GPT-2 model failed to load. Using only static description."

    prompt = (
        f"The following is an accurate description of the {bird_name}. Keep all facts unchanged but add a simple, engaging final paragraph:\n\n"
        f"{static_description}\n\n"
        f"Now, add a **single concise paragraph** to make this description more engaging. Do NOT introduce any new facts."
    )

    try:
        gpt2_output = gpt2_pipe(
            prompt,
            max_new_tokens=40,   # Short, readable output
            num_return_sequences=1,
            top_p=0.6,           # Keeps it focused, less randomness
            top_k=30,            # Restricts choices for clarity
            temperature=0.3,     # Keeps output structured and fact-based
            repetition_penalty=1.4,  # Avoids repetitive words
            return_full_text=False  # Ensures clean output
        )

        ai_generated_paragraph = gpt2_output[0]["generated_text"].strip()

        final_description = f"{static_description}\n\nDid you know? {ai_generated_paragraph}"

        return final_description

    except Exception as e:
        return f"{static_description}\n\nGPT-2 generation error: {e}"
