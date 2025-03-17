import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define Model Path (Ensure this is correct)
MODEL_PATH = "E:/IIT Lecs & Tutorials/Y1 SEM1/DSGP/Information genaration/FeatherFind/BirdInfoGenerater/Model/"

# Load GPT-2 Model and Tokenizer
def load_gpt2_model():
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model directory '{MODEL_PATH}' does not exist.")
            return None, None

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
        model.eval()  # Set model to evaluation mode
        print("✅ GPT-2 model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        return None, None

# Generate AI-Based Bird Description
def generate_dynamic_description(bird_data):
    tokenizer, model = load_gpt2_model()

    if model is None or tokenizer is None:
        return "Error: AI model not available."

    # Construct input prompt
    prompt = f"The {bird_data['Name']} is {bird_data.get('Distinctive Features', 'a unique bird')}." \
             f" It is found in {bird_data.get('Habitat', 'various locations')} and is known for {bird_data.get('Behavior', 'interesting behavior')}."

    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate description
    with torch.no_grad():
        output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id 
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
