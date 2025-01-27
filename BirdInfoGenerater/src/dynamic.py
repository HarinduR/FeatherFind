from transformers import pipeline

# Load GPT-2 model for text generation
def generate_dynamic_description(bird_data):
    generator = pipeline("text-generation", model="gpt-2")

    # Prepare input prompt for AI model
    prompt = f"Describe the bird {bird_data['Name']} with distinctive features: {bird_data['Distinctive Features']}. " \
             f"It is found in {bird_data['Habitat']} and behaves like {bird_data['Behavior']}."

    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]["generated_text"]
