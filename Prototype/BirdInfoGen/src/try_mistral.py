import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ✅ Load Mistral 7B Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# ✅ Define Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_mistral_response(user_query, retrieved_chunk):
    """
    Enhances the retrieved answer using the Mistral 7B model.
    If the query is a full description request, return retrieved_chunk directly.
    Otherwise, use the LLM to enhance the response.
    """
    if "tell me about" in user_query.lower():  
        return retrieved_chunk  

    prompt = (
        f"Instruction: Improve the following response with engaging and informative wording.\n\n"
        f"Question: {user_query}\n"
        f"Retrieved Chunk: {retrieved_chunk}\n"
        f"Enhanced Response:"
    )

    try:
        output = pipe(prompt, max_new_tokens=80, temperature=0.7, top_p=0.9)
        enhanced_response = output[0]["generated_text"].split("Enhanced Response:")[-1].strip()
        return enhanced_response
    except Exception as e:
        print(f"⚠️ Mistral 7B Generation Error: {e}")
        return retrieved_chunk  # Fallback to original answer if LLM fails

