import openai
import os
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv("")  # Store in environment variable

if not OPENAI_API_KEY:
    raise ValueError(" OpenAI API key not found. Set it in environment variables or .env file.")

openai.api_key = OPENAI_API_KEY

def generate_gpt3_response(user_query, retrieved_chunk):
    
    prompt = f"""You are an AI assistant that provides concise, well-structured responses about birds. 
    Improve the following bird-related fact while keeping it factually accurate and engaging.
    
    Question: {user_query}
    Retrieved Information: {retrieved_chunk}
    
    Enhanced Response:"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": "You are an expert in birds and wildlife."},
                      {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )

        enhanced_response = response["choices"][0]["message"]["content"].strip()
        return enhanced_response

    except Exception as e:
        print(f"GPT-3.5 API Error: {e}")
        return retrieved_chunk  

