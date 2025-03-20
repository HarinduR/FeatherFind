from try_gpt3 import generate_gpt3_response
from retrieve_answer import retrieve_answer

print("Bird Information Generator (GPT-3.5 Version) Running...\n")

while True:
    user_query = input(" Enter a bird question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    retrieved_chunk, matched_question = retrieve_answer(user_query)
    
    print(f"\nRetrieved Chunk: {retrieved_chunk}")

    enhanced_response = generate_gpt3_response(user_query, retrieved_chunk)
    
    print(f"\n✅ GPT-3.5 Enhanced Response: {enhanced_response}\n")
