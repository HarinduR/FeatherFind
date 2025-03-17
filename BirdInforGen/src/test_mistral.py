from retrieve_answer import retrieve_answer
from try_mistral import generate_mistral_response

print("🦜 Bird Information Generator (RAG System) is Running...\n")

while True:
    user_query = input("🔍 Enter a bird question (or type 'exit' to quit): ").strip()
    
    if user_query.lower() == "exit":
        print("👋 Exiting Bird Information Generator. Goodbye!")
        break

    retrieved_chunk = retrieve_answer(user_query)
    final_response = generate_mistral_response(user_query, retrieved_chunk)

    print(f"\n🦜 **Final Response:** {final_response}\n")
