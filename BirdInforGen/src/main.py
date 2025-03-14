import sys
from retrieve_answer import retrieve_answer
from response_generator import generate_gpt2_response

def process_query(user_query):
    
    retrieved_chunk, matched_question = retrieve_answer(user_query)

    final_response = generate_gpt2_response(user_query, retrieved_chunk)

    return final_response

if __name__ == "__main__":
    print("\nBird Information Generator (RAG System) is Running...")

    while True:
        user_query = input("\nEnter a bird question (or type 'exit' to quit): ").strip()

        if user_query.lower() == "exit":
            print("Exiting the system. Goodbye!")
            sys.exit()

        response = process_query(user_query)
        print(f"\n**Final Response:** {response}")
