from retrieve_answer import retrieve_answer
from response_generator import generate_response

if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Enter a bird question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        retrieved_chunk = retrieve_answer(user_query)
        print(f"\n🔍 Retrieved Fact: {retrieved_chunk}")

        final_response = generate_response(user_query, retrieved_chunk)
        print(f"\n🦜 GPT-3.5 Response: {final_response}")
