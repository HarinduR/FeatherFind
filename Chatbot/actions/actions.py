from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from langchain.llms import GPT2
from haystack.document_stores import FAISS
from haystack.nodes import Retriever

class ActionRetrieveBirdInfo(Action):
    def name(self):
        return "action_retrieve_bird_info"

    def run(self, dispatcher, tracker, domain):
        bird_name = tracker.get_slot("bird")
        
        # Retrieve information from FAISS knowledge base
        retrieved_data = retrieve_bird_info(bird_name)

        # Generate enhanced response using GPT-2
        ai_response = generate_ai_response(bird_name, retrieved_data)

        dispatcher.utter_message(text=ai_response)
        return []

def retrieve_bird_info(bird_name):
    # FAISS retrieval logic
    return "The Kingfisher is a small bird found near water."

def generate_ai_response(bird_name, retrieved_data):
    # LangChain GPT-2 response generation
    llm = GPT2()
    return llm(f"Tell me about {bird_name}: {retrieved_data}")
