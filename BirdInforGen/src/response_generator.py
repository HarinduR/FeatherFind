import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
from transformers import pipeline

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = faiss.read_index("../vector_database/faiss_index.bin")
qa_mapping = np.load("../vector_database/qa_mapping.npy", allow_pickle=True)

gpt2_pipe = pipeline("text-generation", model="../model/gpt2_finetuned", tokenizer="../model/gpt2_finetuned")

def retrieve_and_enhance_answer(user_query):

    query_embedding = model.encode([user_query])
    _, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=1)

    print("\n FAISS Best Match:")
    
    idx = indices[0][0] 
    matched_question, retrieved_chunk = qa_mapping[idx]

    if isinstance(retrieved_chunk, tuple):  
        retrieved_chunk = retrieved_chunk[0] 

    match_vector = model.encode([matched_question])
    similarity = cosine_similarity(query_embedding, match_vector)[0][0]  

    print(f" FAISS Matched Question: {matched_question} | Score: {similarity:.4f}")

    questions_list = [entry[0] for entry in qa_mapping]
    
    fuzzy_match = process.extractOne(user_query, questions_list) 
    best_keyword_match, keyword_score, _ = fuzzy_match  

    print(f"RapidFuzz Match: {best_keyword_match} | Score: {keyword_score:.4f}")


    if similarity < 0.5 and keyword_score < 70:  
        return "Sorry, I don't have information about that. Can you ask about a bird?"

    if keyword_score > similarity * 100:  
        retrieved_chunk = next(ans for q, ans in qa_mapping if q == best_keyword_match)
        print("✅ Using RapidFuzz Answer (Higher Score)")
    else:
        print("✅ Using FAISS Answer (Higher Score)")

    enhanced_response = generate_gpt2_response(user_query, matched_question, retrieved_chunk)

    return enhanced_response if enhanced_response else "No relevant information found."

def generate_gpt2_response(user_query, matched_question, retrieved_chunk):

    if matched_question.lower().startswith("tell me about"):
        return retrieved_chunk  

    prompt = f"""Instruction: Improve the following response with engaging and informative wording while keeping the facts unchanged.

    Question: {user_query}
    
    Retrieved Chunk: {retrieved_chunk}
    
    Enhanced Response:"""

    try:
        output = gpt2_pipe(prompt, max_new_tokens=50, temperature=0.5, top_p=0.7)
        enhanced_response = output[0]["generated_text"].split("Enhanced Response:")[-1].strip()

        if not enhanced_response or len(enhanced_response.split()) < 5:  
            print("GPT-2 generated an incomplete response. Falling back to original answer.")
            return retrieved_chunk  

        return enhanced_response  

    except Exception as e:
        print(f"❌ GPT-2 Generation Error: {e}")
        return "Sorry, I couldn't generate an enhanced response. Here’s the original information: " + retrieved_chunk  
