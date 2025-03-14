import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = faiss.read_index("../vector_database/faiss_index.bin")
qa_mapping = np.load("../vector_database/qa_mapping.npy", allow_pickle=True)

def retrieve_answer(user_query):
    """Retrieve the best-matching bird answer using Hybrid Search (FAISS + Fuzzy Matching)."""

    query_embedding = model.encode([user_query])

    _, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=1)

    best_faiss_match = None
    best_faiss_score = -1

    print("\n FAISS Best Match:")
    
    idx = indices[0][0] 
    matched_question, matched_answer = qa_mapping[idx] 

    match_vector = model.encode([matched_question])
    similarity = cosine_similarity(query_embedding, match_vector)[0][0]  

    print(f" FAISS Matched Question: {matched_question} | Score: {similarity:.4f}")

    questions_list = [entry[0] for entry in qa_mapping]
    
    fuzzy_match = process.extractOne(user_query, questions_list)  
    best_keyword_match, keyword_score, _ = fuzzy_match 

    print(f"RapidFuzz Match: {best_keyword_match} | Score: {keyword_score:.4f}")

    if keyword_score > similarity * 100:  
        best_answer = next(ans for q, ans in qa_mapping if q == best_keyword_match)
        matched_question = best_keyword_match  
        print("✅ Using RapidFuzz Answer (Higher Score)")
    else:
        best_answer = matched_answer
        print("✅ Using FAISS Answer (Higher Score)")

    return best_answer, matched_question 

