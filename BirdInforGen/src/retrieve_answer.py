import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = faiss.read_index("../vector_database/faiss_index.bin")

qa_mapping = np.load("../vector_database/qa_mapping.npy", allow_pickle=True)

def retrieve_answer(user_query):

    query_embedding = model.encode([user_query])

    _, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=5)

    best_match = None
    best_score = -1

    print("\n🔍 Top 5 Matches (Checking for the Best One):")
    
    for idx in indices[0]:  
        matched_question, matched_answer = qa_mapping[idx]

        match_vector = model.encode([matched_question])
        similarity = cosine_similarity(query_embedding, match_vector)[0][0]  

        print(f"🔹 Matched Question: {matched_question} | Similarity Score: {similarity:.4f}")

        # ✅ Ensure correct bird is matched
        matched_bird_name = matched_question.split(" of the ")[-1].replace("?", "").strip()
        user_bird_name = user_query.split(" of the ")[-1].replace("?", "").strip()

        if user_bird_name.lower() == matched_bird_name.lower():
            if similarity > best_score:
                best_score = similarity
                best_match = matched_answer

    return best_match if best_match else "No relevant information found."

