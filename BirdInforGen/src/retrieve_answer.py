import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load FAISS Index
faiss_index = faiss.read_index("../vector_database/faiss_index.bin")

# ✅ Load Stored Questions & Answers
qa_mapping = np.load("../vector_database/qa_mapping.npy", allow_pickle=True)

# ✅ Retrieve Answer with Improved Matching
def retrieve_answer(user_query):
    """Retrieve relevant bird info by searching vectorized questions"""

    # ✅ Convert User Query to Vector
    query_embedding = model.encode([user_query])

    # ✅ Perform FAISS Search (Get Top 5 Matches)
    _, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=5)

    # ✅ Retrieve Top 5 Matches
    best_match = None
    best_score = -1

    print("\n🔍 Top 5 Matches (Checking for the Best One):")
    
    for idx in indices[0]:  # Loop through top 5 matches
        matched_question, matched_answer = qa_mapping[idx]

        # ✅ Compute Cosine Similarity Score
        match_vector = model.encode([matched_question])
        similarity = cosine_similarity(query_embedding, match_vector)[0][0]  # Score between 0-1

        print(f"🔹 Matched Question: {matched_question} | Similarity Score: {similarity:.4f}")

        # ✅ Choose the Best Matching Answer
        if similarity > best_score:
            best_score = similarity
            best_match = matched_answer

    # ✅ Return Best Match
    return best_match if best_match else "No relevant information found."

# ✅ Test Retrieval
user_query = "where does White-Throated Kingfisher live in"
retrieved_info = retrieve_answer(user_query)
print(f"\n✅ Final Retrieved Bird Information: {retrieved_info}")
