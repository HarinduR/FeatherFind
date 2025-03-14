import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# File Paths
faiss_index_file = "../vector_database/birds_faiss_index.bin"
embeddings_file = "../RAG_dataset/bird_embeddings.npy"
bird_mapping_file = "../RAG_dataset/bird_mapping.npy"

# Load Data
print("🔄 Loading FAISS index and embeddings...")
index = faiss.read_index(faiss_index_file)
embeddings = np.load(embeddings_file, allow_pickle=True)
bird_mapping = np.load(bird_mapping_file, allow_pickle=True)

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_bird(query, top_k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)

    # ✅ Find the best-matching bird
    best_bird_name, best_description = bird_mapping[indices[0][0]]
    return best_bird_name, best_description

# 🔹 Run Test Queries
if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Enter a bird name or characteristic (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        bird_name, full_description = retrieve_bird(user_query)

        print(f"\n🔎 Best Match: {bird_name}")
        print("\n📌 Complete Description:")
        print(full_description)
