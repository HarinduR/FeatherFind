import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# ✅ Load FAISS Index & Bird Mapping
faiss_index_file = "vector_database/birds_faiss_index.bin"
bird_mapping_file = "RAG_dataset/bird_mapping.npy"
embeddings_file = "RAG_dataset/bird_embeddings.npy"

# ✅ Load Sentence Transformer Model (Same as Training)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load FAISS Index
index = faiss.read_index(faiss_index_file)
bird_mapping = np.load(bird_mapping_file, allow_pickle=True)  # List of (bird_name, description)

# 🔹 Function: Retrieve Bird Information
def retrieve_bird_info(query, top_k=1):
    """Retrieve best-matching bird description based on user query."""
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)

    # Get best match
    best_match = bird_mapping[indices[0][0]]
    bird_name, bird_description = best_match

    return bird_name, bird_description
