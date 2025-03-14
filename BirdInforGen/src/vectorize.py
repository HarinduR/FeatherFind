import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

# 🔹 File Paths
qa_dataset_file = "../RAG_dataset/qa_dataset.json"
faiss_index_file = "../vector_database/faiss_index.bin"
question_embeddings_file = "../vector_database/question_embeddings.npy"
answer_embeddings_file = "../vector_database/answer_embeddings.npy"
qa_mapping_file = "../vector_database/qa_mapping.npy"

# 🔹 Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Load Q&A Dataset
with open(qa_dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 🔹 Initialize Lists
questions = []
answers = []

# 🔹 Store Questions and Answers Separately
for entry in dataset:
    questions.append(entry["question"])
    answers.append(entry["retrieved_chunk"])  # Ensures we retrieve specific chunks, NOT full descriptions

# 🔹 Convert to Embeddings
question_embeddings = model.encode(questions)  # Vectorize questions
answer_embeddings = model.encode(answers)  # Vectorize retrieved chunks

# 🔹 Create FAISS Index
dimension = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)

# 🔹 Add Answer Embeddings (for retrieval)
faiss_index.add(answer_embeddings)

# 🔹 Save FAISS Index & Embeddings
faiss.write_index(faiss_index, faiss_index_file)
np.save(question_embeddings_file, question_embeddings)
np.save(answer_embeddings_file, answer_embeddings)
np.save(qa_mapping_file, np.array(list(zip(questions, answers)), dtype=object))

print("✅ FAISS index & embeddings saved successfully!")
