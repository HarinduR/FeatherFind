import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

qa_dataset_file = "../RAG_dataset/qa_dataset.json"
faiss_index_file = "../vector_database/faiss_index.bin"
question_embeddings_file = "../vector_database/question_embeddings.npy"
answer_embeddings_file = "../vector_database/answer_embeddings.npy"
qa_mapping_file = "../vector_database/qa_mapping.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(qa_dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

questions = []
answers = []

for entry in dataset:
    questions.append(entry["question"])
    answers.append(entry["retrieved_chunk"])

question_embeddings = model.encode(questions, normalize_embeddings=True)
answer_embeddings = model.encode(answers, normalize_embeddings=True)

dimension = question_embeddings.shape[1]
assert dimension == answer_embeddings.shape[1], "Mismatch in embedding dimensions!"

faiss.normalize_L2(question_embeddings)
faiss.normalize_L2(answer_embeddings)

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(answer_embeddings)  

faiss.write_index(faiss_index, faiss_index_file)
np.save(question_embeddings_file, question_embeddings)
np.save(answer_embeddings_file, answer_embeddings)
np.save(qa_mapping_file, np.array(list(zip(questions, answers)), dtype=object))

print("✅ FAISS index & embeddings saved successfully!")
