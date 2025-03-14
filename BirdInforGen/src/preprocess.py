# import pandas as pd
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import spacy
# import os

# nlp = spacy.load("en_core_web_sm")

# # Load and Clean Dataset
# def preprocess_data(input_file, output_file):
#     df = pd.read_csv(input_file, encoding="utf-8")
#     df.fillna("Information not available", inplace=True)

#     for col in ["Name", "Scientific Name", "Habitat", "Size", "Distinctive Features", "Behavior", "Conservation Status", "Range"]:
#         df[col] = df[col].str.strip().str.lower()

#     df.drop_duplicates(subset=["Name"], keep="first", inplace=True)
#     df.to_csv(output_file, index=False, encoding="utf-8")
#     return df

# # Define Predefined Template for Bird Description
# def format_bird_description(row):
#     return f"""The {row['Name'].title()} (*{row['Scientific Name']}*) is a {row['Size']} bird, commonly found in {row['Habitat']}.
#     It is recognized by its {row['Distinctive Features']}. 
#     This species is known to be {row['Behavior']}. 
#     It has a conservation status of "{row['Conservation Status']}" and is primarily found in {row['Range']}."""

# # Chunk Text for Retrieval (Using SpaCy)
# def chunk_text(text, chunk_size=2):
#     doc = nlp(text)
#     sentences = [sent.text for sent in doc.sents]
#     return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# # Convert Text to Embeddings & Flatten Data
# def generate_embeddings(df, model):
#     df["Formatted_Text"] = df.apply(format_bird_description, axis=1)
#     df["Chunks"] = df["Formatted_Text"].apply(chunk_text)

#     all_embeddings = []
#     all_text_chunks = []
#     bird_mapping = []  # ✅ Addition: Store bird name for each chunk

#     for index, row in df.iterrows():
#         bird_name = row["Name"].title()  # Store proper bird name
#         for chunk in row["Chunks"]:
#             chunk_embedding = model.encode(chunk)  # Convert text chunks into embeddings
#             all_embeddings.append(chunk_embedding)
#             all_text_chunks.append(chunk)  # Save text chunks for retrieval
#             bird_mapping.append(bird_name)  # ✅ Map chunk to bird name

#     return df, all_embeddings, all_text_chunks, bird_mapping

# # Store Embeddings in FAISS
# def store_in_faiss(embeddings, index_file):
#     embeddings = np.array(embeddings, dtype=np.float32)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     faiss.write_index(index, index_file)

# # 🔹 Retrieve Bird Information (Fixed Version)
# def search_bird(query, model, index_file, text_chunks, bird_mapping):
#     index = faiss.read_index(index_file)
#     query_embedding = model.encode([query])
#     _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=5)

#     # ✅ Group chunks by bird name
#     bird_chunks = {}
#     for idx in indices[0]:
#         bird_name = bird_mapping[idx]
#         if bird_name not in bird_chunks:
#             bird_chunks[bird_name] = []
#         bird_chunks[bird_name].append(text_chunks[idx])

#     # ✅ Return only the best-matching bird description
#     best_bird = max(bird_chunks, key=lambda bird: len(bird_chunks[bird]))
#     return best_bird, bird_chunks[best_bird]

# # 🔹 Run the pipeline
# if __name__ == "__main__":
#     dataset_file = "../RAG_dataset/dataset.csv"
#     cleaned_file = "../RAG_dataset/cleaned_birds.csv"
#     embeddings_file = "../RAG_dataset/bird_embeddings.npy"
#     faiss_index_file = "../vector_database/birds_faiss_index.bin"
#     text_chunks_file = "../RAG_dataset/text_chunks.npy"  
#     bird_mapping_file = "../RAG_dataset/bird_mapping.npy"  # ✅ Addition: Save bird mapping

#     print("🔄 Processing dataset...")
#     df = preprocess_data(dataset_file, cleaned_file)

#     print("🔄 Generating embeddings...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     df, all_embeddings, all_text_chunks, bird_mapping = generate_embeddings(df, model)

#     # ✅ Save embeddings and mappings correctly
#     np.save(embeddings_file, np.array(all_embeddings, dtype=np.float32))
#     np.save(text_chunks_file, np.array(all_text_chunks, dtype=object))
#     np.save(bird_mapping_file, np.array(bird_mapping, dtype=object))  # ✅ Save bird name mapping

#     print(f"✅ Text chunks saved to: {text_chunks_file}")
#     print(f"✅ Bird mapping saved to: {bird_mapping_file}")

#     print("🔄 Storing embeddings in FAISS...")
#     store_in_faiss(all_embeddings, faiss_index_file)

#     print("✅ FAISS index saved successfully!")

#     # ✅ Debugging: Check if everything is working
#     if os.path.exists(faiss_index_file) and os.path.exists(text_chunks_file) and os.path.exists(bird_mapping_file):
#         print("✅ FAISS, text chunks, and bird mapping are properly stored.")
#     else:
#         print("❌ ERROR: Some files are missing!")

#     # Example Query
#     user_query = "Tell me about the Kingfisher"
#     bird_name, retrieved_chunks = search_bird(user_query, model, faiss_index_file, all_text_chunks, bird_mapping)

#     print(f"\n🔍 Retrieved Bird: {bird_name}")
#     for i, chunk in enumerate(retrieved_chunks):
#         print(f"{i+1}. {chunk}")




import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# File Paths
dataset_file = "../RAG_dataset/dataset.csv"
cleaned_file = "../RAG_dataset/cleaned_birds.csv"
embeddings_file = "../RAG_dataset/bird_embeddings.npy"
faiss_index_file = "../vector_database/birds_faiss_index.bin"
bird_mapping_file = "../RAG_dataset/bird_mapping.npy"

# 🔹 Load and Clean Dataset
def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file, encoding="utf-8")
    df.fillna("unknown", inplace=True) 

    for col in ["Name", "Scientific Name", "Habitat", "Size", "Distinctive Features", "Behavior", "Conservation Status", "Range"]:
        df[col] = df[col].str.strip().str.lower()

    df.drop_duplicates(subset=["Name"], keep="first", inplace=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    return df

# 🔹 Handle Unknown Values
def clean_text(value, field_name):
    """Replaces 'unknown' or 'information not available' with meaningful defaults."""
    if value in ["unknown", "information not available"]:
        defaults = {
            "Size": "a bird of varying sizes",
            "Habitat": "various environments",
            "Distinctive Features": "unique physical traits",
            "Behavior": "different behaviors depending on location",
            "Range": "several regions around the world",
            "Conservation Status": "not well-documented"
        }
        return defaults.get(field_name, "unknown")  # Return meaningful default
    return value

# 🔹 Create Full Bird Descriptions (Improved)
def format_bird_description(row):
    size = clean_text(row["Size"], "Size")
    habitat = clean_text(row["Habitat"], "Habitat")
    features = clean_text(row["Distinctive Features"], "Distinctive Features")
    behavior = clean_text(row["Behavior"], "Behavior")
    conservation = clean_text(row["Conservation Status"], "Conservation Status")
    range_info = clean_text(row["Range"], "Range")

    description = f"""The {row['Name'].title()} (*{row['Scientific Name']}*) is {size}, commonly found in {habitat}. It is recognized by its {features}. This species is known to be {behavior}. It has a conservation status of "{conservation}" and is primarily found in {range_info}."""
    
    return description.replace("\n    ", " ")  

# 🔹 Generate Embeddings for Each Bird
def generate_embeddings(df, model):
    df["Full_Description"] = df.apply(format_bird_description, axis=1)

    all_embeddings = []
    bird_mapping = []  # Store bird names separately

    for index, row in df.iterrows():
        description = row["Full_Description"]
        bird_name = row["Name"].title()

        description_embedding = model.encode(description) 
        all_embeddings.append(description_embedding)
        bird_mapping.append((bird_name, description))  # Store bird name & description

    return df, all_embeddings, bird_mapping

# 🔹 Store Embeddings in FAISS
def store_in_faiss(embeddings, index_file):
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)

# 🔹 Run the Pipeline
if __name__ == "__main__":
    print("🔄 Processing dataset...")
    df = preprocess_data(dataset_file, cleaned_file)

    print("🔄 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df, all_embeddings, bird_mapping = generate_embeddings(df, model)

    # ✅ Save embeddings and mappings correctly
    np.save(embeddings_file, np.array(all_embeddings, dtype=np.float32))
    np.save(bird_mapping_file, np.array(bird_mapping, dtype=object))

    print("🔄 Storing embeddings in FAISS...")
    store_in_faiss(all_embeddings, faiss_index_file)

    print("✅ FAISS index saved successfully!")
