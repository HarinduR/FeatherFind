# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import os

# # File Paths
# dataset_file = "../RAG_dataset/dataset.csv"
# cleaned_file = "../RAG_dataset/cleaned_birds.csv"
# embeddings_file = "../RAG_dataset/bird_embeddings.npy"
# faiss_index_file = "../vector_database/birds_faiss_index.bin"
# bird_mapping_file = "../RAG_dataset/bird_mapping.npy"

# # 🔹 Load and Clean Dataset
# def preprocess_data(input_file, output_file):
#     df = pd.read_csv(input_file, encoding="utf-8")
#     df.fillna("unknown", inplace=True) 

#     for col in ["Name", "Scientific Name", "Habitat", "Size", "Distinctive Features", "Behavior", "Conservation Status", "Range"]:
#         df[col] = df[col].str.strip().str.lower()

#     df.drop_duplicates(subset=["Name"], keep="first", inplace=True)
#     df.to_csv(output_file, index=False, encoding="utf-8")
#     return df

# # 🔹 Handle Unknown Values
# def clean_text(value, field_name):
#     if value in ["unknown", "information not available"]:
#         defaults = {
#             "Size": "a bird of varying sizes",
#             "Habitat": "various environments",
#             "Distinctive Features": "unique physical traits",
#             "Behavior": "different behaviors depending on location",
#             "Range": "several regions around the world",
#             "Conservation Status": "not well-documented"
#         }
#         return defaults.get(field_name, "unknown")  # Return meaningful default
#     return value

# # 🔹 Create Full Bird Descriptions (Improved)
# def format_bird_description(row):
#     size = clean_text(row["Size"], "Size")
#     habitat = clean_text(row["Habitat"], "Habitat")
#     features = clean_text(row["Distinctive Features"], "Distinctive Features")
#     behavior = clean_text(row["Behavior"], "Behavior")
#     conservation = clean_text(row["Conservation Status"], "Conservation Status")
#     range_info = clean_text(row["Range"], "Range")

#     description = f"""The {row['Name'].title()} (*{row['Scientific Name']}*) is {size}, commonly found in {habitat}. It is recognized by its {features}. This species is known to be {behavior}. It has a conservation status of "{conservation}" and is primarily found in {range_info}."""
    
#     return description.replace("\n    ", " ")  

# # 🔹 Generate Embeddings for Each Bird
# def generate_embeddings(df, model):
#     df["Full_Description"] = df.apply(format_bird_description, axis=1)

#     all_embeddings = []
#     bird_mapping = []  # Store bird names separately

#     for index, row in df.iterrows():
#         description = row["Full_Description"]
#         bird_name = row["Name"].title()

#         description_embedding = model.encode(description) 
#         all_embeddings.append(description_embedding)
#         bird_mapping.append((bird_name, description))  # Store bird name & description

#     return df, all_embeddings, bird_mapping

# # 🔹 Store Embeddings in FAISS
# def store_in_faiss(embeddings, index_file):
#     embeddings = np.array(embeddings, dtype=np.float32)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     faiss.write_index(index, index_file)

# # 🔹 Run the Pipeline
# if __name__ == "__main__":
#     print("🔄 Processing dataset...")
#     df = preprocess_data(dataset_file, cleaned_file)

#     print("🔄 Generating embeddings...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     df, all_embeddings, bird_mapping = generate_embeddings(df, model)

#     # ✅ Save embeddings and mappings correctly
#     np.save(embeddings_file, np.array(all_embeddings, dtype=np.float32))
#     np.save(bird_mapping_file, np.array(bird_mapping, dtype=object))

#     print("🔄 Storing embeddings in FAISS...")
#     store_in_faiss(all_embeddings, faiss_index_file)

#     print("✅ FAISS index saved successfully!")
