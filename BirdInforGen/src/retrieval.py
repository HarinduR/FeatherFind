# import pandas as pd
# import faiss
# import numpy as np
# from rapidfuzz import process
# from gpt2_enhancer import load_gpt2_model, enhance_description

# gpt2_pipe = load_gpt2_model()

# dataset_file = "../dataset/cleaned.csv"
# df = pd.read_csv(dataset_file)

# df["Name"] = df["Name"].str.strip().str.lower()

# print(f"Dataset Loaded with {len(df)} birds.")

# # Convert Bird Names into Numeric Vectors (Using Hashing)
# def create_faiss_index(df):
#     bird_names = df["Name"].tolist()  

#     # Convert bird names to numerical vectors 
#     bird_vectors = np.array([hash(name) % (10**6) for name in bird_names], dtype=np.int64).reshape(-1, 1)

#     # Create FAISS Index
#     index = faiss.IndexFlatL2(1)
#     index.add(bird_vectors)

#     return index, bird_names

# # Initialize FAISS Index
# faiss_index, bird_names_list = create_faiss_index(df)

# print(f"FAISS Index Created with {len(bird_names_list)} birds.")

# def find_closest_bird(query):
#     query = query.lower().strip()  

#     # Use FAISS for Fast Search
#     query_vector = np.array([[hash(query) % (10**6)]], dtype=np.int64)
#     _, indices = faiss_index.search(query_vector, 5)

#     top_candidates = [bird_names_list[i] for i in indices[0]]

#     # Use RapidFuzz for Better Matching
#     best_match = process.extractOne(query, top_candidates)

#     if best_match and best_match[1] > 85:
#         return best_match[0]

#     # 🔹 Step 3: If FAISS Fails, Search the Full Dataset
#     print("FAISS failed, switching to full dataset search...")
#     best_match_full = process.extractOne(query, bird_names_list)

#     if best_match_full and best_match_full[1] > 75:
#         return best_match_full[0]

#     return None  

# # Retrieve Bird Information
# def get_bird_info(bird_name):
#     bird_data = df[df["Name"] == bird_name]
#     return bird_data.to_dict(orient="records")[0] if not bird_data.empty else None

# # Format Bird Description
# def format_bird_description(bird_data):
#     if not bird_data:
#         return "No bird information found."

#     def safe_get(field):
#         return bird_data.get(field, "Information not available")

#     template = (
#         f"The {bird_data['Name'].title()} (*{bird_data['Scientific Name']}*) is a bird species "
#         f"commonly found in {safe_get('Habitat')}. It is best known for its {safe_get('Distinctive Features')}. "
#         f"Classified as {safe_get('Conservation Status')}, this bird is often seen {safe_get('Behavior')}."
#     )

#     return template.strip()

# def get_bird_description(bird_name):

#     matched_bird = find_closest_bird(bird_name)

#     if matched_bird:

#         bird_info = get_bird_info(matched_bird)

#         static_description = format_bird_description(bird_info)

#         dynamic_description = enhance_description(gpt2_pipe, matched_bird, static_description)

#         return {
#             "bird_name": matched_bird.title(),
#             "static_description": static_description,
#             "enhanced_description": dynamic_description
#         }
#     else:
#         return {
#             "error": "No matching bird found. Try entering a different bird name."
#         }

# if __name__ == "__main__":
#     user_input = input("\n\nEnter a bird name: ")
#     response = get_bird_description(user_input)

#     if "error" in response:
#         print(response["error"])
#     else:
#         print(f"\n{response['bird_name']} Description:")
#         print(response["enhanced_description"])
