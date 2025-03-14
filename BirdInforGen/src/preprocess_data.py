import pandas as pd

input_file = "../dataset/dataset.csv"  
output_file = "../RAG_dataset/cleaned.csv"  

print("Loading dataset...")
df = pd.read_csv(input_file, encoding="utf-8") 

df.fillna("information not available", inplace=True)

df["Name"] = df["Name"].str.strip().str.lower()
df["Scientific Name"] = df["Scientific Name"].str.strip().str.lower()
df["Conservation Status"] = df["Conservation Status"].str.strip().str.lower()
df["Distinctive Features"] = df["Distinctive Features"].str.strip().str.lower()
df["Size"] = df["Size"].str.strip().str.lower()
df["Habitat"] = df["Habitat"].str.strip().str.lower()
df["Behavior"] = df["Behavior"].str.strip().str.lower()
df["Range"] = df["Range"].str.strip().str.lower()

df.drop_duplicates(subset=["Name"], keep="first", inplace=True)

df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Dataset cleaned and saved as: {output_file}")
print(f"Total Birds: {len(df)}")
print(df.head()) 
