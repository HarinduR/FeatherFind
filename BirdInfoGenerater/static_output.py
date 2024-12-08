import pandas as pd

def load_template(template_path):
    with open(template_path, "r") as file:
        template = file.read()
    return template

def generate_description(template, bird_data):
    return template.format(
        Name=bird_data["Name"],
        Scientific_Name=bird_data["Scientific Name"],
        Conservation_Status=bird_data["Conservation Status"],
        Distinctive_Features=bird_data["Distinctive Features"],
        Size=bird_data["Size"] if bird_data["Size"] != "Unknown" else "an unknown size",
        Habitat=bird_data["Habitat"] if bird_data["Habitat"] != "Unknown" else "an unknown habitat",
        Behavior=bird_data["Behavior"] if bird_data["Behavior"] != "Unknown" else "behavior details not available",
        Range=bird_data["Range"] if bird_data["Range"] != "Unknown" else "range details not available"
    )

try:
    # Load the dataset
    dataset_path = r"../Dataset/dataset.csv"
    birds_df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError as e:
    raise e
except Exception as e:
    raise e

try:
    # Load the template
    template_path = r"../Templates/template1.txt"
    template = load_template(template_path)
    print("Template loaded successfully.")
except FileNotFoundError as e:
    raise e
except Exception as e:
    raise e


bird_name = input("Enter the bird name: ").strip()

# Search for the bird in the dataset
bird_row = birds_df[birds_df["Name"].str.contains(bird_name, case=False, na=False)]
if not bird_row.empty:
    bird_data = bird_row.iloc[0].to_dict()
    description = generate_description(template, bird_data)
    print("\n\nGenerated Description:")
    print(description)
else:
    print(f"No information found for the bird '{bird_name}'.")
