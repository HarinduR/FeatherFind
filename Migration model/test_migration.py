import pandas as pd
from migration_prediction import interpret_migration_prediction

# ✅ Load Processed Dataset
migration_df = pd.read_csv(r'C:\Users\Deshan\Documents\IIT LECS\DSGP Models\Migration model\data\migration_data.csv')

# ✅ Define Test Cases
test_cases = [
    {
        "name": "Location-Based Query",
        "input": {"species": "Blue-tailed Bee-eater", "date": "2025-03-12", "time": "07:00"},
        "expected": "The *Blue-tailed Bee-eater* is most likely to be found at"
    },
    {
        "name": "Species-Based Query",
        "input": {"location": "Muruthawela Lake", "date": "2025-04-15", "time": "17:00"},
        "expected": "At **Muruthawela Lake** on **2025-04-15 at 17:00**, you are likely to see"
    },
    {
        "name": "Time-Based Query",
        "input": {"location": "Bundala National Park"},
        "expected": "The best time for birdwatching at **Bundala National Park** is"
    },
    {
        "name": "Full Prediction Query",
        "input": {"date": "2025-12-25", "time": "06:00"},
        "expected": "On **2025-12-25 at 06:00**, the best locations are"
    }
]

# ✅ Run Tests
for test in test_cases:
    print(f"\n🟢 Running Test: {test['name']}")
    result = interpret_migration_prediction(test['input'])
    
    # ✅ Check if expected phrase exists in response
    if test["expected"] in result:
        print(f"✅ Test PASSED! \n🔹 Output:\n{result}\n")
    else:
        print(f"❌ Test FAILED! \n🔹 Output:\n{result}\n🔺 Expected a response containing: {test['expected']}\n")
