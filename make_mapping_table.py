import pandas as pd
import os
import joblib

DATA_DIR = "data"
TRAINING_PATH = os.path.join(DATA_DIR, "Training_enriched.csv")

MODELS_DIR = "models"
urgency_map_path = os.path.join(MODELS_DIR, "disease_to_urgency.pkl")

# Load the dataset to get all diseases
df = pd.read_csv(TRAINING_PATH)
all_diseases = sorted(df["prognosis"].unique())

# Load the disease-to-urgency mapping
mapping = joblib.load(urgency_map_path)
emergency_diseases = set(mapping["emergency"])
urgent_diseases = set(mapping["urgent"])

# Build the mapping table
rows = []
for disease in all_diseases:
    if disease in emergency_diseases:
        urgency = "Emergency"
    elif disease in urgent_diseases:
        urgency = "Urgent"
    else:
        urgency = "Routine"
    rows.append([disease, urgency])

# Save as a CSV you can paste into Word
mapping_df = pd.DataFrame(rows, columns=["Disease", "Urgency"])
os.makedirs("output", exist_ok=True)
mapping_df.to_csv("output/disease_urgency_mapping.csv", index=False)

# Also print it nicely in the console
print("\nAppendix A – Full Disease-to-Urgency Mapping Table")
print("=" * 70)
print(f"{'Disease':<40} {'Urgency':<15}")
print("-" * 70)
for disease, urgency in rows:
    print(f"{disease:<40} {urgency:<15}")
print("-" * 70)
print(f"Total diseases: {len(rows)}")
print(f"  Emergency: {sum(1 for _, u in rows if u == 'Emergency')}")
print(f"  Urgent:    {sum(1 for _, u in rows if u == 'Urgent')}")
print(f"  Routine:   {sum(1 for _, u in rows if u == 'Routine')}")
print("\nCSV saved to: output/disease_urgency_mapping.csv")
