import pandas as pd
import os
import joblib
import sys

# Add your project folder to path if needed
sys.path.insert(0, ".")

# Import your extractors
try:
    from symptom_extractor import extract_symptoms
    from followup_extractor import extract_followup_features, merge_followup_features
except ImportError as e:
    print(f"Error importing extractors: {e}")
    print("Make sure symptom_extractor.py and followup_extractor.py exist in this folder.")
    sys.exit(1)

MODELS_DIR = "models"
feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

# Load feature columns
try:
    feature_columns = joblib.load(feature_cols_path)
except FileNotFoundError:
    print(f"Error: Could not find {feature_cols_path}")
    print("Run data_and_model.py first to generate the model artefacts.")
    sys.exit(1)

# ---------------------------------------------------------------------
# Define test scenarios (these should match your T7-T9 test cases)
# ---------------------------------------------------------------------

test_scenarios = [
    {
        "name": "Headache scenario (T7)",
        "messages": [
            "I have a bad headache for 2 days and it's getting worse"
        ],
    },
    {
        "name": "Chest pain scenario (T8)",
        "messages": [
            "I have sudden severe chest pain and I can't breathe properly"
        ],
    },
]

# ---------------------------------------------------------------------
# Helper: create feature vector from extracted data
# ---------------------------------------------------------------------

def create_feature_vector(collected_symptoms, followup_data, feature_columns):
    """
    Create a single-row DataFrame matching the training schema.
    """
    row = {col: 0 for col in feature_columns}

    # Set symptom features
    for symptom in collected_symptoms:
        # Handle case-insensitive matching
        symptom_lower = symptom.lower().replace(" ", "_")
        for col in feature_columns:
            if symptom_lower in col.lower() or col.lower() in symptom_lower:
                row[col] = 1
                break

    # Set follow-up features
    for key, value in followup_data.items():
        if key in row:
            row[key] = value

    return pd.DataFrame([row])

# ---------------------------------------------------------------------
# Process each scenario and output feature vectors
# ---------------------------------------------------------------------

os.makedirs("output", exist_ok=True)

print("\nAppendix D – Extracted Feature Vectors for Test Cases")
print("=" * 80)

for scenario in test_scenarios:
    print(f"\n{scenario['name']}")
    print("-" * 80)

    collected_symptoms = []
    followup_data = {}

    for msg in scenario["messages"]:
        print(f"Input: \"{msg}\"")

        # Extract symptoms
        symptoms, negated = extract_symptoms(msg)
        collected_symptoms.extend(symptoms)
        print(f"  Detected symptoms: {symptoms}")
        if negated:
            print(f"  Negated symptoms: {negated}")

        # Extract follow-up
        followup = extract_followup_features(msg)
        followup_data = merge_followup_features(followup_data, followup)
        if followup:
            print(f"  Follow-up features: {followup}")

    # Build feature vector
    feature_df = create_feature_vector(
        list(set(collected_symptoms)),  # deduplicate
        followup_data,
        feature_columns
    )

    # Show non-zero features only
    non_zero_cols = feature_df.columns[feature_df.iloc[0] != 0]
    print(f"\n  Non-zero features in vector ({len(non_zero_cols)}):")
    for col in non_zero_cols:
        print(f"    {col} = {feature_df[col].iloc[0]}")

    # Save full row to CSV
    out_csv = f"output/feature_vector_{scenario['name'].replace(' ', '_').lower()}.csv"
    feature_df.to_csv(out_csv, index=False)
    print(f"\n  Full vector saved to: {out_csv}")

print("\n" + "=" * 80)
print("All feature vector CSVs saved to: output/")
