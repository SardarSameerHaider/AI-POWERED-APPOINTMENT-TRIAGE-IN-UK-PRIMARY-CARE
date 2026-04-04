import pprint

from symptom_extractor import extract_symptoms, pretty_symptom_name
from followup_extractor import (
    extract_followup_features,
    merge_followup_features,
    blank_followup_features,
)
from app import create_feature_vector 

pp = pprint.PrettyPrinter(indent=2)

def test_symptom_extraction():
    print("\n=== T1/T2: Symptom extraction tests ===")

    cases = [
        "I have fever and cough but no chest pain.",
        "I feel dizzy and have blurred vision but not a headache.",
    ]

    for text in cases:
        result = extract_symptoms(text)
        symptoms = result["symptoms"]
        negated = result["negated_symptoms"]
        print(f"\nInput: {text!r}")
        print("  Symptoms:", [pretty_symptom_name(s) for s in symptoms])
        print("  Negated :", [pretty_symptom_name(s) for s in negated])

def test_followup_extraction():
    print("\n=== T3/T4: Follow-up extraction tests ===")

    cases = [
        "It has been 3 days and the pain is severe.",
        "For two weeks it's been getting worse.",
        "Since this morning, it's mild but still there.",
    ]

    for text in cases:
        feats = extract_followup_features(text)
        print(f"\nInput: {text!r}")
        # Only show active features
        active = {k: v for k, v in feats.items() if v == 1}
        print("  Active follow-up features:")
        pp.pprint(active)

def test_merge_followup():
    print("\n=== Merge follow-up features (conflict resolution) ===")

    base = blank_followup_features()
    base["duration_short"] = 1
    base["severity_moderate"] = 1

    new_info = extract_followup_features("It has been 10 days and is now severe.")

    merged = merge_followup_features(base, new_info)
    print("Base active:", {k: v for k, v in base.items() if v == 1})
    print("New active :", {k: v for k, v in new_info.items() if v == 1})
    print("Merged     :", {k: v for k, v in merged.items() if v == 1})

def test_feature_vector():
    print("\n=== T5: Feature vector construction ===")

    collected_symptoms = ["fever", "cough"]
    followup_data = blank_followup_features()
    followup_data["duration_short"] = 1

    df_row = create_feature_vector(collected_symptoms, followup_data)
    print("Feature columns present:", len(df_row.columns))
    print("Non-zero features in the test row:")
    non_zero = {c: float(df_row.iloc[0][c]) for c in df_row.columns if df_row.iloc[0][c] != 0}
    pp.pprint(non_zero)

if __name__ == "__main__":
    test_symptom_extraction()
    test_followup_extraction()
    test_merge_followup()
    test_feature_vector()
    print("\nAll manual function tests executed. Inspect outputs above for correctness.")