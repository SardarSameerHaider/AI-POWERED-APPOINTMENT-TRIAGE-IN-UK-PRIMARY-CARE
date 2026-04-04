import os
import pandas as pd

DATA_DIR = "data"
INPUT_PATH = os.path.join(DATA_DIR, "Training.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "Training_enriched.csv")

FOLLOWUP_COLUMNS = [
    "duration_short",
    "duration_medium",
    "duration_long",
    "severity_mild",
    "severity_moderate",
    "severity_severe",
    "onset_sudden",
    "getting_worse",
]

EMERGENCY_DISEASES = {
    "Heart attack",
    "Paralysis brain hemorrhage",
    "Hepatitis E",
    "Dimorphic hemmorhoidspiles",
    "Peptic ulcer diseae",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Varicose veins",
}

URGENT_DISEASES = {
    "Pneumonia",
    "Bronchial Asthma",
    "Malaria",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Chronic cholestasis",
    "Gastroenteritis",
    "Hypertension",
    "Migraine",
    "Diabetes",
    "Urinary tract infection",
}

def blank_followup():
    return {col: 0 for col in FOLLOWUP_COLUMNS}

def build_variants(prognosis):
    if prognosis in EMERGENCY_DISEASES:
        return [
            {
                "duration_short": 1,
                "severity_severe": 1,
                "onset_sudden": 1,
                "getting_worse": 1,
            },
            {
                "duration_medium": 1,
                "severity_severe": 1,
                "onset_sudden": 0,
                "getting_worse": 1,
            },
        ]

    if prognosis in URGENT_DISEASES:
        return [
            {
                "duration_short": 1,
                "severity_moderate": 1,
                "onset_sudden": 0,
                "getting_worse": 1,
            },
            {
                "duration_medium": 1,
                "severity_severe": 1,
                "onset_sudden": 0,
                "getting_worse": 1,
            },
            {
                "duration_medium": 1,
                "severity_moderate": 1,
                "onset_sudden": 0,
                "getting_worse": 0,
            },
        ]

    return [
        {
            "duration_short": 1,
            "severity_mild": 1,
            "onset_sudden": 0,
            "getting_worse": 0,
        },
        {
            "duration_medium": 1,
            "severity_moderate": 1,
            "onset_sudden": 0,
            "getting_worse": 0,
        },
    ]

def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(axis=1, how="all")

    if df.columns[-1].lower() != "prognosis":
        df = df.rename(columns={df.columns[-1]: "prognosis"})

    enriched_rows = []

    for _, row in df.iterrows():
        prognosis = row["prognosis"]
        variants = build_variants(prognosis)

        for variant in variants:
            row_dict = row.to_dict()
            row_dict.update(blank_followup())
            row_dict.update(variant)
            enriched_rows.append(row_dict)

    enriched_df = pd.DataFrame(enriched_rows)

    ordered_columns = [c for c in enriched_df.columns if c != "prognosis"] + ["prognosis"]
    enriched_df = enriched_df[ordered_columns]

    enriched_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved enriched dataset to:", OUTPUT_PATH)
    print("Original shape:", df.shape)
    print("Enriched shape:", enriched_df.shape)

if __name__ == "__main__":
    main()