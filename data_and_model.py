import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.table import Table


DATA_DIR = "data"
TRAINING_PATH = os.path.join(DATA_DIR, "Training_enriched.csv")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

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


def load_training_binary(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")

    if df.columns[-1].lower() != "prognosis":
        df = df.rename(columns={df.columns[-1]: "prognosis"})

    return df


# ---------------------------------------------------------------------
# Helper function: render a classification report as a PNG table
# ---------------------------------------------------------------------

def save_classification_report_png(y_true, y_pred, model_name, output_dir):
    """
    Save a classification report (precision, recall, F1) as a PNG table image.
    """
    labels = sorted(list(set(y_true)))
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    rows = []
    for label in labels:
        p = report_dict[label]["precision"]
        r = report_dict[label]["recall"]
        f = report_dict[label]["f1-score"]
        s = report_dict[label]["support"]
        rows.append([label, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", str(int(s))])

    # Add macro and weighted avg rows
    rows.append(["Macro avg",
                 f"{report_dict['macro avg']['precision']:.3f}",
                 f"{report_dict['macro avg']['recall']:.3f}",
                 f"{report_dict['macro avg']['f1-score']:.3f}",
                 "-"])
    rows.append(["Weighted avg",
                 f"{report_dict['weighted avg']['precision']:.3f}",
                 f"{report_dict['weighted avg']['recall']:.3f}",
                 f"{report_dict['weighted avg']['f1-score']:.3f}",
                 "-"])

    # Build figure with table
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_axis_off()

    table = Table(ax, bbox=[0, 0, 1, 1])
    col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    n_rows = len(rows) + 1
    n_cols = len(col_labels)
    col_w = 1.0 / n_cols
    row_h = 1.0 / n_rows

    # Header row
    for j, label in enumerate(col_labels):
        table.add_cell(
            0, j, col_w, row_h,
            text=label,
            loc="center",
            facecolor="#4472c4"
        )

    # Data rows
    for i, row in enumerate(rows, start=1):
        face = "#dddddd" if i % 2 == 0 else "white"
        for j, val in enumerate(row):
            table.add_cell(i, j, col_w, row_h, text=str(val), loc="center", facecolor=face)

    ax.add_table(table)

    header_cells = [cell for cell in table.get_celld().values() if cell.get_visible()]
    for cell in header_cells[:len(col_labels)]:
        cell.set_text_props(weight="bold", color="white")

    out_path = os.path.join(output_dir, f"classification_report_{model_name.replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path



# ---------------------------------------------------------------------
# Helper function: save model accuracy bar chart as PNG
# ---------------------------------------------------------------------

def save_accuracy_bar_chart(results, output_dir):
    """
    Save a bar chart showing accuracy for each classifier.
    """
    models = list(results.keys())
    accs = list(results.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, accs, color="#4472c4")

    # Annotate bars with values
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison on Urgency Prediction")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "model_accuracy_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main data loading and preprocessing
# ---------------------------------------------------------------------

print("Loading enriched training data...")
df_all = load_training_binary(TRAINING_PATH)

# Ensure follow-up columns exist
for col in FOLLOWUP_COLUMNS:
    if col not in df_all.columns:
        df_all[col] = 0

all_diseases = sorted(df_all["prognosis"].unique())

# Disease → urgency groupings
emergency_diseases = [
    "Heart attack",
    "Paralysis brain hemorrhage",
    "Hepatitis E",
    "Dimorphic hemmorhoidspiles",
    "Peptic ulcer diseae",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Varicose veins",
]

urgent_diseases = [
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
]


def map_urgency(disease: str) -> str:
    if disease in emergency_diseases:
        return "emergency"
    if disease in urgent_diseases:
        return "urgent"
    return "routine"


df_all["urgency"] = df_all["prognosis"].apply(map_urgency)

feature_cols = [c for c in df_all.columns if c not in ["prognosis", "urgency"]]
X = df_all[feature_cols]
y = df_all["urgency"]
y_disease = df_all["prognosis"]

feature_columns = feature_cols

print(f"Dataset shape: {df_all.shape}")
print(f"Feature count: {len(feature_columns)}")
print(f"Unique diseases: {df_all['prognosis'].nunique()}")

# Split for urgency model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(),
}

best_model = None
best_name = None
best_acc = 0.0
results = {}
all_predictions = {}

# Train and evaluate urgency models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    all_predictions[name] = y_pred

    print(f"{name} accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\nBest urgency model: {best_name} ({best_acc:.3f})")

# Disease model (trained on full data)
disease_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=20,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

print("\nTraining disease model...")
disease_model.fit(X, y_disease)

# Paths for artefacts
best_model_path = os.path.join(MODELS_DIR, "best_triage_model.pkl")
disease_model_path = os.path.join(MODELS_DIR, "best_disease_model.pkl")
feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
urgency_map_path = os.path.join(MODELS_DIR, "disease_to_urgency.pkl")
results_path = os.path.join(MODELS_DIR, "model_results.csv")

# Save models and metadata
joblib.dump(best_model, best_model_path)
joblib.dump(disease_model, disease_model_path)
joblib.dump(feature_columns, feature_cols_path)
joblib.dump(
    {
        "emergency": emergency_diseases,
        "urgent": urgent_diseases,
        "all_diseases": all_diseases,
    },
    urgency_map_path,
)

pd.Series(results, name="accuracy").to_csv(results_path)

# Confusion matrix for best urgency model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=["emergency", "urgent", "routine"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["emergency", "urgent", "routine"],
    yticklabels=["emergency", "urgent", "routine"],
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# Dataset summary table figure for Table 1
# ---------------------------------------------------------------------

n_cases = len(df_all)
symptom_cols = [c for c in df_all.columns if c not in ["prognosis", "urgency"]]
n_symptoms = len(symptom_cols)
n_diseases = df_all["prognosis"].nunique()
urgency_counts = df_all["urgency"].value_counts().to_dict()

summary_rows = [
    ["Total cases", n_cases],
    ["Symptom features", n_symptoms],
    ["Distinct diseases", n_diseases],
    [
        "Urgency distribution",
        ", ".join(f"{k}: {v}" for k, v in urgency_counts.items()),
    ],
]

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.set_axis_off()

table = Table(ax, bbox=[0, 0, 1, 1])
col_labels = ["Metric", "Value"]
n_rows = len(summary_rows) + 1
n_cols = len(col_labels)
col_w = 1.0 / n_cols
row_h = 1.0 / n_rows

# Header row
for j, label in enumerate(col_labels):
    table.add_cell(
        0, j, col_w, row_h,
        text=label,
        loc="center",
        facecolor="#dddddd"
    )

# Data rows
for i, (metric, value) in enumerate(summary_rows, start=1):
    table.add_cell(i, 0, col_w, row_h, text=str(metric), loc="left")
    table.add_cell(i, 1, col_w, row_h, text=str(value), loc="left")

ax.add_table(table)

os.makedirs("output", exist_ok=True)
summary_path = os.path.join("output", "dataset_summary_table.png")
fig.savefig(summary_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Generate classification report PNGs for ALL models (Appendix B)
# ---------------------------------------------------------------------

report_paths = {}
for name, y_pred in all_predictions.items():
    path = save_classification_report_png(y_test, y_pred, name, MODELS_DIR)
    report_paths[name] = path
    print(f"Classification report PNG for {name}: {path}")

# ---------------------------------------------------------------------
# Generate model accuracy bar chart (Appendix E / Design section)
# ---------------------------------------------------------------------

bar_chart_path = save_accuracy_bar_chart(results, MODELS_DIR)
print(f"Model accuracy bar chart: {bar_chart_path}")

# ---------------------------------------------------------------------

print("\nSaved:")
print(" - Best urgency model:", best_model_path)
print(" - Disease model:", disease_model_path)
print(" - Feature columns:", feature_cols_path)
print(" - Disease map:", urgency_map_path)
print(" - Results CSV:", results_path)
print(" - Confusion matrix PNG:", cm_path)
print(" - Dataset summary table PNG:", summary_path)
print(" - Model accuracy bar chart PNG:", bar_chart_path)
print(" - Classification report PNGs:")
for name, path in report_paths.items():
    print(f"     {name}: {path}")
