# 🏥 AI-Powered Appointment Triage in UK Primary Care

A conversational AI chatbot that triages patient symptoms and recommends appointment urgency levels (Emergency, Urgent, or Routine) — designed in the context of UK primary care (NHS GP surgeries).

---

## 📌 Overview

This project uses machine learning and natural language processing to assist patients in describing their symptoms through a chat interface. The system extracts symptoms, asks intelligent follow-up questions, predicts likely conditions, and classifies the appointment urgency — reducing strain on GP receptionists and supporting faster triage decisions.

The chatbot collects symptoms conversationally, asks follow-up questions about severity, duration, onset, and progression, then presents the top predicted conditions and a triage urgency level on a result page.

---

## 🚀 Features

- 💬 Conversational chatbot interface built with Flask
- 🔍 Semantic symptom extraction using `sentence-transformers` (all-MiniLM-L6-v2)
- 🧠 Multi-class disease prediction with probability scores
- 🚨 Triage classification: **Emergency**, **Urgent**, or **Routine**
- 📋 Intelligent follow-up questions for severity, duration, onset, and progression
- 🔄 Negation handling (e.g. "I have fever but no cough")
- 📊 Result page with top predicted conditions and urgency level
- 🧪 Functional test suite included

---

## 🗂️ Project Structure

```
ai-triage/
├── app.py                       # Flask web application & chatbot logic
├── symptom_extractor.py         # NLP symptom extraction (semantic embeddings)
├── followup_extractor.py        # Follow-up question feature extraction
├── data_and_model.py            # ML model training & evaluation
├── enrich_training_data.py      # Training data enrichment
├── make_feature_vector_demo.py  # Feature vector construction demo
├── make_mapping_table.py        # Symptom-to-disease mapping
├── run_function_tests.py        # Functional test suite
├── requirements.txt             # Python dependencies
├── data/
│   ├── Training.csv             # Original training dataset
│   ├── Training_enriched.csv    # Enriched training dataset
│   └── Testing.csv              # Test dataset
├── models/                      # Saved ML models (generated after training)
│   ├── best_disease_model.pkl
│   ├── best_triage_model.pkl
│   ├── feature_columns.pkl
│   ├── symptom_columns.pkl
│   ├── disease_to_urgency.pkl
│   └── model_results.csv
├── static/
│   └── style.css                # Global stylesheet
└── templates/
    ├── index.html               # Landing page
    ├── chat.html                # Chatbot interface
    └── result.html              # Triage result page
```

---

## ✅ Prerequisites

Before running this project, make sure you have the following installed:

- **Python 3.10 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** (comes bundled with Python)
- **Git** — [Download here](https://git-scm.com/)
- At least **4GB of free RAM** (sentence-transformers loads a neural model into memory)
- A stable internet connection for the **first run** (downloads the embedding model automatically)

---

## ⚙️ Full Setup & Installation Guide

### Step 1 — Clone the repository

```bash
git clone https://github.com/SardarSameerHaider/AI-POWERED-APPOINTMENT-TRIAGE-IN-UK-PRIMARY-CARE.git
cd AI-POWERED-APPOINTMENT-TRIAGE-IN-UK-PRIMARY-CARE
```

---

### Step 2 — Create a virtual environment

A virtual environment keeps all dependencies isolated from your global Python installation.

**Windows (CMD / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ You should see `(venv)` at the start of your terminal prompt once activated.

---

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This may take several minutes as it installs PyTorch, sentence-transformers, scikit-learn, and other packages.

If you encounter a version conflict error, run:
```bash
pip install --upgrade scipy scikit-learn sentence-transformers
```

---

### Step 4 — Train the ML models

This generates all `.pkl` model files inside the `models/` folder.
**You only need to run this once.**

```bash
python data_and_model.py
```

> ✅ After completion, the `models/` folder will contain:
> `best_disease_model.pkl`, `best_triage_model.pkl`, `feature_columns.pkl`,
> `symptom_columns.pkl`, `disease_to_urgency.pkl`, `model_results.csv`

---

### Step 5 — Start the Flask application

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

To stop the server press **Ctrl + C** in the terminal.

---

## 💬 How to Use the Chatbot

1. Open `http://127.0.0.1:5000` in your browser
2. You will see a welcome message from the AI Triage Assistant
3. Type your symptoms naturally in the message box, for example:
   ```
   I have fever, cough, chest pain, and headache for 3 days
   ```
4. The chatbot will:
   - Confirm the symptoms it has detected
   - Ask follow-up questions one at a time:
     - **Severity** — mild, moderate, or severe?
     - **Duration** — how long have you had the symptoms?
     - **Onset** — did they start suddenly or gradually?
     - **Progression** — are they getting worse, staying the same, or improving?
   - Show the top predicted conditions with probability scores
   - Display your triage urgency classification (Emergency / Urgent / Routine)
5. After all follow-up questions are answered, you are automatically redirected to the **Result Page**
6. Click **Reset** at any time to clear the chat and start a new session

---

## 🧪 Running the Test Suite

```bash
python run_function_tests.py
```

Tests core components including symptom extraction, follow-up feature extraction, feature vector construction, and disease prediction.

---

## 🔁 Re-training the Models

If you modify the training data or enrichment logic, re-run the pipeline in order:

```bash
# Step 1 — Re-enrich the training data
python enrich_training_data.py

# Step 2 — Re-train and save updated model files
python data_and_model.py
```

---

## 🤖 ML Models & Performance

Four classifiers are trained and evaluated automatically. The best-performing model is selected and saved.

| Model | Accuracy |
|---|---|
| Logistic Regression | 100% |
| Decision Tree | 100% |
| Random Forest | 100% |
| Naive Bayes | 98.73% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| ML Models | scikit-learn |
| NLP / Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Deep Learning | PyTorch, Hugging Face transformers |
| Data Processing | pandas, numpy, scipy |
| Visualisation | matplotlib, seaborn |
| Frontend | HTML, CSS, JavaScript |

---

## 🐛 Common Issues & Fixes

| Problem | Solution |
|---|---|
| `No such file: models/best_disease_model.pkl` | Run `python data_and_model.py` first |
| `ModuleNotFoundError: No module named 'flask'` | Activate venv and run `pip install -r requirements.txt` |
| `scipy` / `sklearn` import crash on startup | Run `pip install --upgrade scipy scikit-learn sentence-transformers` |
| Symptoms not being detected | Describe clearly, e.g. `"I have fever, cough, and chest pain"` |
| Duration answers not recognised | Use phrases like `"2 days"`, `"a week"`, `"about a month"` |
| App crashes immediately | Check Python version is 3.10+ using `python --version` |
| Port 5000 already in use | Change port in app.py: `app.run(port=5001)` |

---

## ⚠️ Disclaimer

This application is a **prototype developed for academic purposes only**. It is not a certified medical device and must not be used for real clinical decision-making. Always consult a qualified healthcare professional for medical advice.

---

## 👤 Author

**Sardar Sameer Haider**
BSc Computer Science — UWE Bristol
[GitHub Profile](https://github.com/SardarSameerHaider)
