import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

from symptom_extractor import extract_symptoms, pretty_symptom_name
from followup_extractor import (
    blank_followup_features,
    extract_followup_features,
    merge_followup_features,
    extract_yes_no,
)

app = Flask(__name__)
app.secret_key = "triage-chatbot-secret-key"

MODELS_DIR = "models"

disease_model_path = os.path.join(MODELS_DIR, "best_disease_model.pkl")
triage_model_path = os.path.join(MODELS_DIR, "best_triage_model.pkl")
feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
disease_to_urgency_path = os.path.join(MODELS_DIR, "disease_to_urgency.pkl")

disease_model = joblib.load(disease_model_path)
triage_model = joblib.load(triage_model_path)
feature_columns = joblib.load(feature_cols_path)
disease_to_urgency = joblib.load(disease_to_urgency_path)

FOLLOWUP_COLUMNS = [
    "duration_short",
    "duration_medium",
    "duration_long",
    "severity_mild",
    "severity_moderate",
    "severity_severe",
    "onset_sudden",
    "onset_gradual",
    "progression_worse",
    "progression_same",
    "progression_improving",
]

SYMPTOM_COLUMNS = [col for col in feature_columns if col not in FOLLOWUP_COLUMNS]


def create_feature_vector(symptoms, followup_data):
    row = {feature: 0 for feature in feature_columns}

    for symptom in symptoms:
        if symptom in row:
            row[symptom] = 1

    for key, value in (followup_data or {}).items():
        if key in row:
            row[key] = int(value)

    return pd.DataFrame([row])


def clean_disease_name(name):
    return str(name).replace("_", " ").replace("-", " ").title()


def get_top_diseases(features, top_n=5):
    try:
        probabilities = disease_model.predict_proba(features)[0]
        classes = disease_model.classes_

        disease_probs = list(zip(classes, probabilities))
        disease_probs.sort(key=lambda x: x[1], reverse=True)

        strong_results = [(d, p) for d, p in disease_probs if p >= 0.03]
        if not strong_results:
            strong_results = disease_probs[:top_n]

        return strong_results[:top_n]

    except Exception:
        pred = disease_model.predict(features)[0]
        return [(pred, 1.0)]


def format_disease_list(top_diseases):
    return ", ".join([f"{clean_disease_name(d)} ({p:.1%})" for d, p in top_diseases])


def prediction_confidence_label(top_diseases):
    if not top_diseases:
        return "low"

    top_prob = top_diseases[0][1]

    if top_prob >= 0.60:
        return "high"
    elif top_prob >= 0.30:
        return "moderate"
    return "low"


def get_missing_followup_slot(followup_data):
    followup_data = followup_data or {}

    has_severity = any(followup_data.get(k, 0) == 1 for k in [
        "severity_mild", "severity_moderate", "severity_severe"
    ])
    has_duration = any(followup_data.get(k, 0) == 1 for k in [
        "duration_short", "duration_medium", "duration_long"
    ])
    has_onset = any(followup_data.get(k, 0) == 1 for k in [
        "onset_sudden", "onset_gradual"
    ])
    has_progression = any(followup_data.get(k, 0) == 1 for k in [
        "progression_worse", "progression_same", "progression_improving"
    ])

    if not has_severity:
        return "severity"
    if not has_duration:
        return "duration"
    if not has_onset:
        return "onset"
    if not has_progression:
        return "progression"

    return None


def get_follow_up_question(followup_data):
    slot = get_missing_followup_slot(followup_data)

    if slot == "severity":
        return "How severe are your symptoms: mild, moderate, or severe?", "severity"
    if slot == "duration":
        return "How long have you had these symptoms?", "duration"
    if slot == "onset":
        return "Did the symptoms start suddenly or gradually?", "onset"
    if slot == "progression":
        return "Are the symptoms getting worse, staying the same, or improving?", "progression"

    return "Thank you. I have enough follow-up detail to continue the triage assessment.", None


def format_followup_summary(followup_data):
    if not followup_data:
        return ""

    parts = []

    if followup_data.get("duration_short"):
        parts.append("short duration")
    elif followup_data.get("duration_medium"):
        parts.append("medium duration")
    elif followup_data.get("duration_long"):
        parts.append("long duration")

    if followup_data.get("severity_mild"):
        parts.append("mild severity")
    elif followup_data.get("severity_moderate"):
        parts.append("moderate severity")
    elif followup_data.get("severity_severe"):
        parts.append("severe symptoms")

    if followup_data.get("onset_sudden"):
        parts.append("sudden onset")
    elif followup_data.get("onset_gradual"):
        parts.append("gradual onset")

    if followup_data.get("progression_worse"):
        parts.append("worsening symptoms")
    elif followup_data.get("progression_same"):
        parts.append("same symptoms")
    elif followup_data.get("progression_improving"):
        parts.append("improving symptoms")

    if not parts:
        return ""

    return "Follow-up details noted: " + ", ".join(parts) + "."


def apply_yes_no_to_context(answer, last_question_type, followup_data):
    updated = dict(followup_data or {})

    if answer is None or last_question_type is None:
        return updated

    if last_question_type == "onset":
        if answer == "yes":
            updated["onset_sudden"] = 1
            updated["onset_gradual"] = 0
        elif answer == "no":
            updated["onset_sudden"] = 0
            updated["onset_gradual"] = 1

    return updated


def is_followup_only_message(user_message, has_existing_symptoms):
    if not has_existing_symptoms:
        return False

    text = user_message.lower().strip()
    yes_no = extract_yes_no(text)
    if yes_no is not None:
        return True

    extracted_followup = extract_followup_features(text)
    has_followup_signal = any(v == 1 for v in extracted_followup.values())

    if has_followup_signal and len(text.split()) <= 6:
        return True

    return False


def generate_chatbot_reply(symptoms, negated_symptoms, top_diseases, triage_pred, followup_data):
    if not symptoms:
        return (
            "I could not clearly identify your symptoms yet. "
            "Please describe them in more detail, for example: "
            "'I have fever, cough, and chest pain for 2 days.'"
        ), None

    symptom_text = ", ".join([pretty_symptom_name(s) for s in symptoms])
    reply_parts = [f"I have noted these symptoms so far: {symptom_text}."]

    if negated_symptoms:
        negated_text = ", ".join([pretty_symptom_name(s) for s in negated_symptoms])
        reply_parts.append(f"You also said you do not have: {negated_text}.")

    followup_summary = format_followup_summary(followup_data)
    if followup_summary:
        reply_parts.append(followup_summary)

    confidence = prediction_confidence_label(top_diseases)

    if top_diseases:
        disease_text = format_disease_list(top_diseases)
        if confidence == "high":
            reply_parts.append(f"The most likely conditions are: {disease_text}.")
        elif confidence == "moderate":
            reply_parts.append(f"Possible matching conditions include: {disease_text}.")
        else:
            reply_parts.append(
                f"At the moment the prediction confidence is low, but possible matches include: {disease_text}."
            )

    triage_display = str(triage_pred).title()
    reply_parts.append(f"Your current triage classification is: {triage_display}.")

    next_question, question_type = get_follow_up_question(followup_data)
    if question_type is not None:
        reply_parts.append(next_question)

   
    return " ".join(reply_parts), question_type


@app.route("/")
def home():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "collected_symptoms" not in session:
        session["collected_symptoms"] = []
    if "followup_data" not in session:
        session["followup_data"] = blank_followup_features()
    if "last_question_type" not in session:
        session["last_question_type"] = None

    return render_template("chat.html", chat_history=session["chat_history"])


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type your symptoms first."})

    collected_symptoms = session.get("collected_symptoms", [])
    followup_data = session.get("followup_data", blank_followup_features())
    last_question_type = session.get("last_question_type")

    followup_only = is_followup_only_message(
        user_message=user_message,
        has_existing_symptoms=bool(collected_symptoms)
    )

    if followup_only:
        extracted_symptoms = []
        negated_symptoms = []
    else:
        extraction_result = extract_symptoms(user_message)
        extracted_symptoms = extraction_result["symptoms"]
        negated_symptoms = extraction_result["negated_symptoms"]

    yes_no_answer = extract_yes_no(user_message)
    followup_data = apply_yes_no_to_context(yes_no_answer, last_question_type, followup_data)

    new_followup_data = extract_followup_features(user_message)

    collected_symptoms = [s for s in collected_symptoms if s not in negated_symptoms]
    collected_symptoms = list(set(collected_symptoms + extracted_symptoms))
    followup_data = merge_followup_features(followup_data, new_followup_data)

    session["collected_symptoms"] = collected_symptoms
    session["followup_data"] = followup_data

    if not collected_symptoms:
        bot_reply = (
            "I still could not identify enough symptoms from your message. "
            "Please describe your symptoms more clearly, for example: "
            "'I have fever, cough, sore throat, and headache for 3 days.'"
        )

        chat_history = session.get("chat_history", [])
        chat_history.append({"sender": "user", "text": user_message})
        chat_history.append({"sender": "bot", "text": bot_reply})
        session["chat_history"] = chat_history

        return jsonify({
            "reply": bot_reply,
            "symptoms": [],
            "negated_symptoms": [],
            "top_diseases": [],
            "triage": None,
            "followup_data": followup_data
        })

    features = create_feature_vector(collected_symptoms, followup_data)
    top_diseases = get_top_diseases(features, top_n=5)

    try:
        triage_pred = triage_model.predict(features)[0]
    except Exception:
        top_disease = top_diseases[0][0]
        if top_disease in disease_to_urgency.get("emergency", []):
            triage_pred = "emergency"
        elif top_disease in disease_to_urgency.get("urgent", []):
            triage_pred = "urgent"
        else:
            triage_pred = "routine"

    bot_reply, new_question_type = generate_chatbot_reply(
        collected_symptoms,
        negated_symptoms,
        top_diseases,
        triage_pred,
        followup_data
    )

    session["last_question_type"] = new_question_type

    chat_history = session.get("chat_history", [])
    chat_history.append({"sender": "user", "text": user_message})
    chat_history.append({"sender": "bot", "text": bot_reply})
    session["chat_history"] = chat_history

    return jsonify({
        "reply": bot_reply,
        "symptoms": [pretty_symptom_name(s) for s in collected_symptoms],
        "negated_symptoms": [pretty_symptom_name(s) for s in negated_symptoms],
        "top_diseases": [
            {"disease": clean_disease_name(d), "probability": float(p)}
            for d, p in top_diseases
        ],
        "triage": str(triage_pred).title(),
        "followup_data": followup_data
    })


@app.route("/reset", methods=["POST"])
def reset_chat():
    session["chat_history"] = []
    session["collected_symptoms"] = []
    session["followup_data"] = blank_followup_features()
    session["last_question_type"] = None
    return jsonify({"message": "Chat reset successful."})


if __name__ == "__main__":
    app.run(debug=True)