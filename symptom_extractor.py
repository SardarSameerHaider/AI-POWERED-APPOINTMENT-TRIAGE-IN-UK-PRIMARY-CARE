import os
import re
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODELS_DIR = "models"
SYMPTOM_COLUMNS_PATH = os.path.join(MODELS_DIR, "symptom_columns.pkl")

NEGATION_WORDS = {"no", "not", "dont", "don't", "without", "never", "none"}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.58
TOP_K = 1
MAX_SYMPTOMS_PER_MESSAGE = 3

FOLLOWUP_ONLY_TERMS = {
    "mild", "moderate", "severe",
    "worse", "better", "same",
    "today", "yesterday",
}

try:
    REAL_SYMPTOM_COLUMNS = joblib.load(SYMPTOM_COLUMNS_PATH)
except Exception:
    REAL_SYMPTOM_COLUMNS = []

def clean_text(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def symptom_to_phrase(symptom_name):
    return symptom_name.lower().replace("_", " ").replace("-", " ").strip()

def build_symptom_label_texts(symptom_columns):
    return {symptom: symptom_to_phrase(symptom) for symptom in symptom_columns}

SYMPTOM_TEXT_MAP = build_symptom_label_texts(REAL_SYMPTOM_COLUMNS)
SYMPTOM_LABELS = list(SYMPTOM_TEXT_MAP.keys())
SYMPTOM_TEXTS = [SYMPTOM_TEXT_MAP[label] for label in SYMPTOM_LABELS]

MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

if SYMPTOM_TEXTS:
    SYMPTOM_EMBEDDINGS = MODEL.encode(SYMPTOM_TEXTS, normalize_embeddings=True)
else:
    SYMPTOM_EMBEDDINGS = np.array([])

def cosine_scores(query_embedding, matrix_embeddings):
    return np.dot(matrix_embeddings, query_embedding)

def contains_duration_pattern(text):
    return bool(re.search(r"\b\d+\s*(day|days|hour|hours|week|weeks|month|months)\b", text))

def is_followup_like_text(user_text):
    text = clean_text(user_text)

    if contains_duration_pattern(text):
        return True

    if text in FOLLOWUP_ONLY_TERMS:
        return True

    if text in {
        "getting worse", "staying the same", "getting better",
        "feeling worse", "feeling better", "feeling the same",
        "progressing worse", "progressing better", "progressing the same"
    }:
        return True

    tokens = text.split()
    if len(tokens) <= 3:
        has_followup_token = any(
            word in {"mild", "moderate", "severe", "worse", "better", "same", "today", "yesterday"}
            for word in tokens
        )
        has_symptom_hint = any(
            word in {"cough", "fever", "pain", "headache", "dizzy", "dizziness", "nausea"}
            for word in tokens
        )
        if has_followup_token and not has_symptom_hint:
            return True

    return False

def split_into_candidate_segments(text):
    text = clean_text(text)
    text = text.replace(" but not ", " ||but_not|| ")
    text = text.replace(" but no ", " ||but_no|| ")

    raw_parts = re.split(r",| and | also | but |;|\|\|but_not\|\||\|\|but_no\|\|", text)

    segments = []
    for part in raw_parts:
        part = part.strip()
        if len(part) >= 3:
            segments.append(part)

    if text and text not in segments:
        segments.append(text)

    return list(dict.fromkeys(segments))

def extract_direct_negated_symptoms(text):
    text = clean_text(text)
    negated = set()

    patterns = [
        r"\bno\s+([a-z\s]+?)(?=,| and | but |;|$)",
        r"\bnot\s+(?:a\s+|an\s+|any\s+)?([a-z\s]+?)(?=,| and | but |;|$)",
        r"\bwithout\s+([a-z\s]+?)(?=,| and | but |;|$)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            negated_phrase = match.group(1).strip()

            if not negated_phrase:
                continue

            neg_embedding = MODEL.encode([negated_phrase], normalize_embeddings=True)[0]
            scores = cosine_scores(neg_embedding, SYMPTOM_EMBEDDINGS)

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= SIMILARITY_THRESHOLD:
                negated.add(SYMPTOM_LABELS[best_idx])

    return negated

def semantic_match_segment(segment, top_k=TOP_K):
    if len(SYMPTOM_LABELS) == 0:
        return []

    segment_embedding = MODEL.encode([segment], normalize_embeddings=True)[0]
    scores = cosine_scores(segment_embedding, SYMPTOM_EMBEDDINGS)

    ranked_idx = np.argsort(scores)[::-1][:top_k]
    matches = []

    token_count = len(segment.split())
    threshold = SIMILARITY_THRESHOLD
    if token_count <= 3:
        threshold = SIMILARITY_THRESHOLD - 0.03

    for idx in ranked_idx:
        score = float(scores[idx])
        if score >= threshold:
            matches.append((SYMPTOM_LABELS[idx], score))

    return matches

def extract_symptoms(user_text):
    text = clean_text(user_text)

    if not text:
        return {"symptoms": [], "negated_symptoms": []}

    if is_followup_like_text(text):
        return {"symptoms": [], "negated_symptoms": []}

    negated_symptoms = extract_direct_negated_symptoms(text)
    segments = split_into_candidate_segments(text)

    found_symptoms = {}

    for segment in segments:
        if is_followup_like_text(segment):
            continue

        if segment.startswith("no ") or segment.startswith("not ") or segment.startswith("without "):
            continue

        matches = semantic_match_segment(segment, top_k=TOP_K)

        for symptom, score in matches:
            if symptom in negated_symptoms:
                continue

            if symptom not in found_symptoms or score > found_symptoms[symptom]:
                found_symptoms[symptom] = score

    final_symptoms = [
        s for s, _ in sorted(found_symptoms.items(), key=lambda x: x[1], reverse=True)
        if s not in negated_symptoms
    ]
    final_symptoms = final_symptoms[:MAX_SYMPTOMS_PER_MESSAGE]

    return {
        "symptoms": final_symptoms,
        "negated_symptoms": sorted(negated_symptoms),
    }

def pretty_symptom_name(symptom):
    return symptom.replace("_", " ").replace("-", " ").title()