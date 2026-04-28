import re

FOLLOWUP_FEATURES = [
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

YES_WORDS = {"yes", "yeah", "yep", "i do", "i have", "correct"}
NO_WORDS = {"no", "nope", "nah", "i don't", "i do not", "not really"}

def blank_followup_features():
    return {feature: 0 for feature in FOLLOWUP_FEATURES}

def _set_duration_features(days):
    features = {
        "duration_short": 0,
        "duration_medium": 0,
        "duration_long": 0,
    }

    if days is None:
        return features

    if days <= 3:
        features["duration_short"] = 1
    elif days <= 10:
        features["duration_medium"] = 1
    else:
        features["duration_long"] = 1

    return features

def extract_yes_no(text):
    text = (text or "").lower().strip()

    if text in YES_WORDS:
        return "yes"
    if text in NO_WORDS:
        return "no"

    return None

def extract_followup_features(text):
    text = (text or "").lower().strip()
    features = blank_followup_features()

    duration_days = None

    day_match = re.search(r"(\d+)\s*(day|days|hour|hours)", text)
    week_match = re.search(r"(\d+)\s*(week|weeks)", text)
    month_match = re.search(r"(\d+)\s*(month|months)", text)

    if day_match:
        value = int(day_match.group(1))
        unit = day_match.group(2)
        if "hour" in unit:
            duration_days = 1
        else:
            duration_days = value
    elif week_match:
        duration_days = int(week_match.group(1)) * 7
    elif month_match:
        duration_days = int(month_match.group(1)) * 30

    # ✅ FIX: catch "a month", "about a month", "month", "months", "a few months"
    elif re.search(r"\b(a month|about a month|one month|a few months|few months|months|month)\b", text):
        duration_days = 30

    # ✅ FIX: catch "a week", "about a week", "week"
    elif re.search(r"\b(a week|about a week|one week|a few weeks|few weeks)\b", text):
        duration_days = 7

    # ✅ FIX: catch "two weeks", "a couple of weeks"
    elif re.search(r"\b(two weeks|couple of weeks|2 weeks)\b", text):
        duration_days = 14

    elif "today" in text or "since morning" in text or "since this morning" in text or "since yesterday" in text or "since last night" in text or "since last evening" in text:
        duration_days = 1
    elif "few days" in text or "a few days" in text:
        duration_days = 3
    elif "long time" in text or "for a while" in text or "ages" in text or "a long time" in text:
        duration_days = 30

    features.update(_set_duration_features(duration_days))

    # Severity
    if any(word in text for word in ["mild", "slight", "a little", "not too bad"]):
        features["severity_mild"] = 1
    elif any(word in text for word in ["moderate", "medium", "fairly bad", "quite bad"]):
        features["severity_moderate"] = 1
    elif any(word in text for word in ["severe", "very severe", "extreme", "terrible", "really bad", "intense", "worst", "horrible", "unbearable", "can't stand"]):
        features["severity_severe"] = 1

    # Onset
    if any(word in text for word in ["sudden", "suddenly", "all of a sudden", "started suddenly"]):
        features["onset_sudden"] = 1
    elif any(word in text for word in ["gradual", "gradually", "started slowly", "came on slowly"]):
        features["onset_gradual"] = 1

    # Progression
    if any(word in text for word in ["getting worse", "worsening", "worse", "more severe", "increasing"]):
        features["progression_worse"] = 1
    elif any(word in text for word in ["staying the same", "same", "unchanged", "no change"]):
        features["progression_same"] = 1
    elif any(word in text for word in ["improving", "better", "getting better"]):
        features["progression_improving"] = 1

    return features

def merge_followup_features(existing, new_features):
    merged = blank_followup_features()

    if existing:
        for key in merged:
            merged[key] = int(existing.get(key, 0))

    if new_features:
        for key in merged:
            if int(new_features.get(key, 0)) == 1:
                merged[key] = 1

    if merged["severity_severe"] == 1:
        merged["severity_mild"] = 0
        merged["severity_moderate"] = 0
    elif merged["severity_moderate"] == 1:
        merged["severity_mild"] = 0

    if merged["duration_long"] == 1:
        merged["duration_short"] = 0
        merged["duration_medium"] = 0
    elif merged["duration_medium"] == 1:
        merged["duration_short"] = 0

    if merged["onset_sudden"] == 1:
        merged["onset_gradual"] = 0
    elif merged["onset_gradual"] == 1:
        merged["onset_sudden"] = 0

    if merged["progression_worse"] == 1:
        merged["progression_same"] = 0
        merged["progression_improving"] = 0
    elif merged["progression_same"] == 1:
        merged["progression_improving"] = 0

    return merged