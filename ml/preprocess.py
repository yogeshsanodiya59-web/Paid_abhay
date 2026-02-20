"""
ml/preprocess.py
----------------
Helper preprocessing functions shared between training (train_model.py)
and inference (app.py). Keeping preprocessing identical in both places
is critical to avoid train-serve skew.
"""

import re


# ---------------------------------------------------------------------------
# Symptom text helpers
# ---------------------------------------------------------------------------

def clean_symptoms(text: str) -> str:
    """
    Normalise a raw symptom string.

    Steps
    -----
    1. Lowercase
    2. Remove non-alpha characters except commas
    3. Split on commas, strip each token, drop empty tokens
    4. Remove duplicate symptoms while preserving order
    5. Re-join with ', '

    Parameters
    ----------
    text : str
        Raw symptom string, e.g. "Fever, Cough, fever, headache"

    Returns
    -------
    str
        Cleaned string, e.g. "fever, cough, headache"
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove characters that are not letters, spaces, or commas
    text = re.sub(r"[^a-z ,]", "", text)

    # Split, strip, deduplicate
    tokens = [t.strip() for t in text.split(",")]
    tokens = [t for t in tokens if t]  # drop empty

    # Deduplicate while preserving order
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    return ", ".join(unique_tokens)


def symptoms_list_to_string(symptom_list: list) -> str:
    """
    Convert a Python list of symptom strings to a cleaned comma-separated string.

    Parameters
    ----------
    symptom_list : list[str]
        e.g. ["Fever", "Cough", "Headache"]

    Returns
    -------
    str
        e.g. "fever, cough, headache"
    """
    raw = ", ".join(symptom_list)
    return clean_symptoms(raw)


def build_feature_dict(age: int, gender: str, symptoms_str: str) -> dict:
    """
    Build a single-row feature dictionary suitable for passing to a
    pandas DataFrame and then to the sklearn pipeline.

    Parameters
    ----------
    age : int
    gender : str   e.g. "Male", "Female", "Other"
    symptoms_str : str  already-cleaned symptom string

    Returns
    -------
    dict
    """
    return {
        "Age": int(age),
        "Gender": gender.strip().capitalize(),
        "Symptoms": symptoms_str,
        "Symptom_Count": len([s for s in symptoms_str.split(",") if s.strip()]),
    }
