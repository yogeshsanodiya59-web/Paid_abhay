"""
ml/train_model.py
-----------------
Trains a RandomForest disease-prediction pipeline on a synthetic dataset
and saves the trained artifacts to the ml/ folder.

Run:
    python ml/train_model.py

Outputs (inside ml/):
    model.pkl       – full sklearn Pipeline (preprocessing + classifier)
    vectorizer.pkl  – TF-IDF vectorizer extracted from the pipeline (reference)
    encoders.pkl    – dict with LabelEncoder for the target Disease column
    dataset.csv     – synthetic training dataset (generated if not present)
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ML_DIR, "dataset.csv")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ML_DIR, "vectorizer.pkl")
ENCODERS_PATH = os.path.join(ML_DIR, "encoders.pkl")

# ---------------------------------------------------------------------------
# Synthetic dataset definition
# ---------------------------------------------------------------------------

DISEASES = [
    "Common Cold", "Influenza", "Pneumonia", "Tuberculosis", "Asthma",
    "Bronchitis", "COVID-19", "Dengue Fever", "Malaria", "Typhoid Fever",
    "Diabetes Mellitus", "Hypertension", "Heart Disease", "Stroke",
    "Kidney Disease", "Liver Disease", "Anemia", "Arthritis",
    "Migraine", "Epilepsy", "Depression", "Anxiety Disorder",
    "Hypothyroidism", "Hyperthyroidism", "Gastroenteritis",
    "Appendicitis", "Urinary Tract Infection", "Skin Allergy",
    "Conjunctivitis", "Chickenpox",
]

# Symptom pool mapped to diseases (weighted association)
DISEASE_SYMPTOMS = {
    "Common Cold":          ["runny nose", "sneezing", "sore throat", "mild fever", "cough", "congestion", "fatigue"],
    "Influenza":            ["high fever", "body aches", "chills", "fatigue", "cough", "headache", "sore throat"],
    "Pneumonia":            ["high fever", "chest pain", "cough", "shortness of breath", "fatigue", "chills", "sweating"],
    "Tuberculosis":         ["persistent cough", "weight loss", "night sweats", "fatigue", "fever", "chest pain", "blood in cough"],
    "Asthma":               ["wheezing", "shortness of breath", "chest tightness", "cough", "breathlessness"],
    "Bronchitis":           ["cough", "mucus production", "fatigue", "shortness of breath", "chest discomfort", "mild fever"],
    "COVID-19":             ["fever", "dry cough", "fatigue", "loss of taste", "loss of smell", "shortness of breath", "body aches"],
    "Dengue Fever":         ["high fever", "severe headache", "joint pain", "muscle pain", "rash", "nausea", "vomiting"],
    "Malaria":              ["cyclical fever", "chills", "sweating", "headache", "nausea", "vomiting", "muscle pain"],
    "Typhoid Fever":        ["sustained fever", "abdominal pain", "headache", "weakness", "constipation", "rash", "loss of appetite"],
    "Diabetes Mellitus":    ["frequent urination", "excessive thirst", "fatigue", "blurred vision", "slow healing", "weight loss"],
    "Hypertension":         ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath", "nosebleed"],
    "Heart Disease":        ["chest pain", "shortness of breath", "palpitations", "fatigue", "dizziness", "swollen legs"],
    "Stroke":               ["sudden numbness", "confusion", "trouble speaking", "vision problems", "severe headache", "dizziness"],
    "Kidney Disease":       ["swelling", "fatigue", "decreased urine", "nausea", "shortness of breath", "back pain"],
    "Liver Disease":        ["jaundice", "abdominal pain", "nausea", "fatigue", "dark urine", "loss of appetite", "swelling"],
    "Anemia":               ["fatigue", "pale skin", "shortness of breath", "dizziness", "cold hands", "brittle nails", "headache"],
    "Arthritis":            ["joint pain", "stiffness", "swelling", "reduced range of motion", "redness", "fatigue"],
    "Migraine":             ["severe headache", "nausea", "vomiting", "light sensitivity", "sound sensitivity", "aura"],
    "Epilepsy":             ["seizures", "confusion", "staring spells", "uncontrollable movements", "loss of consciousness"],
    "Depression":           ["persistent sadness", "loss of interest", "fatigue", "sleep disturbance", "appetite changes", "concentration problems"],
    "Anxiety Disorder":     ["excessive worry", "restlessness", "fatigue", "concentration problems", "irritability", "sleep disturbance", "muscle tension"],
    "Hypothyroidism":       ["fatigue", "weight gain", "cold intolerance", "constipation", "dry skin", "hair loss", "depression"],
    "Hyperthyroidism":      ["weight loss", "rapid heartbeat", "sweating", "nervousness", "tremors", "heat intolerance", "frequent bowel movements"],
    "Gastroenteritis":      ["nausea", "vomiting", "diarrhea", "abdominal cramps", "fever", "headache", "muscle aches"],
    "Appendicitis":         ["abdominal pain", "nausea", "vomiting", "fever", "loss of appetite", "rebound tenderness"],
    "Urinary Tract Infection": ["burning urination", "frequent urination", "cloudy urine", "pelvic pain", "fever", "back pain"],
    "Skin Allergy":         ["rash", "itching", "redness", "swelling", "hives", "dry skin", "blistering"],
    "Conjunctivitis":       ["red eyes", "eye discharge", "itching", "tearing", "swollen eyelids", "light sensitivity"],
    "Chickenpox":           ["itchy rash", "blisters", "fever", "fatigue", "headache", "loss of appetite"],
}

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]


def generate_dataset(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset and return as DataFrame."""
    random.seed(seed)
    np.random.seed(seed)

    records = []
    samples_per_disease = n_samples // len(DISEASES)
    remainder = n_samples % len(DISEASES)

    for i, disease in enumerate(DISEASES):
        count = samples_per_disease + (1 if i < remainder else 0)
        symptom_pool = DISEASE_SYMPTOMS[disease]

        for j in range(count):
            patient_id = f"P{len(records) + 1:05d}"
            age = int(np.random.normal(45, 18))
            age = max(1, min(100, age))
            gender = random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]

            # Pick 3–7 symptoms, mostly from disease pool, occasionally add noise
            n_syms = random.randint(3, 7)
            primary = random.sample(symptom_pool, min(n_syms, len(symptom_pool)))

            # Add 0–1 noise symptoms from other diseases
            if len(primary) < n_syms:
                all_other = [s for d, syms in DISEASE_SYMPTOMS.items()
                             if d != disease for s in syms]
                noise = random.sample(all_other, n_syms - len(primary))
                primary.extend(noise)

            symptoms_str = ", ".join(primary)
            symptom_count = len(primary)

            records.append({
                "Patient_ID": patient_id,
                "Age": age,
                "Gender": gender,
                "Symptoms": symptoms_str,
                "Symptom_Count": symptom_count,
                "Disease": disease,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("  Disease Prediction Model — Training Script")
    print("=" * 60)

    # 1. Load or generate dataset
    if os.path.exists(DATASET_PATH):
        print(f"\n[INFO] Loading existing dataset from: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        print("\n[INFO] Generating synthetic dataset (25,000 records, 30 diseases)...")
        df = generate_dataset(n_samples=25000)
        df.to_csv(DATASET_PATH, index=False)
        print(f"[INFO] Dataset saved to: {DATASET_PATH}")

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Disease classes: {df['Disease'].nunique()}")
    print(f"[INFO] Class distribution:\n{df['Disease'].value_counts().to_string()}\n")

    # 2. Import preprocess helper (same module used at inference)
    sys.path.insert(0, os.path.dirname(ML_DIR))
    from ml.preprocess import clean_symptoms

    print("[INFO] Cleaning Symptoms column...")
    df["Symptoms"] = df["Symptoms"].apply(clean_symptoms)

    # 3. Features & target
    X = df[["Age", "Gender", "Symptoms", "Symptom_Count"]]
    y = df["Disease"]

    # 4. Label-encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 6. Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ), "Symptoms"),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Gender"]),
            ("passthrough", "passthrough", ["Age", "Symptom_Count"]),
        ],
        remainder="drop",
    )

    # 7. Full Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 8. Train
    print("\n[INFO] Training RandomForestClassifier (n_estimators=200)...")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # 9. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc * 100:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 10. Save artifacts
    print("\n[INFO] Saving artifacts...")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"  [OK] model.pkl      -> {MODEL_PATH}")

    # Extract and save TF-IDF vectorizer separately (for reference / Flask use)
    tfidf_vectorizer = pipeline.named_steps["preprocessor"].named_transformers_["tfidf"]
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    print(f"  [OK] vectorizer.pkl -> {VECTORIZER_PATH}")

    # Save encoders dict (LabelEncoder for target)
    encoders = {
        "label_encoder": le,
        "classes": list(le.classes_),
    }
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"  [OK] encoders.pkl   -> {ENCODERS_PATH}")

    print("\n[DONE] All artifacts saved. You can now run: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
