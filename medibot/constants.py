from __future__ import annotations

from pathlib import Path

APP_NAME = "MEDIBOT"
APP_SUBTITLE = "Gentle health guidance for everyday symptoms, common care questions, and first-aid basics."

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
MODELS_DIR = WORKSPACE_ROOT / "models"

SEED_DATASET_PATH = DATA_DIR / "symptom_drug_dataset.csv"
SESSION_LOG_PATH = DATA_DIR / "user_entries.csv"
FEEDBACK_PATH = DATA_DIR / "feedback_entries.csv"
MODEL_PATH = MODELS_DIR / "medibot_pipeline.joblib"
INTENTS_DATA_PATH = DATA_DIR / "intents.json"
INTENT_MODEL_PATH = MODELS_DIR / "medibot_intent_pipeline.joblib"
COMBINED_MODEL_PATH = MODELS_DIR / "medibot_combined_pipeline.joblib"

SESSION_COLUMNS = [
    "timestamp",
    "name",
    "age",
    "age_group",
    "gender",
    "symptoms",
    "predicted_drug",
    "predicted_category",
    "confidence",
    "blocked_for_safety",
    "warning_summary",
]

FEEDBACK_COLUMNS = [
    "timestamp",
    "name",
    "age",
    "age_group",
    "gender",
    "symptoms",
    "confirmed_drug",
    "notes",
]

TRAINING_COLUMNS = [
    "age_group",
    "gender",
    "symptoms",
    "recommended_drug",
    "drug_category",
    "advisor_note",
    "seek_care_if",
    "source",
]

GENERAL_SAFETY_NOTES = [
    "Educational demo only. It is not a diagnosis tool and it should not replace a doctor, pharmacist, or emergency care.",
    "MEDIBOT is intentionally limited to common symptom patterns and lightweight over-the-counter style guidance.",
    "Pregnancy, chronic kidney or liver disease, ulcers, medication allergies, or ongoing prescriptions need clinician review before any medicine choice.",
]
