from __future__ import annotations

import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .constants import MODEL_PATH


def bucket_age(age: int) -> str:
    if age <= 12:
        return "child"
    if age <= 19:
        return "teen"
    if age <= 59:
        return "adult"
    return "senior"


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def build_feature_text(symptoms: str, age_group: str, gender: str) -> str:
    symptom_text = normalize_text(symptoms)
    gender_text = normalize_text(gender)
    age_text = normalize_text(age_group)
    return f"symptoms {symptom_text} age {age_text} gender {gender_text}"


def prepare_training_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    prepared = dataset.copy()
    prepared["feature_text"] = prepared.apply(
        lambda row: build_feature_text(
            symptoms=str(row["symptoms"]),
            age_group=str(row["age_group"]),
            gender=str(row["gender"]),
        ),
        axis=1,
    )
    return prepared


def build_label_metadata(dataset: pd.DataFrame) -> dict[str, dict[str, str]]:
    unique_rows = dataset.drop_duplicates("recommended_drug")
    metadata: dict[str, dict[str, str]] = {}
    for row in unique_rows.to_dict("records"):
        metadata[row["recommended_drug"]] = {
            "category": row["drug_category"],
            "advisor_note": row["advisor_note"],
            "seek_care_if": row["seek_care_if"],
        }
    return metadata


def train_bundle(dataset: pd.DataFrame) -> dict[str, object]:
    prepared = prepare_training_frame(dataset)
    features = prepared["feature_text"]
    labels = prepared["recommended_drug"]

    pipeline = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "dataset_size": int(len(dataset)),
        "class_count": int(labels.nunique()),
    }

    pipeline.fit(features, labels)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_metadata": build_label_metadata(dataset),
        "dataset_size": int(len(dataset)),
        "feedback_rows": int((dataset["source"] == "confirmed-feedback").sum()),
    }


def save_bundle(bundle: dict[str, object], model_path=MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_bundle(model_path=MODEL_PATH) -> dict[str, object]:
    return joblib.load(model_path)


def predict_drugs(
    bundle: dict[str, object],
    symptoms: str,
    age: int,
    gender: str,
    top_k: int = 3,
) -> list[dict[str, object]]:
    pipeline: Pipeline = bundle["pipeline"]
    feature_text = build_feature_text(symptoms=symptoms, age_group=bucket_age(age), gender=gender)
    probabilities = pipeline.predict_proba([feature_text])[0]
    classes = pipeline.classes_
    order = np.argsort(probabilities)[::-1][:top_k]

    predictions: list[dict[str, object]] = []
    label_metadata: dict[str, dict[str, str]] = bundle["label_metadata"]
    for index in order:
        label = str(classes[index])
        metadata = label_metadata.get(label, {})
        predictions.append(
            {
                "drug": label,
                "confidence": float(probabilities[index]),
                "category": metadata.get("category", "Educational guidance"),
                "advisor_note": metadata.get("advisor_note", ""),
                "seek_care_if": metadata.get("seek_care_if", ""),
            }
        )
    return predictions
