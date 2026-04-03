from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .constants import COMBINED_MODEL_PATH
from .intents import build_intent_metadata, load_intents_payload
from .modeling import build_feature_text, bucket_age
from .storage import load_training_dataset


def build_combined_frame() -> pd.DataFrame:
    drug_dataset = load_training_dataset()
    rows: list[dict[str, str]] = []

    for entry in drug_dataset.to_dict("records"):
        label_id = f"drug::{entry['recommended_drug']}"
        symptom_text = str(entry["symptoms"]).strip()
        rows.append({"feature_text": symptom_text, "label_id": label_id})
        rows.append(
            {
                "feature_text": build_feature_text(
                    symptoms=symptom_text,
                    age_group=str(entry["age_group"]),
                    gender=str(entry["gender"]),
                ),
                "label_id": label_id,
            }
        )

    payload = load_intents_payload()
    for item in payload.get("intents", []):
        tag = str(item.get("tag", "")).strip()
        label_id = f"intent::{tag}"
        for pattern in item.get("patterns", []):
            rows.append({"feature_text": str(pattern).strip(), "label_id": label_id})

    return pd.DataFrame(rows)


def build_combined_metadata() -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}

    drug_dataset = load_training_dataset().drop_duplicates("recommended_drug")
    for entry in drug_dataset.to_dict("records"):
        label_id = f"drug::{entry['recommended_drug']}"
        metadata[label_id] = {
            "kind": "drug",
            "name": entry["recommended_drug"],
            "category": entry["drug_category"],
            "advisor_note": entry["advisor_note"],
            "seek_care_if": entry["seek_care_if"],
        }

    intent_metadata = build_intent_metadata()
    for tag, item in intent_metadata.items():
        label_id = f"intent::{tag}"
        metadata[label_id] = {
            "kind": "intent",
            "name": tag,
            "response": item.get("responses", ["No response available."])[0],
            "examples": item.get("patterns", [])[:3],
        }

    return metadata


def train_combined_bundle() -> dict[str, object]:
    frame = build_combined_frame()
    features = frame["feature_text"]
    labels = frame["label_id"]

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
        test_size=0.3,
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
        "dataset_size": int(len(frame)),
        "class_count": int(labels.nunique()),
    }

    pipeline.fit(features, labels)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": build_combined_metadata(),
        "dataset_size": int(len(frame)),
    }


def save_combined_bundle(bundle: dict[str, object], model_path=COMBINED_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_combined_bundle(model_path=COMBINED_MODEL_PATH) -> dict[str, object]:
    return joblib.load(model_path)


def train_and_persist_combined() -> dict[str, object]:
    bundle = train_combined_bundle()
    save_combined_bundle(bundle, COMBINED_MODEL_PATH)
    return bundle


def predict_combined(
    bundle: dict[str, object],
    query: str,
    *,
    age: int | None = None,
    gender: str | None = None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    pipeline: Pipeline = bundle["pipeline"]

    variants = [query.strip()]
    if age is not None and gender:
        variants.append(build_feature_text(query, bucket_age(age), gender))

    probabilities = pipeline.predict_proba(variants).mean(axis=0)
    classes = pipeline.classes_
    order = np.argsort(probabilities)[::-1][:top_k]
    metadata: dict[str, dict[str, object]] = bundle["metadata"]

    predictions: list[dict[str, object]] = []
    for index in order:
        label_id = str(classes[index])
        item = metadata.get(label_id, {})
        predictions.append(
            {
                "label_id": label_id,
                "kind": item.get("kind", "unknown"),
                "name": item.get("name", label_id),
                "confidence": float(probabilities[index]),
                "category": item.get("category", ""),
                "advisor_note": item.get("advisor_note", ""),
                "seek_care_if": item.get("seek_care_if", ""),
                "response": item.get("response", ""),
                "examples": item.get("examples", []),
            }
        )
    return predictions
