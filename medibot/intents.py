from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .constants import INTENT_MODEL_PATH, INTENTS_DATA_PATH


def load_intents_payload(path: Path = INTENTS_DATA_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_intent_frame(path: Path = INTENTS_DATA_PATH) -> pd.DataFrame:
    payload = load_intents_payload(path)
    rows: list[dict[str, str]] = []
    for item in payload.get("intents", []):
        tag = str(item.get("tag", "")).strip()
        responses = item.get("responses", [])
        response = responses[0] if responses else "No response available."
        for pattern in item.get("patterns", []):
            rows.append(
                {
                    "pattern": str(pattern).strip(),
                    "tag": tag,
                    "response": response,
                }
            )
    return pd.DataFrame(rows)


def build_intent_metadata(path: Path = INTENTS_DATA_PATH) -> dict[str, dict[str, object]]:
    payload = load_intents_payload(path)
    metadata: dict[str, dict[str, object]] = {}
    for item in payload.get("intents", []):
        tag = str(item.get("tag", "")).strip()
        metadata[tag] = {
            "responses": item.get("responses", []),
            "patterns": item.get("patterns", []),
        }
    return metadata


def train_intent_bundle(path: Path = INTENTS_DATA_PATH) -> dict[str, object]:
    frame = build_intent_frame(path)
    features = frame["pattern"]
    labels = frame["tag"]

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
        "metadata": build_intent_metadata(path),
        "dataset_size": int(len(frame)),
    }


def save_intent_bundle(bundle: dict[str, object], model_path: Path = INTENT_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_intent_bundle(model_path: Path = INTENT_MODEL_PATH) -> dict[str, object]:
    return joblib.load(model_path)


def train_and_persist_intents() -> dict[str, object]:
    bundle = train_intent_bundle(INTENTS_DATA_PATH)
    save_intent_bundle(bundle, INTENT_MODEL_PATH)
    return bundle


def predict_intent(bundle: dict[str, object], query: str, top_k: int = 3) -> list[dict[str, object]]:
    pipeline: Pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([query])[0]
    classes = pipeline.classes_
    order = np.argsort(probabilities)[::-1][:top_k]
    metadata: dict[str, dict[str, object]] = bundle["metadata"]

    predictions: list[dict[str, object]] = []
    for index in order:
        tag = str(classes[index])
        tag_data = metadata.get(tag, {})
        responses = tag_data.get("responses", [])
        patterns = tag_data.get("patterns", [])
        predictions.append(
            {
                "tag": tag,
                "confidence": float(probabilities[index]),
                "response": responses[0] if responses else "No response available.",
                "examples": patterns[:3],
            }
        )
    return predictions
