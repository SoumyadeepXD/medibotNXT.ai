from __future__ import annotations

from .combined import train_and_persist_combined
from .constants import COMBINED_MODEL_PATH, INTENT_MODEL_PATH, MODEL_PATH
from .intents import train_and_persist_intents
from .modeling import save_bundle, train_bundle
from .storage import ensure_runtime_files, load_training_dataset


def train_and_persist() -> dict[str, object]:
    ensure_runtime_files()
    dataset = load_training_dataset()
    bundle = train_bundle(dataset)
    save_bundle(bundle, MODEL_PATH)
    return bundle


def train_all_models() -> dict[str, dict[str, object]]:
    drug_bundle = train_and_persist()
    intent_bundle = train_and_persist_intents()
    combined_bundle = train_and_persist_combined()
    return {
        "drug": drug_bundle,
        "intent": intent_bundle,
        "combined": combined_bundle,
    }


if __name__ == "__main__":
    trained = train_all_models()
    print(f"Drug model written to: {MODEL_PATH}")
    print(f"Intent model written to: {INTENT_MODEL_PATH}")
    print(f"Combined model written to: {COMBINED_MODEL_PATH}")
    for name, bundle in trained.items():
        metrics = bundle["metrics"]
        print(
            f"{name}: samples={bundle['dataset_size']} classes={metrics['class_count']} "
            f"accuracy={metrics['accuracy']:.2%} macro_f1={metrics['macro_f1']:.2%}"
        )
