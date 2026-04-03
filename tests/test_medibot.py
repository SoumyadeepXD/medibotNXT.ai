from __future__ import annotations

import unittest

from medibot.combined import predict_combined, train_combined_bundle
from medibot.intents import train_intent_bundle, predict_intent
from medibot.modeling import predict_drugs, train_bundle
from medibot.reporting import build_report_pdf
from medibot.safety import analyze_symptoms, requires_urgent_care
from medibot.storage import load_training_dataset


class MediBotSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = train_bundle(load_training_dataset())
        cls.intent_bundle = train_intent_bundle()
        cls.combined_bundle = train_combined_bundle()

    def test_predicts_acidity_pattern(self) -> None:
        predictions = predict_drugs(
            self.bundle,
            symptoms="burning acidity and sour reflux after dinner",
            age=29,
            gender="Female",
        )
        self.assertEqual(predictions[0]["drug"], "Pantoprazole")

    def test_predicts_allergy_pattern(self) -> None:
        predictions = predict_drugs(
            self.bundle,
            symptoms="itchy eyes, sneezing, and runny nose from dust",
            age=21,
            gender="Male",
        )
        self.assertEqual(predictions[0]["drug"], "Cetirizine")

    def test_detects_red_flags(self) -> None:
        findings = analyze_symptoms("chest pain and difficulty breathing")
        self.assertTrue(requires_urgent_care(findings))

    def test_pdf_report_generation(self) -> None:
        predictions = predict_drugs(
            self.bundle,
            symptoms="burning acidity after spicy food",
            age=28,
            gender="Female",
        )
        pdf_bytes = build_report_pdf(
            name="Test User",
            age=28,
            gender="Female",
            symptoms="burning acidity after spicy food",
            warnings=[],
            predictions=predictions,
            generated_at="2026-04-04T01:00:00+00:00",
            blocked_for_safety=False,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_intent_prediction_for_cuts(self) -> None:
        predictions = predict_intent(self.intent_bundle, "Which medicine to apply for cuts?")
        self.assertEqual(predictions[0]["tag"], "Cuts")

    def test_combined_prediction_for_drug_query(self) -> None:
        predictions = predict_combined(
            self.combined_bundle,
            "burning acidity after spicy food",
            age=28,
            gender="Female",
        )
        self.assertEqual(predictions[0]["kind"], "drug")
        self.assertEqual(predictions[0]["name"], "Pantoprazole")

    def test_combined_prediction_for_intent_query(self) -> None:
        predictions = predict_combined(
            self.combined_bundle,
            "What should I do for cuts?",
            age=28,
            gender="Female",
        )
        self.assertEqual(predictions[0]["kind"], "intent")
        self.assertEqual(predictions[0]["name"], "Cuts")


if __name__ == "__main__":
    unittest.main()
