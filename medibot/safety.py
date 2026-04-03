from __future__ import annotations

import re


RED_FLAG_RULES = [
    {
        "keywords": ["chest pain", "pressure in chest"],
        "level": "urgent",
        "title": "Chest symptoms need clinician review",
        "message": "Chest pain or pressure should not be self-managed with an AI recommendation.",
    },
    {
        "keywords": ["shortness of breath", "difficulty breathing", "can not breathe", "cannot breathe", "breathing trouble"],
        "level": "urgent",
        "title": "Breathing trouble detected",
        "message": "Breathing difficulty needs urgent medical attention, especially if it started suddenly.",
    },
    {
        "keywords": ["fainting", "passed out", "unconscious", "confusion", "seizure"],
        "level": "urgent",
        "title": "Neurologic warning signs detected",
        "message": "Confusion, fainting, or seizure-like symptoms should be assessed immediately.",
    },
    {
        "keywords": ["blood in stool", "bloody stool", "vomiting blood", "black stool", "bloody cough", "coughing blood"],
        "level": "urgent",
        "title": "Bleeding symptom detected",
        "message": "Possible bleeding symptoms need urgent medical review instead of self-treatment.",
    },
    {
        "keywords": ["lip swelling", "tongue swelling", "throat closing", "anaphylaxis"],
        "level": "urgent",
        "title": "Possible severe allergic reaction",
        "message": "Swelling of the lips, tongue, or throat may indicate an emergency reaction.",
    },
    {
        "keywords": ["pregnant", "pregnancy"],
        "level": "caution",
        "title": "Pregnancy warning",
        "message": "Medicine choices during pregnancy should be confirmed with a clinician or pharmacist.",
    },
    {
        "keywords": ["kidney disease", "liver disease", "ulcer", "ulcers"],
        "level": "caution",
        "title": "Chronic-condition warning",
        "message": "Existing kidney, liver, or ulcer conditions change the safety profile of common medicines.",
    },
]


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def analyze_symptoms(symptoms: str) -> list[dict[str, str]]:
    text = normalize_text(symptoms)
    findings: list[dict[str, str]] = []
    for rule in RED_FLAG_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            findings.append(rule)
    return findings


def requires_urgent_care(findings: list[dict[str, str]]) -> bool:
    return any(item["level"] == "urgent" for item in findings)

