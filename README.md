# MEDIBOT

MEDIBOT is a Streamlit-based educational demo that turns the project brief from `mediboooot.pdf` into a runnable application. It uses TF-IDF + Multinomial Naive Bayes to map symptom text to lightweight, over-the-counter style drug guidance, generates PDF session reports, stores logs, and accepts confirmed feedback for later retraining. It also includes an intent-based first-aid assistant powered by the `intents.json` dataset you shared, plus a unified model trained on both datasets together.

## What It Includes

- Natural-language symptom entry with age and gender context
- A safety layer that blocks recommendations for urgent red-flag symptoms
- Top-3 educational drug matches with confidence scores
- A second first-aid assistant tab trained from `data/intents.json`
- A unified classifier trained on both the drug dataset and the intent dataset
- PDF session report download
- Local session logging in CSV
- Optional clinician-confirmed feedback capture for retraining
- Re-trainable scikit-learn model bundle saved with Joblib

## Project Structure

- `app.py` - Streamlit application
- `medibot/modeling.py` - feature building, training, prediction
- `medibot/intents.py` - intent-model training and prediction for first-aid Q&A
- `medibot/combined.py` - combined classifier trained on both datasets
- `medibot/safety.py` - urgent symptom keyword checks
- `medibot/reporting.py` - PDF generation
- `medibot/storage.py` - CSV persistence and dataset loading
- `medibot/train.py` - model training entrypoint
- `data/symptom_drug_dataset.csv` - seeded symptom-drug examples
- `data/intents.json` - first-aid intent dataset
- `data/user_entries.csv` - generated runtime session log
- `data/feedback_entries.csv` - generated runtime confirmed feedback log

## Local Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m medibot.train
.venv/bin/streamlit run app.py
```

## Testing

```bash
.venv/bin/python -m unittest discover -s tests
```

## Safety Note

This app is intentionally educational only. It does not diagnose disease, prescribe medicine, calculate dosage, or replace a doctor, pharmacist, or emergency service.
