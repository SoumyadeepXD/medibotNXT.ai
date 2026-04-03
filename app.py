from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from medibot.combined import (
    load_combined_bundle,
    predict_combined,
    train_and_persist_combined,
)
from medibot.constants import (
    APP_NAME,
    APP_SUBTITLE,
    COMBINED_MODEL_PATH,
    GENERAL_SAFETY_NOTES,
    INTENT_MODEL_PATH,
    INTENTS_DATA_PATH,
    MODEL_PATH,
)
from medibot.intents import load_intent_bundle, predict_intent, train_and_persist_intents
from medibot.modeling import bucket_age, predict_drugs
from medibot.reporting import build_report_pdf
from medibot.safety import analyze_symptoms, requires_urgent_care
from medibot.storage import (
    append_feedback_entry,
    append_session_log,
    ensure_runtime_files,
    load_recent_feedback,
    load_recent_sessions,
    load_seed_dataset,
    utc_timestamp,
)
from medibot.train import train_and_persist


st.set_page_config(page_title=APP_NAME, page_icon="💊", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, object]:
    ensure_runtime_files()
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_persist()


@st.cache_data(show_spinner=False)
def dataset_snapshot() -> pd.DataFrame:
    return load_seed_dataset()


@st.cache_resource(show_spinner=False)
def load_first_aid_bundle() -> dict[str, object] | None:
    if not INTENTS_DATA_PATH.exists():
        return None
    if INTENT_MODEL_PATH.exists():
        return load_intent_bundle()
    return train_and_persist_intents()


@st.cache_resource(show_spinner=False)
def load_unified_bundle() -> dict[str, object] | None:
    if not INTENTS_DATA_PATH.exists():
        return None
    if COMBINED_MODEL_PATH.exists():
        return load_combined_bundle()
    return train_and_persist_combined()


def retrain_model() -> dict[str, object]:
    bundle = train_and_persist()
    load_model_bundle.clear()
    dataset_snapshot.clear()
    return bundle


def render_shell() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(125, 211, 252, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(45, 212, 191, 0.12), transparent 22%),
                linear-gradient(180deg, #07101d 0%, #0b1324 45%, #0e182a 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 17, 31, 0.97), rgba(12, 21, 37, 0.97));
            border-right: 1px solid rgba(125, 211, 252, 0.12);
        }
        [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: rgba(9, 18, 33, 0.78);
            border: 1px solid rgba(125, 211, 252, 0.12);
            padding: 0.35rem;
            border-radius: 18px;
        }
        [data-baseweb="tab"] {
            color: #c8d7ea;
            border-radius: 14px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(61, 137, 255, 0.18), rgba(24, 205, 170, 0.18));
            color: #ffffff;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(125, 211, 252, 0.25);
            background: linear-gradient(135deg, #142847, #0f3f54);
            color: #eff6ff;
            font-weight: 700;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(125, 211, 252, 0.38);
            background: linear-gradient(135deg, #17345c, #14566a);
        }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
        }
        .hero {
            padding: 2.1rem 2.2rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(125, 211, 252, 0.16), transparent 24%),
                linear-gradient(135deg, rgba(12, 22, 40, 0.95), rgba(14, 28, 49, 0.92));
            color: #eff6ff;
            border: 1px solid rgba(125, 211, 252, 0.12);
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.4);
            margin-bottom: 1.1rem;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(125, 211, 252, 0.12);
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
            color: #bde9ff;
        }
        .hero h1 {
            margin: 0;
            font-family: "Avenir Next", "Trebuchet MS", sans-serif;
            font-size: 2.9rem;
            letter-spacing: 0.02em;
        }
        .hero p {
            margin-top: 0.65rem;
            margin-bottom: 0;
            line-height: 1.75;
            max-width: 800px;
            font-size: 1.03rem;
            color: #d9e7f7;
        }
        .soft-card {
            background: rgba(15, 25, 43, 0.78);
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            color: #dce8f7;
        }
        .soft-card-title {
            font-weight: 800;
            color: #f8fbff;
            margin-bottom: 0.35rem;
        }
        .soft-card-text {
            color: #b8c8da;
            line-height: 1.6;
            font-size: 0.94rem;
        }
        .danger-card {
            background: linear-gradient(135deg, rgba(91, 33, 41, 0.85), rgba(73, 23, 31, 0.75));
            border: 1px solid rgba(251, 113, 133, 0.25);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
            color: #ffe4e8;
        }
        .result-card {
            background: linear-gradient(180deg, rgba(15, 27, 46, 0.96), rgba(12, 23, 39, 0.94));
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 14px 34px rgba(2, 6, 23, 0.35);
        }
        .result-label {
            color: #9ac7e6;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .result-drug {
            margin: 0;
            color: #f8fbff;
            font-size: 2rem;
            font-weight: 800;
        }
        .result-category {
            color: #7dd3fc;
            font-weight: 700;
            margin-top: 0.2rem;
            margin-bottom: 0.9rem;
        }
        .mini-card {
            background: rgba(14, 25, 43, 0.9);
            border-radius: 18px;
            border: 1px solid rgba(125, 211, 252, 0.12);
            padding: 1rem;
            height: 100%;
        }
        .mini-title {
            font-weight: 700;
            color: #eef6ff;
            margin-bottom: 0.3rem;
        }
        .mini-subtitle {
            color: #9cb1c9;
            margin-bottom: 0.5rem;
            font-size: 0.92rem;
        }
        .plain-note {
            background: rgba(12, 23, 39, 0.82);
            border: 1px solid rgba(125, 211, 252, 0.1);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            color: #d9e7f7;
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-badge">CALM, EVERYDAY HEALTH SUPPORT</div>
            <h1>{APP_NAME}</h1>
            <p>{APP_SUBTITLE} Use simple words like you would with a caring friend. MEDIBOT can guide everyday symptom questions,
            suggest common medicine matches, and share basic care advice while clearly warning when something needs real medical help.</p>
            <div class="hero-grid">
                <div class="soft-card">
                    <div class="soft-card-title">1. Tell MEDIBOT what is going on</div>
                    <div class="soft-card-text">Short plain sentences are enough. Example: “my throat hurts and I keep coughing at night”.</div>
                </div>
                <div class="soft-card">
                    <div class="soft-card-title">2. Get gentle guidance</div>
                    <div class="soft-card-text">You will see either a medicine-style suggestion, a simple care guide, or a warning to get real medical help.</div>
                </div>
                <div class="soft-card">
                    <div class="soft-card-title">3. Use it as support, not a diagnosis</div>
                    <div class="soft-card-text">This tool is for learning and light guidance. It should never replace a doctor, pharmacist, or emergency care.</div>
                </div>
            </div>
            <div class="danger-card">
                <strong>Get urgent help now</strong><br/>
                Chest pain, breathing trouble, fainting, confusion, bleeding, or swelling of the lips/tongue should be treated as urgent.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: dict[str, object]) -> None:
    metrics = bundle["metrics"]
    st.sidebar.title("Start Here")
    st.sidebar.caption("A calm place to ask simple health questions.")
    st.sidebar.markdown(
        """
        - Use everyday language.
        - Include your age if you can.
        - If something feels severe or frightening, skip the app and get real medical help.
        """
    )
    st.sidebar.warning(
        "Urgent signs: chest pain, trouble breathing, fainting, heavy bleeding, confusion, or swelling of the lips/tongue."
    )
    st.sidebar.info(
        "MEDIBOT does not prescribe medicine or diagnose disease. It gives light guidance only."
    )

    first_aid_bundle = load_first_aid_bundle()
    unified_bundle = load_unified_bundle()

    with st.sidebar.expander("Project owner tools"):
        st.caption("These controls are mainly for the project owner or evaluator.")
        st.metric("Medicine model accuracy", f"{metrics['accuracy']:.0%}")
        st.metric("Medicine examples", int(bundle["dataset_size"]))
        st.metric("Saved feedback rows", int(bundle["feedback_rows"]))
        if first_aid_bundle:
            st.metric("Care guide accuracy", f"{first_aid_bundle['metrics']['accuracy']:.0%}")
        if unified_bundle:
            st.metric("Unified model accuracy", f"{unified_bundle['metrics']['accuracy']:.0%}")
        st.caption(f"Last medicine training run: {bundle['trained_at']}")
        if st.button("Retrain medicine model with confirmed feedback", use_container_width=True):
            with st.spinner("Retraining MEDIBOT..."):
                refreshed = retrain_model()
            st.sidebar.success(
                f"Retrained on {refreshed['dataset_size']} rows with {refreshed['feedback_rows']} confirmed feedback examples."
            )
            st.rerun()


def confidence_phrase(score: float) -> str:
    if score >= 0.85:
        return "strong match"
    if score >= 0.65:
        return "likely match"
    if score >= 0.45:
        return "possible match"
    return "low-confidence match"


def render_result_card(result: dict[str, object]) -> None:
    top = result["predictions"][0]
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Possible next step</div>
            <p class="result-drug">{top['drug']}</p>
            <div class="result-category">{top['category']} - {confidence_phrase(top['confidence'])}</div>
            <strong>Why MEDIBOT picked this</strong>
            <div>{top['advisor_note']}</div>
            <div style="margin-top:0.85rem;"><strong>Get real medical help if</strong></div>
            <div>{top['seek_care_if']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alternatives(result: dict[str, object]) -> None:
    alternatives = result["predictions"][1:]
    if not alternatives:
        return
    st.subheader("Other possibilities")
    columns = st.columns(len(alternatives))
    for column, item in zip(columns, alternatives):
        with column:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">{item['drug']}</div>
                    <div class="mini-subtitle">{item['category']}</div>
                    <div><strong>{confidence_phrase(item['confidence'])}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_warning_messages(warnings: list[dict[str, str]]) -> None:
    for warning in warnings:
        if warning["level"] == "urgent":
            st.error(f"{warning['title']}: {warning['message']}")
        else:
            st.warning(f"{warning['title']}: {warning['message']}")


def render_feedback_capture(
    result: dict[str, object],
    bundle: dict[str, object],
    *,
    form_key: str,
) -> None:
    top_prediction = result["predictions"][0]
    with st.expander("Project owner only: save a clinician-confirmed outcome"):
        st.caption(
            "Only use this if a clinician or pharmacist later confirmed the medicine choice."
        )
        with st.form(form_key):
            confirmed_drug = st.selectbox(
                "Confirmed medicine name",
                options=sorted(bundle["label_metadata"].keys()),
                index=sorted(bundle["label_metadata"].keys()).index(top_prediction["drug"]),
            )
            notes = st.text_input("Optional note")
            feedback_submitted = st.form_submit_button("Save Learning Example")

        if feedback_submitted:
            append_feedback_entry(
                {
                    "timestamp": utc_timestamp(),
                    "name": result["name"],
                    "age": result["age"],
                    "age_group": bucket_age(int(result["age"])),
                    "gender": result["gender"],
                    "symptoms": result["symptoms"],
                    "confirmed_drug": confirmed_drug,
                    "notes": notes,
                }
            )
            st.success(
                "Confirmed feedback saved. Use the sidebar retrain button when you want MEDIBOT to learn from it."
            )


def render_drug_result_panel(
    result: dict[str, object],
    bundle: dict[str, object],
    *,
    model_accuracy: float,
    model_label: str,
    feedback_form_key: str,
    feedback_bundle: dict[str, object] | None = None,
) -> None:
    st.subheader("What MEDIBOT suggests")
    render_warning_messages(result["warnings"])

    if result.get("blocked", False):
        st.error(
            "MEDIBOT stopped here because your words matched urgent warning signs. Please contact urgent care or emergency services instead of relying on this app."
        )
        return

    render_result_card(result)

    metric_one, metric_two, metric_three = st.columns(3)
    top_prediction = result["predictions"][0]
    metric_one.metric("Match strength", confidence_phrase(top_prediction["confidence"]).title())
    metric_two.metric(model_label, f"{model_accuracy:.0%}")
    metric_three.metric("Knowledge examples", int(bundle["dataset_size"]))

    render_alternatives(result)

    if result.get("pdf_bytes"):
        st.download_button(
            "Download PDF Report",
            data=result["pdf_bytes"],
            file_name=f"medibot-report-{result['timestamp'].replace(':', '-')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    render_feedback_capture(result, feedback_bundle or bundle, form_key=feedback_form_key)


def handle_unified_submission(
    unified_bundle: dict[str, object],
    *,
    name: str,
    age: int,
    gender: str,
    query: str,
) -> None:
    warnings = analyze_symptoms(query)
    blocked = requires_urgent_care(warnings)
    timestamp = utc_timestamp()
    warning_summary = " | ".join(item["title"] for item in warnings)

    if blocked:
        append_session_log(
            {
                "timestamp": timestamp,
                "name": name,
                "age": age,
                "age_group": bucket_age(age),
                "gender": gender,
                "symptoms": query,
                "predicted_drug": "Urgent clinical review suggested",
                "predicted_category": "Safety override",
                "confidence": 0,
                "blocked_for_safety": True,
                "warning_summary": warning_summary,
            }
        )
        st.session_state["last_assistant_result"] = {
            "interaction_mode": "Guided help (recommended)",
            "route_kind": "safety",
            "blocked": True,
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": query,
            "warnings": warnings,
            "predictions": [],
            "raw_predictions": [],
            "timestamp": timestamp,
        }
        return

    predictions = predict_combined(
        unified_bundle,
        query,
        age=age,
        gender=gender,
    )
    top = predictions[0]

    append_session_log(
        {
            "timestamp": timestamp,
            "name": name,
            "age": age,
            "age_group": bucket_age(age),
            "gender": gender,
            "symptoms": query,
            "predicted_drug": top["name"],
            "predicted_category": top["category"] if top["kind"] == "drug" else "Intent route",
            "confidence": round(top["confidence"], 4),
            "blocked_for_safety": False,
            "warning_summary": warning_summary,
        }
    )

    result: dict[str, object] = {
        "interaction_mode": "Smart Route (Recommended)",
        "route_kind": top["kind"],
        "blocked": False,
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": query,
        "warnings": warnings,
        "timestamp": timestamp,
        "raw_predictions": predictions,
    }

    if top["kind"] == "drug":
        drug_predictions = [
            {
                "drug": item["name"],
                "confidence": item["confidence"],
                "category": item["category"],
                "advisor_note": item["advisor_note"],
                "seek_care_if": item["seek_care_if"],
            }
            for item in predictions
            if item["kind"] == "drug"
        ]
        drug_predictions = drug_predictions[:3] or [
            {
                "drug": top["name"],
                "confidence": top["confidence"],
                "category": top["category"],
                "advisor_note": top["advisor_note"],
                "seek_care_if": top["seek_care_if"],
            }
        ]
        result["predictions"] = drug_predictions
        result["pdf_bytes"] = build_report_pdf(
            name=name,
            age=age,
            gender=gender,
            symptoms=query,
            warnings=warnings,
            predictions=drug_predictions,
            generated_at=timestamp,
            blocked_for_safety=False,
        )
    else:
        result["predictions"] = predictions

    st.session_state["last_assistant_result"] = result


def handle_prediction_submission(bundle: dict[str, object], *, name: str, age: int, gender: str, symptoms: str) -> None:
    warnings = analyze_symptoms(symptoms)
    blocked = requires_urgent_care(warnings)
    timestamp = utc_timestamp()
    warning_summary = " | ".join(item["title"] for item in warnings)

    if blocked:
        append_session_log(
            {
                "timestamp": timestamp,
                "name": name,
                "age": age,
                "age_group": bucket_age(age),
                "gender": gender,
                "symptoms": symptoms,
                "predicted_drug": "Urgent clinical review suggested",
                "predicted_category": "Safety override",
                "confidence": 0,
                "blocked_for_safety": True,
                "warning_summary": warning_summary,
            }
        )
        result = {
            "interaction_mode": "Drug Guidance Only",
            "blocked": True,
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "warnings": warnings,
            "predictions": [],
            "timestamp": timestamp,
        }
        st.session_state["last_result"] = result
        st.session_state["last_drug_result"] = result
        return

    predictions = predict_drugs(bundle, symptoms=symptoms, age=age, gender=gender)
    top = predictions[0]

    append_session_log(
        {
            "timestamp": timestamp,
            "name": name,
            "age": age,
            "age_group": bucket_age(age),
            "gender": gender,
            "symptoms": symptoms,
            "predicted_drug": top["drug"],
            "predicted_category": top["category"],
            "confidence": round(top["confidence"], 4),
            "blocked_for_safety": False,
            "warning_summary": warning_summary,
        }
    )

    result = {
        "interaction_mode": "Drug Guidance Only",
        "blocked": False,
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "warnings": warnings,
        "predictions": predictions,
        "timestamp": timestamp,
        "pdf_bytes": build_report_pdf(
            name=name,
            age=age,
            gender=gender,
            symptoms=symptoms,
            warnings=warnings,
            predictions=predictions,
            generated_at=timestamp,
            blocked_for_safety=False,
        ),
    }
    st.session_state["last_result"] = result
    st.session_state["last_drug_result"] = result


def assistant_hub_tab(bundle: dict[str, object], unified_bundle: dict[str, object] | None) -> None:
    st.subheader("Get Help")
    st.caption(
        "Start with the guided mode if you are unsure. It can decide whether your message fits a simple care guide or a medicine-oriented suggestion."
    )

    modes = ["Guided help (recommended)", "Medicine suggestions only"]
    if unified_bundle is None:
        modes = ["Medicine suggestions only"]

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        mode = st.radio(
            "How would you like MEDIBOT to help?",
            options=modes,
            horizontal=True,
        )

        with st.form("assistant_hub_form", clear_on_submit=False):
            name = st.text_input("Your name (optional)", key="assistant_name")
            age = st.number_input("Age", min_value=1, max_value=110, value=28, step=1, key="assistant_age")
            gender = st.selectbox(
                "Gender (optional, helps tailor the result a little)",
                options=["Female", "Male", "Non-binary", "Prefer not to say"],
                index=3,
                key="assistant_gender",
            )
            query_label = (
                "In your own words, what is going on?"
                if mode == "Guided help (recommended)"
                else "Tell me your symptoms"
            )
            query_placeholder = (
                "Example: my throat hurts and I keep coughing at night OR what should I do for a cut?"
                if mode == "Guided help (recommended)"
                else "Example: fever, body ache, headache for two days"
            )
            query = st.text_area(
                query_label,
                height=170,
                placeholder=query_placeholder,
                key="assistant_query",
            )
            st.caption("Plain language is best. Short sentences work well.")
            submitted = st.form_submit_button(
                "Get gentle guidance" if mode == "Guided help (recommended)" else "See medicine ideas",
                use_container_width=True,
            )

        if submitted:
            if not query.strip():
                st.warning("Please tell MEDIBOT what is happening first.")
            elif mode == "Guided help (recommended)" and unified_bundle is not None:
                with st.spinner("Looking for the gentlest useful path..."):
                    handle_unified_submission(
                        unified_bundle,
                        name=name.strip(),
                        age=int(age),
                        gender=gender,
                        query=query.strip(),
                    )
            else:
                with st.spinner("Looking through common medicine matches..."):
                    handle_prediction_submission(
                        bundle,
                        name=name.strip(),
                        age=int(age),
                        gender=gender,
                        symptoms=query.strip(),
                    )

        if mode == "Guided help (recommended)" and unified_bundle is not None:
            result = st.session_state.get("last_assistant_result")
            if result:
                st.divider()
                if result.get("blocked", False):
                    render_warning_messages(result["warnings"])
                    st.error(
                        "MEDIBOT stopped here because your words matched urgent warning signs. Please contact urgent care or emergency services instead of relying on this app."
                    )
                elif result["route_kind"] == "drug":
                    st.info("MEDIBOT thinks this question is best handled as a medicine-style symptom check.")
                    render_drug_result_panel(
                        result,
                        unified_bundle,
                        model_accuracy=unified_bundle["metrics"]["accuracy"],
                        model_label="Guided model accuracy",
                        feedback_form_key="assistant_hub_unified_feedback",
                        feedback_bundle=bundle,
                    )
                else:
                    render_warning_messages(result["warnings"])
                    st.subheader("Gentle care guide")
                    top = result["predictions"][0]
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Most relevant care guide</div>
                            <p class="result-drug">{top['name']}</p>
                            <div class="result-category">{confidence_phrase(top['confidence']).title()}</div>
                            <strong>Suggested care advice</strong>
                            <div>{top['response']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if len(result["predictions"]) > 1:
                        st.subheader("Other likely matches")
                        for alternative in result["predictions"][1:4]:
                            subtitle = (
                                alternative["category"]
                                if alternative["kind"] == "drug"
                                else f"{alternative['kind'].title()} response"
                            )
                            st.markdown(
                                f"""
                                <div class="mini-card" style="margin-bottom:0.75rem;">
                                    <div class="mini-title">{alternative['name']}</div>
                                    <div class="mini-subtitle">{subtitle}</div>
                                    <div><strong>{confidence_phrase(alternative['confidence'])}</strong></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        else:
            result = st.session_state.get("last_drug_result")
            if result and result.get("interaction_mode") == mode:
                st.divider()
                render_drug_result_panel(
                    result,
                    bundle,
                    model_accuracy=bundle["metrics"]["accuracy"],
                    model_label="Medicine model accuracy",
                    feedback_form_key="assistant_hub_drug_feedback",
                )

    with right:
        st.markdown(
            """
            <div class="plain-note">
                <strong>What MEDIBOT does gently</strong><br/><br/>
                1. Reads your words in plain language.<br/>
                2. Looks for common care advice or medicine-style matches.<br/>
                3. Shows a clear warning if your message sounds urgent.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="plain-note" style="margin-top:0.9rem;">
                <strong>Good things to include</strong><br/><br/>
                How long it has been happening, the main symptom, and anything that makes it worse or better.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Try examples like these")
        examples = [
            "burning acidity after spicy food",
            "itchy eyes and sneezing from dust",
            "What should I do for cuts?",
            "How do I treat a sore throat?",
            "What should I do for nasal congestion?",
        ]
        for example in examples:
            st.code(example, language=None)

        if unified_bundle is not None:
            with st.expander("Project owner: model details"):
                st.markdown(
                    f"""
                    - Guided model accuracy: `{unified_bundle['metrics']['accuracy']:.0%}`
                    - Medicine model accuracy: `{bundle['metrics']['accuracy']:.0%}`
                    - Guided model classes: `{unified_bundle['metrics']['class_count']}`
                    - Medicine classes: `{bundle['metrics']['class_count']}`
                    """
                )


def advisor_tab(bundle: dict[str, object]) -> None:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Symptom Intake")
        with st.form("medibot_form", clear_on_submit=False):
            name = st.text_input("Name (optional)")
            age = st.number_input("Age", min_value=1, max_value=110, value=28, step=1)
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male", "Non-binary", "Prefer not to say"],
                index=0,
            )
            symptoms = st.text_area(
                "Describe the symptoms",
                height=170,
                placeholder="Example: fever, body ache, headache for two days",
            )
            submitted = st.form_submit_button("Analyze Symptoms", use_container_width=True)

        if submitted:
            if not symptoms.strip():
                st.warning("Please enter at least one symptom before running MEDIBOT.")
            else:
                with st.spinner("Analyzing symptom text..."):
                    handle_prediction_submission(
                        bundle,
                        name=name.strip(),
                        age=int(age),
                        gender=gender,
                        symptoms=symptoms.strip(),
                    )

        result = st.session_state.get("last_result")
        if result:
            st.divider()
            st.subheader("MEDIBOT Output")

            render_warning_messages(result["warnings"])

            if result["blocked"]:
                st.error(
                    "MEDIBOT stopped before suggesting a medicine because the symptoms matched urgent-care warning signs."
                )
            else:
                render_result_card(result)

                metric_one, metric_two, metric_three = st.columns(3)
                top_prediction = result["predictions"][0]
                metrics = bundle["metrics"]
                metric_one.metric("Confidence", f"{top_prediction['confidence']:.0%}")
                metric_two.metric("Model Accuracy", f"{metrics['accuracy']:.0%}")
                metric_three.metric("Dataset Size", int(bundle["dataset_size"]))

                render_alternatives(result)

                st.download_button(
                    "Download PDF Report",
                    data=result["pdf_bytes"],
                    file_name=f"medibot-report-{result['timestamp'].replace(':', '-')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                with st.expander("Save confirmed outcome to improve the model"):
                    st.caption(
                        "Only save feedback if a clinician or pharmacist later confirmed the medicine choice."
                    )
                    with st.form("feedback_form"):
                        confirmed_drug = st.selectbox(
                            "Confirmed medicine",
                            options=sorted(bundle["label_metadata"].keys()),
                            index=sorted(bundle["label_metadata"].keys()).index(top_prediction["drug"]),
                        )
                        notes = st.text_input("Optional note")
                        feedback_submitted = st.form_submit_button("Save Learning Example")

                    if feedback_submitted:
                        append_feedback_entry(
                            {
                                "timestamp": utc_timestamp(),
                                "name": result["name"],
                                "age": result["age"],
                                "age_group": bucket_age(int(result["age"])),
                                "gender": result["gender"],
                                "symptoms": result["symptoms"],
                                "confirmed_drug": confirmed_drug,
                                "notes": notes,
                            }
                        )
                        st.success(
                            "Confirmed feedback saved. Use the sidebar retrain button when you want MEDIBOT to learn from it."
                        )

    with right:
        st.subheader("How This Version Works")
        st.markdown(
            """
            - Free-text symptom entry is converted into TF-IDF features.
            - A Multinomial Naive Bayes classifier predicts the most likely educational match.
            - The app shows confidence scores and 2 backup matches.
            - Urgent red-flag symptoms trigger a safety override instead of a drug suggestion.
            - Each session can generate a PDF report and optional confirmed feedback.
            """
        )

        st.subheader("Good Example Inputs")
        examples = [
            "burning acidity after spicy food",
            "dry cough and sore throat at night",
            "itchy eyes, sneezing, and runny nose from dust",
            "wet cough with mucus and chest congestion",
            "blocked nose and dry nasal passages",
        ]
        for example in examples:
            st.code(example, language=None)


def learning_tab() -> None:
    st.subheader("Recent Visits")
    st.caption("These entries are saved locally on this device to help the demo keep a simple history.")

    recent_sessions = load_recent_sessions(limit=10)
    if recent_sessions.empty:
        st.info("No visits have been saved on this device yet.")
    else:
        friendly_sessions = recent_sessions[
            ["timestamp", "symptoms", "predicted_drug", "predicted_category", "confidence", "blocked_for_safety"]
        ].rename(
            columns={
                "timestamp": "When",
                "symptoms": "What you asked",
                "predicted_drug": "What MEDIBOT showed",
                "predicted_category": "Type",
                "confidence": "Confidence",
                "blocked_for_safety": "Urgent warning",
            }
        )
        st.dataframe(friendly_sessions, use_container_width=True, hide_index=True)

    recent_feedback = load_recent_feedback(limit=10)
    with st.expander("Project owner: confirmed feedback queue"):
        if recent_feedback.empty:
            st.info("No confirmed feedback has been saved yet.")
        else:
            st.dataframe(recent_feedback, use_container_width=True, hide_index=True)

    with st.expander("Project owner: medicine dataset snapshot"):
        seed = dataset_snapshot()
        st.dataframe(
            seed[["age_group", "gender", "symptoms", "recommended_drug"]],
            use_container_width=True,
            hide_index=True,
        )


def unified_tab(unified_bundle: dict[str, object] | None) -> None:
    st.subheader("Unified Assistant")
    if not unified_bundle:
        st.info("The unified model needs both the symptom-drug dataset and `data/intents.json` to be present.")
        return

    st.caption(
        "This classifier is trained on both datasets together. It can route a query to either a drug-style recommendation or an intent-style first-aid response."
    )

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.form("unified_form", clear_on_submit=False):
            age = st.number_input("Age", min_value=1, max_value=110, value=28, step=1, key="unified_age")
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male", "Non-binary", "Prefer not to say"],
                index=0,
                key="unified_gender",
            )
            query = st.text_area(
                "Ask a medical-support question or describe symptoms",
                height=170,
                placeholder="Example: burning acidity after spicy food OR What should I do for cuts?",
            )
            submitted = st.form_submit_button("Run Unified MEDIBOT", use_container_width=True)

        if submitted:
            if not query.strip():
                st.warning("Please enter a query first.")
            else:
                warnings = analyze_symptoms(query.strip())
                for warning in warnings:
                    if warning["level"] == "urgent":
                        st.error(f"{warning['title']}: {warning['message']}")
                    else:
                        st.warning(f"{warning['title']}: {warning['message']}")

                predictions = predict_combined(
                    unified_bundle,
                    query.strip(),
                    age=int(age),
                    gender=gender,
                )
                top = predictions[0]
                if top["kind"] == "drug":
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Unified Match - Drug Guidance</div>
                            <p class="result-drug">{top['name']}</p>
                            <div class="result-category">{top['category']} - {top['confidence']:.0%} confidence</div>
                            <strong>Why it matched</strong>
                            <div>{top['advisor_note']}</div>
                            <div style="margin-top:0.85rem;"><strong>Seek care if</strong></div>
                            <div>{top['seek_care_if']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Unified Match - First Aid Intent</div>
                            <p class="result-drug">{top['name']}</p>
                            <div class="result-category">Intent confidence - {top['confidence']:.0%}</div>
                            <strong>Dataset response</strong>
                            <div>{top['response']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("Other likely matches")
                for alternative in predictions[1:4]:
                    label = alternative["name"]
                    subtitle = (
                        alternative["category"]
                        if alternative["kind"] == "drug"
                        else f"{alternative['kind'].title()} response"
                    )
                    st.markdown(
                        f"""
                        <div class="mini-card" style="margin-bottom:0.75rem;">
                            <div class="mini-title">{label}</div>
                            <div class="mini-subtitle">{subtitle}</div>
                            <div><strong>{alternative['confidence']:.0%}</strong> confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with right:
        metrics = unified_bundle["metrics"]
        st.subheader("Unified Model Summary")
        st.markdown(
            f"""
            - Combined training rows: `{unified_bundle['dataset_size']}`
            - Combined classes: `{metrics['class_count']}`
            - Validation accuracy: `{metrics['accuracy']:.0%}`
            - Macro F1: `{metrics['macro_f1']:.0%}`
            """
        )
        st.subheader("Example Queries")
        for example in [
            "burning acidity after spicy food",
            "itchy eyes and sneezing from dust",
            "What should I do for cuts?",
            "How do I treat a sore throat?",
            "What should I do for nasal congestion?",
        ]:
            st.code(example, language=None)


def assistant_tab(first_aid_bundle: dict[str, object] | None) -> None:
    st.subheader("Common Care Guides")
    if not first_aid_bundle:
        st.info("No `data/intents.json` file was found, so the first-aid assistant is unavailable.")
        return

    st.caption(
        "Use this page when you want simple care guidance for common issues like cuts, fever, sprains, sore throat, or congestion."
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        with st.form("intent_form", clear_on_submit=False):
            question = st.text_area(
                "Ask a simple care question",
                height=160,
                placeholder="Example: What should I do for a cut on my hand?",
            )
            st.caption("You can ask in plain language, like you would ask a friend or family member.")
            submitted = st.form_submit_button("Show care guidance", use_container_width=True)

        if submitted:
            if not question.strip():
                st.warning("Please enter a question first.")
            else:
                warnings = analyze_symptoms(question.strip())
                for warning in warnings:
                    if warning["level"] == "urgent":
                        st.error(f"{warning['title']}: {warning['message']}")
                    else:
                        st.warning(f"{warning['title']}: {warning['message']}")

                predictions = predict_intent(first_aid_bundle, question.strip())
                top = predictions[0]
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Most relevant care guide</div>
                        <p class="result-drug">{top['tag']}</p>
                        <div class="result-category">{confidence_phrase(top['confidence']).title()}</div>
                        <strong>Suggested care advice</strong>
                        <div>{top['response']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if len(predictions) > 1:
                    st.subheader("Other close care guides")
                    for alternative in predictions[1:]:
                        st.markdown(
                            f"""
                            <div class="mini-card" style="margin-bottom:0.75rem;">
                                <div class="mini-title">{alternative['tag']}</div>
                                <div><strong>{confidence_phrase(alternative['confidence'])}</strong></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with right:
        st.markdown(
            """
            <div class="plain-note">
                <strong>Best for</strong><br/><br/>
                Cuts, sore throat, cough, fever, sprains, congestion, and other common day-to-day care questions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("You can ask things like")
        examples = [
            "What should I do for cuts?",
            "How do I treat a sore throat?",
            "What to do for nasal congestion?",
            "How to cure fever?",
            "What should I do for sprains?",
        ]
        for example in examples:
            st.code(example, language=None)

        with st.expander("Project owner: care guide model details"):
            metrics = first_aid_bundle["metrics"]
            st.markdown(
                f"""
                - Dataset patterns: `{metrics['dataset_size']}`
                - Intent tags: `{metrics['class_count']}`
                - Validation accuracy: `{metrics['accuracy']:.0%}`
                - Macro F1: `{metrics['macro_f1']:.0%}`
                """
            )


def project_tab(bundle: dict[str, object]) -> None:
    metrics = bundle["metrics"]
    st.subheader("About MEDIBOT")
    st.markdown(
        """
        MEDIBOT is designed to be a gentle helper for everyday symptom questions and simple care guidance.

        It can:
        - suggest common medicine-style matches for mild symptom descriptions
        - answer common care questions like cuts, sore throat, sprains, or congestion
        - show safety warnings when a message sounds urgent
        - save a simple local history on this device
        """
    )

    st.subheader("Important limits")
    st.markdown(
        """
        - This is not a diagnosis tool.
        - It does not replace a doctor, nurse, pharmacist, or emergency service.
        - It works best for mild, common questions and short symptom descriptions.
        - The safety checks are simple keyword checks, so they are helpful but not a full medical triage system.
        """
    )

    with st.expander("Project owner: technical details"):
        st.markdown(
            f"""
            - Medicine model accuracy: `{metrics['accuracy']:.0%}`
            - Medicine model macro F1: `{metrics['macro_f1']:.0%}`
            - Medicine classes: `{metrics['class_count']}`
            - Confirmed feedback rows: `{bundle['feedback_rows']}`
            """
        )


def main() -> None:
    ensure_runtime_files()
    render_shell()
    bundle = load_model_bundle()
    first_aid_bundle = load_first_aid_bundle()
    unified_bundle = load_unified_bundle()
    render_sidebar(bundle)

    assistant, first_aid, learning, project = st.tabs(
        ["Get Help", "Common Care Guides", "Recent Visits", "About MEDIBOT"]
    )
    with assistant:
        assistant_hub_tab(bundle, unified_bundle)
    with first_aid:
        assistant_tab(first_aid_bundle)
    with learning:
        learning_tab()
    with project:
        project_tab(bundle)


if __name__ == "__main__":
    main()
