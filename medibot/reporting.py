from __future__ import annotations

from fpdf import FPDF


def _coerce_pdf_bytes(raw_output: object) -> bytes:
    if isinstance(raw_output, bytearray):
        return bytes(raw_output)
    if isinstance(raw_output, bytes):
        return raw_output
    if isinstance(raw_output, str):
        return raw_output.encode("latin-1", errors="replace")
    raise TypeError("Unsupported PDF output type")


def _write_block(pdf: FPDF, text: str, *, line_height: float) -> None:
    pdf.multi_cell(0, line_height, text, new_x="LMARGIN", new_y="NEXT")


def build_report_pdf(
    *,
    name: str,
    age: int,
    gender: str,
    symptoms: str,
    warnings: list[dict[str, str]],
    predictions: list[dict[str, object]],
    generated_at: str,
    blocked_for_safety: bool,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "MEDIBOT SESSION REPORT", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    _write_block(
        pdf,
        "Educational-use summary only. This report should not be used as a prescription, diagnosis, or emergency assessment.",
        line_height=6,
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "User Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    _write_block(
        pdf,
        f"Name: {name or 'Not provided'}\nAge: {age}\nGender: {gender}\nGenerated: {generated_at}",
        line_height=7,
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Symptoms Entered", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    _write_block(pdf, symptoms, line_height=7)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Safety Review", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    if warnings:
        for warning in warnings:
            _write_block(pdf, f"- {warning['title']}: {warning['message']}", line_height=7)
    else:
        _write_block(pdf, "- No high-priority keyword flags were detected by the simple safety layer.", line_height=7)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recommendation Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    if blocked_for_safety:
        _write_block(
            pdf,
            "MEDIBOT did not provide a drug recommendation because the symptom text matched urgent-care warning signs. Please seek clinician review.",
            line_height=7,
        )
    else:
        top = predictions[0]
        _write_block(
            pdf,
            (
                f"Primary educational match: {top['drug']}\n"
                f"Category: {top['category']}\n"
                f"Confidence: {top['confidence']:.0%}\n"
                f"Why it matched: {top['advisor_note']}\n"
                f"Seek care if: {top['seek_care_if']}"
            ),
            line_height=7,
        )
        if len(predictions) > 1:
            pdf.ln(2)
            _write_block(pdf, "Other likely matches:", line_height=7)
            for alternative in predictions[1:]:
                _write_block(
                    pdf,
                    f"- {alternative['drug']} ({alternative['confidence']:.0%}) - {alternative['category']}",
                    line_height=7,
                )

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 10)
    _write_block(
        pdf,
        "Always confirm medicine choices with a clinician or pharmacist, especially for children, pregnancy, chronic conditions, allergies, or if symptoms persist.",
        line_height=6,
    )

    return _coerce_pdf_bytes(pdf.output())
