"""
CardioSense - Report PDF Builder
===================================
Renders a single-assessment risk report as a PDF:
  patient info -> vitals -> risk result -> SHAP contribution chart -> recommendations

Takes plain Python values (not SQLAlchemy objects directly) so it stays
testable without a DB connection.
"""

import io
import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)


def _build_shap_chart(ranked_shap: list[dict], top_n: int = 6) -> io.BytesIO:
    """
    ranked_shap: list of {"feature": str, "contribution": float},
    already sorted by absolute contribution (this is exactly the shape
    produced by shap_analysis.explain_prediction()["ranked"]).
    Returns a PNG image buffer, positive contributions in red (raises risk),
    negative in green (lowers risk).
    """
    top = ranked_shap[:top_n]
    features = [item["feature"] for item in top][::-1]
    values = [item["contribution"] for item in top][::-1]
    bar_colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(features, values, color=bar_colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Contribution to risk score")
    ax.set_title("Top factors influencing this prediction")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_report_pdf(
    *,
    patient: dict,
    assessment: dict,
    lab_result: dict,
    lifestyle: dict,
    prediction: dict,
    recommendations: list[dict],
) -> io.BytesIO:
    """
    Each argument is a plain dict of the fields the report needs:

    patient: name, patient_code, gender, date_of_birth
    assessment: id, assessment_date, height, weight, ap_hi, ap_lo,
                bmi, pulse_pressure, map, bp_category
    lab_result: cholesterol, gluc
    lifestyle: smoke, alco, active
    prediction: model_used, risk_probability, risk_level, confidence_score,
                shap_values (dict with a "ranked" key)
    recommendations: list of {recommendation_text, priority}

    Returns an in-memory PDF (BytesIO) — caller decides whether to stream
    it, save it to disk, or attach it to an email.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1a3c5e"),
    )
    story = []

    # --- Header ---
    story.append(Paragraph("CardioSense Risk Assessment Report", styles["Title"]))
    story.append(Paragraph(
        f"Generated: {assessment.get('assessment_date', '')}", styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # --- Patient info ---
    story.append(Paragraph("Patient Information", section_style))
    gender_label = "Male" if patient.get("gender") == 1 else "Female"
    patient_table = Table([
        ["Name", patient.get("name", "")],
        ["Patient Code", patient.get("patient_code", "")],
        ["Gender", gender_label],
        ["Date of Birth", str(patient.get("date_of_birth", ""))],
    ], colWidths=[1.8 * inch, 4.2 * inch])
    patient_table.setStyle(_info_table_style())
    story.append(patient_table)

    # --- Vitals ---
    story.append(Paragraph("Vitals & Measurements", section_style))
    bp_labels = {0: "Normal", 1: "Elevated", 2: "Hypertension Stage 1", 3: "Hypertension Stage 2"}
    chol_labels = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    vitals_table = Table([
        ["Height", f"{assessment.get('height')} cm", "Weight", f"{assessment.get('weight')} kg"],
        ["Blood Pressure", f"{assessment.get('ap_hi')}/{assessment.get('ap_lo')} mmHg",
         "BMI", f"{assessment.get('bmi')}"],
        ["Pulse Pressure", f"{assessment.get('pulse_pressure')} mmHg",
         "MAP", f"{assessment.get('map')} mmHg"],
        ["BP Category", bp_labels.get(assessment.get("bp_category"), "-"),
         "Cholesterol", chol_labels.get(lab_result.get("cholesterol"), "-")],
        ["Glucose", chol_labels.get(lab_result.get("gluc"), "-"),
         "Smoker", "Yes" if lifestyle.get("smoke") else "No"],
        ["Alcohol", "Yes" if lifestyle.get("alco") else "No",
         "Physically Active", "Yes" if lifestyle.get("active") else "No"],
    ], colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    vitals_table.setStyle(_info_table_style())
    story.append(vitals_table)

    # --- Risk result ---
    story.append(Paragraph("Risk Assessment Result", section_style))
    risk_level = prediction.get("risk_level", "unknown")
    risk_color = {"low": "#2ca02c", "moderate": "#d4a017", "high": "#d62728"}.get(
        risk_level, "#333333"
    )
    prob_pct = prediction.get("risk_probability", 0) * 100
    risk_style = ParagraphStyle(
        "Risk", parent=styles["Normal"], fontSize=14, textColor=colors.HexColor(risk_color),
        spaceAfter=4,
    )
    story.append(Paragraph(f"Risk Level: <b>{risk_level.upper()}</b>", risk_style))
    story.append(Paragraph(f"Risk Probability: {prob_pct:.1f}%", styles["Normal"]))
    if prediction.get("confidence_score") is not None:
        story.append(Paragraph(
            f"Model Confidence: {prediction['confidence_score'] * 100:.1f}%", styles["Normal"]
        ))
    story.append(Paragraph(f"Model Used: {prediction.get('model_used', '-')}", styles["Normal"]))

    # --- SHAP chart ---
    shap_values = prediction.get("shap_values") or {}
    ranked = shap_values.get("ranked")
    if ranked:
        chart_buf = _build_shap_chart(ranked)
        story.append(Spacer(1, 8))
        story.append(Image(chart_buf, width=6 * inch, height=3.2 * inch))

    # --- Recommendations ---
    story.append(Paragraph("Recommendations", section_style))
    if recommendations:
        priority_color = {"high": "#d62728", "medium": "#d4a017", "low": "#2ca02c"}
        for rec in recommendations:
            color = priority_color.get(rec.get("priority", "medium"), "#333333")
            story.append(Paragraph(
                f'<font color="{color}">&#9679;</font> '
                f'<b>[{rec.get("priority", "medium").upper()}]</b> '
                f'{rec.get("recommendation_text", "")}',
                styles["Normal"],
            ))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No specific recommendations generated.", styles["Normal"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "This report is generated by an automated risk-prediction model and is "
        "intended to support, not replace, clinical judgment.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buf.seek(0)
    return buf


def _info_table_style() -> TableStyle:
    return TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f4f8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])