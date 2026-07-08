"""
pdf_export.py
-------------
Generates a professional single-analysis PDF report using ReportLab,
including the uploaded image, extracted metrics, prediction, confidence,
recommendations and sustainability score.
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)

from utils.helpers import INSTITUTION_NAME, PROJECT_TITLE, REPORTS_DIR, now_filename_safe

GREEN = colors.HexColor("#1DB954")
DARKGRAY = colors.HexColor("#1E1E1E")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontSize=18,
            leading=22,
            textColor=DARKGRAY,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#555555"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            fontSize=13,
            leading=16,
            textColor=GREEN,
            spaceBefore=14,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        )
    )
    return styles


def generate_pdf_report(record: dict, image_path: str = None) -> str:
    """
    Build a PDF report for a single prediction record.

    Parameters
    ----------
    record : dict
        A row of prediction data (matching CSV_COLUMNS in helpers.py).
    image_path : str, optional
        Path to the (processed/segmented) image to embed in the report.

    Returns
    -------
    str : path to the generated PDF file
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = f"FoodWaste_Report_{now_filename_safe()}.pdf"
    output_path = os.path.join(REPORTS_DIR, filename)

    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    story = []

    # --- Header ---
    story.append(Paragraph(INSTITUTION_NAME, styles["ReportSubtitle"]))
    story.append(Paragraph(PROJECT_TITLE, styles["ReportTitle"]))
    story.append(Paragraph(f"Report generated on: {record.get('timestamp', '')}", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", color=GREEN, thickness=1.2))
    story.append(Spacer(1, 10))

    # --- Image ---
    if image_path and os.path.exists(image_path):
        story.append(Paragraph("Analyzed Plate Image", styles["SectionHeader"]))
        img = RLImage(image_path, width=8 * cm, height=8 * cm)
        story.append(img)
        story.append(Spacer(1, 8))

    # --- Prediction Summary ---
    story.append(Paragraph("Prediction Summary", styles["SectionHeader"]))
    summary_data = [
        ["Metric", "Value"],
        ["Predicted Category", record.get("predicted_category", "-")],
        ["Waste Percentage", f"{record.get('waste_percentage', 0):.2f} %"],
        ["Confidence Score", f"{record.get('confidence', 0):.2f} %"],
        ["Sustainability Score", f"{record.get('sustainability_score', 0)} / 100"],
        ["Food Coverage Ratio", f"{record.get('coverage_ratio', 0):.3f}"],
        ["Processing Time", f"{record.get('processing_time_ms', 0):.1f} ms"],
    ]
    table = Table(summary_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    # --- Feature Details ---
    story.append(Paragraph("Extracted Feature Details", styles["SectionHeader"]))
    feature_data = [
        ["Feature", "Value"],
        ["Food Area (px)", str(record.get("food_area", "-"))],
        ["Plate Area (px)", str(record.get("plate_area", "-"))],
        ["Contour Count", str(record.get("contour_count", "-"))],
        ["Average RGB", f"({record.get('avg_r', 0):.0f}, {record.get('avg_g', 0):.0f}, {record.get('avg_b', 0):.0f})"],
        ["Average HSV", f"({record.get('avg_h', 0):.0f}, {record.get('avg_s', 0):.0f}, {record.get('avg_v', 0):.0f})"],
    ]
    ftable = Table(feature_data, colWidths=[7 * cm, 7 * cm])
    ftable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARKGRAY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ftable)
    story.append(Spacer(1, 10))

    # --- Recommendations ---
    recommendations = record.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("AI-Generated Recommendations", styles["SectionHeader"]))
        for tip in recommendations:
            story.append(Paragraph(f"&bull; {tip}", styles["Normal"]))
        story.append(Spacer(1, 10))

    # --- Footer ---
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DDDDDD"), thickness=0.8))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "This report was generated automatically by the AI-Powered Canteen Food Waste "
            "Quantification System using classical computer vision (OpenCV) segmentation and a "
            "RandomForestClassifier (scikit-learn). Figures are estimates intended to support "
            "canteen waste-reduction decisions.",
            styles["ReportSubtitle"],
        )
    )

    doc.build(story)
    return output_path
