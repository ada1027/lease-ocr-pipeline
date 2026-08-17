"""Generate a PDF benchmark report from all benchmark CSV results."""

import csv
import glob
from collections import defaultdict
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

# ── Colours ────────────────────────────────────────────────────────────────
TEAL      = colors.HexColor("#0B9E93")
NAVY      = colors.HexColor("#0C1A2E")
GREEN     = colors.HexColor("#177A42")
AMBER     = colors.HexColor("#C97A0A")
RED       = colors.HexColor("#C0392B")
SLATE     = colors.HexColor("#364F68")
LIGHT_BG  = colors.HexColor("#F3F6FA")
MID_GREY  = colors.HexColor("#6C849A")
BORDER    = colors.HexColor("#CDD7E6")
WHITE     = colors.white
GREEN_BG  = colors.HexColor("#E6F4EC")
AMBER_BG  = colors.HexColor("#FDF3E3")
RED_BG    = colors.HexColor("#FDECEA")
TEAL_BG   = colors.HexColor("#E6F6F5")
HEADER_BG = colors.HexColor("#1A2E45")


# ── Styles ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

COVER_TITLE = S("CoverTitle", fontName="Helvetica-Bold", fontSize=28,
                leading=34, textColor=WHITE, spaceAfter=8)
COVER_SUB   = S("CoverSub",   fontName="Helvetica",      fontSize=14,
                leading=20, textColor=colors.HexColor("#A0C4D8"), spaceAfter=4)
COVER_META  = S("CoverMeta",  fontName="Helvetica",      fontSize=10,
                leading=14, textColor=colors.HexColor("#7AABB8"))

H1    = S("H1",    fontName="Helvetica-Bold", fontSize=16, leading=20,
          textColor=NAVY,  spaceBefore=18, spaceAfter=6)
H2    = S("H2",    fontName="Helvetica-Bold", fontSize=12, leading=16,
          textColor=TEAL,  spaceBefore=14, spaceAfter=4)
BODY  = S("Body",  fontName="Helvetica",      fontSize=9,  leading=14,
          textColor=SLATE, spaceAfter=6)
SMALL = S("Small", fontName="Helvetica",      fontSize=8,  leading=12,
          textColor=MID_GREY, spaceAfter=4)
LABEL = S("Label", fontName="Helvetica-Bold", fontSize=8,  leading=11,
          textColor=TEAL, spaceBefore=10, spaceAfter=2)
FIND  = S("Find",  fontName="Helvetica",      fontSize=9,  leading=13,
          textColor=SLATE)
FIND_TITLE = S("FindTitle", fontName="Helvetica-Bold", fontSize=9, leading=13,
               textColor=NAVY)

EYEBROW = S("Eyebrow", fontName="Helvetica-Bold", fontSize=8, leading=10,
            textColor=TEAL, spaceBefore=4, spaceAfter=2, letterSpacing=1)


def rule(color=BORDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=8, spaceBefore=4)


# ── Data ────────────────────────────────────────────────────────────────────
def load_data():
    all_rows = []
    for f in sorted(glob.glob("benchmark/results/*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                all_rows.append(r)

    stats = defaultdict(lambda: {"ext": 0, "high": 0, "total": 0,
                                  "cost": 0.0, "lat": []})
    for r in all_rows:
        if r.get("prompt_style") not in ("relaxed", ""):
            continue
        if r.get("reasoning", "False") not in ("False", "off"):
            continue
        if r.get("temperature", "0") not in ("0", "0.0"):
            continue
        key = r["model_name"]
        stats[key]["total"] += 1
        if r["square_footage"] and r["square_footage"] not in ("None", ""):
            stats[key]["ext"] += 1
        if r["confidence"] == "high":
            stats[key]["high"] += 1
        stats[key]["cost"] += float(r["cost_usd"] or 0)
        stats[key]["lat"].append(float(r["latency_sec"] or 0))

    total_cost  = sum(float(r["cost_usd"] or 0) for r in all_rows)
    total_calls = len(all_rows)
    runs        = len(set(r.get("timestamp", "")[:13] for r in all_rows))
    return stats, total_cost, total_calls


# ── Table helpers ───────────────────────────────────────────────────────────
def cell(text, style=None, bold=False, color=SLATE, size=9, align="LEFT"):
    s = ParagraphStyle("tc", fontName="Helvetica-Bold" if bold else "Helvetica",
                       fontSize=size, leading=12, textColor=color,
                       alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[align])
    return Paragraph(str(text), s)


def acc_color(ext, total):
    pct = ext / total if total else 0
    if pct >= 13/15 - 0.01:
        return GREEN, GREEN_BG
    elif pct >= 12/15 - 0.01:
        return AMBER, AMBER_BG
    elif pct > 0:
        return RED, RED_BG
    return MID_GREY, LIGHT_BG


MODEL_FAMILIES = {
    "Claude Sonnet 4.5": "Anthropic",
    "GPT-4o": "OpenAI", "GPT-4o Mini": "OpenAI",
    "GPT-4.1 Mini": "OpenAI", "GPT-4.1 Nano": "OpenAI", "o4-mini": "OpenAI",
    "DeepSeek Chat": "DeepSeek", "DeepSeek R1": "DeepSeek",
    "DeepSeek V3": "DeepSeek",
    "Llama 3.3 70B": "Meta", "Llama 4 Scout": "Meta",
    "Llama 4 Maverick": "Meta",
    "Mistral Large": "Mistral", "Mistral Nemo": "Mistral",
    "Qwen 2.5 72B": "Alibaba", "Qwen 2.5 VL 72B": "Alibaba",
    "Qwen 3 235B": "Alibaba", "Qwen 3 30B": "Alibaba",
    "Amazon Nova Lite": "Amazon", "Amazon Nova Pro": "Amazon",
    "Gemini 2.5 Flash": "Google", "Gemini 2.5 Flash Lite": "Google",
    "Gemini 2.5 Pro": "Google", "Gemma 3 12B": "Google",
    "Gemma 3 27B": "Google", "Gemma 3n E4B": "Google",
    "Kimi K2": "Moonshot", "MiniMax": "MiniMax",
    "Cohere Command R7B": "Cohere", "Phi-4": "Microsoft",
}

NEW_MODELS = {
    "Gemini 2.5 Flash", "Gemini 2.5 Flash Lite", "Gemini 2.5 Pro",
    "Gemma 3 12B", "Gemma 3 27B", "Gemma 3n E4B",
    "Phi-4", "Llama 4 Scout", "Llama 4 Maverick",
    "Cohere Command R7B", "GPT-4.1 Mini", "GPT-4.1 Nano", "o4-mini",
}


# ── Page template ───────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Footer
    canvas.setFillColor(BORDER)
    canvas.rect(0, 0, letter[0], 28, fill=1, stroke=0)
    canvas.setFillColor(MID_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(inch * 0.75, 10, "ChainXY · Sitewise Analytics · Lease OCR Pipeline — Benchmark Report · Confidential")
    canvas.drawRightString(letter[0] - inch * 0.75, 10, f"Page {doc.page}")
    canvas.restoreState()


def on_cover(canvas, doc):
    # Full-bleed dark header background
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    # Teal accent bar at top
    canvas.setFillColor(TEAL)
    canvas.rect(0, letter[1] - 6, letter[0], 6, fill=1, stroke=0)
    # Footer
    canvas.setFillColor(colors.HexColor("#0A1420"))
    canvas.rect(0, 0, letter[0], 40, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#3D5166"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(inch * 0.75, 14, "ChainXY · Sitewise Analytics · Internal Report · Confidential")
    canvas.drawRightString(letter[0] - inch * 0.75, 14, f"Generated {date.today().strftime('%B %d, %Y')}")
    canvas.restoreState()


# ── Build ────────────────────────────────────────────────────────────────────
def build_report(output_path="benchmark/lease_ocr_benchmark_report.pdf"):
    stats, total_cost, total_calls = load_data()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.6 * inch,
    )

    story = []

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * inch))
    story.append(Paragraph("LEASE OCR EXTRACTION PIPELINE", COVER_META))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Model Benchmark Report", COVER_TITLE))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "A comprehensive record of AI model testing for automated square footage "
        "extraction from commercial real estate documents.", COVER_SUB))
    story.append(Spacer(1, 0.5 * inch))
    story.append(rule(colors.HexColor("#1E3450"), 1))
    story.append(Spacer(1, 0.2 * inch))

    meta_data = [
        ["Author", "Ada Huang"],
        ["Organisation", "ChainXY · Sitewise Analytics"],
        ["Date", date.today().strftime("%B %d, %Y")],
        ["API Platform", "OpenRouter"],
        ["Total API Calls", f"{total_calls:,}"],
        ["Total Spend", f"${total_cost:.2f}"],
        ["Models Tested", str(len(stats))],
    ]
    meta_table = Table(meta_data, colWidths=[1.5 * inch, 4 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",  (0, 0), (0, -1), TEAL),
        ("TEXTCOLOR",  (1, 0), (1, -1), colors.HexColor("#A0C4D8")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LINEBEFORE",  (0, 0), (0, -1), 2, TEAL),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
        ("LEFTPADDING", (1, 0), (1, -1), 12),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ── PAGE 2: OVERVIEW ────────────────────────────────────────────────────
    story.append(Paragraph("OVERVIEW", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Project Summary", H1))
    story.append(Paragraph(
        "This report documents the complete benchmarking process for an automated "
        "lease document OCR pipeline built for ChainXY. The pipeline ingests "
        "commercial real estate PDFs — including formal leases, marketing flyers, "
        "site plans, and tenant rosters — and extracts the primary square footage "
        "figure using AI language models via the OpenRouter API.",
        BODY))
    story.append(Paragraph(
        "The testing covered 25 different AI models across multiple prompt styles, "
        "effort levels, extraction modes, temperatures, and reasoning configurations, "
        f"totalling {total_calls:,} API calls and ${total_cost:.2f} in spend.",
        BODY))

    story.append(Spacer(1, 0.15 * inch))

    # Stat boxes
    stats_data = [
        [cell("25", bold=True, color=TEAL, size=22, align="CENTER"),
         cell("1,015", bold=True, color=TEAL, size=22, align="CENTER"),
         cell(f"${total_cost:.2f}", bold=True, color=TEAL, size=22, align="CENTER"),
         cell("203", bold=True, color=TEAL, size=22, align="CENTER")],
        [cell("Models Tested", color=MID_GREY, size=8, align="CENTER"),
         cell("Total API Calls", color=MID_GREY, size=8, align="CENTER"),
         cell("Total Spend", color=MID_GREY, size=8, align="CENTER"),
         cell("PDFs Processed", color=MID_GREY, size=8, align="CENTER")],
    ]
    stat_t = Table(stats_data, colWidths=[1.55 * inch] * 4)
    stat_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",           (0, 0), (0, -1), 0.5, BORDER),
        ("BOX",           (1, 0), (1, -1), 0.5, BORDER),
        ("BOX",           (2, 0), (2, -1), 0.5, BORDER),
        ("BOX",           (3, 0), (3, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(stat_t)

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Pipeline Architecture", H2))
    story.append(Paragraph(
        "The pipeline processes each PDF through the following stages:", BODY))

    pipeline_steps = [
        ("1. PDF Ingest", "PyMuPDF (fitz) opens the file, validates page count (minimum 1 page), and prepares it for processing."),
        ("2. Page Classification", "Each page is classified as searchable (extractable text) or scanned (image-only) using a 100-character non-whitespace threshold."),
        ("3. Text / OCR Extraction", "Searchable pages are read directly. Scanned pages are rasterised at 200 DPI and processed through Tesseract OCR. Pages are filtered by keywords (GLA, sq ft, demised premises, etc.) to find candidates."),
        ("4. AI Extraction", "Candidate text is sent to an LLM via OpenRouter with a structured prompt requesting square footage, unit, confidence, and evidence snippet as raw JSON."),
        ("5. Output", "Results are appended to an enrichment CSV (store_id, filename, square_footage, unit, confidence, evidence_snippet)."),
    ]
    for title, desc in pipeline_steps:
        story.append(Paragraph(
            f'<b><font color="#0B9E93">{title}</font></b> — {desc}', BODY))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Production Run Results", H2))
    story.append(Paragraph(
        "A full batch was run on 203 PDFs from the SiteFiles dataset using "
        "Claude Sonnet 4.5 with the relaxed prompt:", BODY))

    prod_data = [
        [cell("Total PDFs", bold=True, size=9), cell("Successful", bold=True, size=9),
         cell("Skipped", bold=True, size=9), cell("Failed", bold=True, size=9)],
        [cell("203", bold=True, color=TEAL, size=14, align="CENTER"),
         cell("176", bold=True, color=GREEN, size=14, align="CENTER"),
         cell("27", bold=True, color=AMBER, size=14, align="CENTER"),
         cell("0", bold=True, color=MID_GREY, size=14, align="CENTER")],
        [cell("100%", color=MID_GREY, size=8, align="CENTER"),
         cell("87%", color=GREEN, size=8, align="CENTER"),
         cell("13% (no candidate text)", color=AMBER, size=8, align="CENTER"),
         cell("0%", color=MID_GREY, size=8, align="CENTER")],
    ]
    prod_t = Table(prod_data, colWidths=[1.55 * inch] * 4)
    prod_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("BACKGROUND",    (0, 1), (-1, -1), LIGHT_BG),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(prod_t)

    story.append(PageBreak())

    # ── PAGE 3: METHODOLOGY ─────────────────────────────────────────────────
    story.append(Paragraph("TESTING METHODOLOGY", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Variables Under Test", H1))
    story.append(Paragraph(
        "Six independent variables were tested in combination. Every run logs "
        "the model, prompt, effort level, extraction mode, temperature, reasoning "
        "flag, extracted value, confidence, cost, and latency into a timestamped CSV.",
        BODY))

    var_rows = [
        ["Variable", "Options Tested", "Finding"],
        ["Model", "25 models across 10+ families", "Biggest impact on accuracy and cost"],
        ["Prompt Style", "relaxed · strict · few-shot", "Strict = near-zero on marketing docs; relaxed = best"],
        ["Effort Level", "first2 · half · full", "first2 matches full accuracy on this dataset"],
        ["Extraction Mode", "text · vision", "Text sufficient; vision untested at full scale"],
        ["Temperature", "0.0 · 0.3", "No meaningful difference found"],
        ["Reasoning", "off · on", "Same or worse accuracy, 2-3x slower and costlier"],
    ]
    var_t = Table(var_rows, colWidths=[1.4 * inch, 2.0 * inch, 2.9 * inch])
    var_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR",     (0, 1), (0, -1), TEAL),
        ("TEXTCOLOR",     (1, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(var_t)

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Prompt Styles", H2))

    prompts = [
        ("Relaxed (production)", "relaxed",
         'Accept any square footage figure: total center size, GLA, demised premises, suite size, or building size. '
         'Marketing flyers and tenant rosters are valid sources. Return highest/most prominent figure.'),
        ("Strict (lease-only)", "strict",
         'Only extract from formal lease clauses: Demised Premises, Premises, GLA, Rentable Area. '
         'Do NOT use marketing text, site plans, or tenant rosters. '
         'Result: near-zero extraction rate on this dataset (documents are mostly flyers).'),
        ("Few-shot (examples)", "few_shot",
         'Includes three worked examples before the task. '
         'Slight accuracy benefit over relaxed on some models, but adds token cost. '
         'No net advantage found in this dataset.'),
    ]

    for title, tag, desc in prompts:
        story.append(Paragraph(
            f'<b><font color="#0C1A2E">{title}</font></b>  '
            f'<font color="#0B9E93" size="8">[{tag}]</font>', BODY))
        story.append(Paragraph(desc, SMALL))

    story.append(PageBreak())

    # ── PAGE 4+: FULL RESULTS TABLE ─────────────────────────────────────────
    story.append(Paragraph("BENCHMARK RESULTS", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Full Model Comparison", H1))
    story.append(Paragraph(
        "All models tested with the relaxed prompt, temperature 0, no reasoning, "
        "effort level: first2 pages. Sorted by accuracy then cost. "
        "Models marked with * have fewer than 15 runs — data is preliminary.",
        BODY))

    sorted_models = sorted(stats.items(),
                           key=lambda x: (-x[1]["ext"]/max(x[1]["total"],1),
                                          x[1]["cost"]))

    header = [
        cell("Model", bold=True, color=WHITE, size=8.5),
        cell("Family", bold=True, color=WHITE, size=8.5),
        cell("Extracted", bold=True, color=WHITE, size=8.5, align="CENTER"),
        cell("High Conf", bold=True, color=WHITE, size=8.5, align="CENTER"),
        cell("Cost / 15 PDFs", bold=True, color=WHITE, size=8.5, align="RIGHT"),
        cell("Avg Speed", bold=True, color=WHITE, size=8.5, align="CENTER"),
        cell("Runs", bold=True, color=WHITE, size=8.5, align="CENTER"),
    ]
    rows = [header]
    row_styles = []
    row_idx = 1

    for model_name, v in sorted_models:
        n    = v["total"]
        ext  = v["ext"]
        high = v["high"]
        cost = v["cost"]
        avg_lat = sum(v["lat"]) / len(v["lat"]) if v["lat"] else 0
        preliminary = n < 15
        fg, bg = acc_color(ext, n)

        display_name = model_name + (" *" if preliminary else "")
        cost_per_15 = (cost / n * 15) if n > 0 else cost

        rows.append([
            cell(display_name, bold=(model_name in NEW_MODELS), color=NAVY, size=8.5),
            cell(MODEL_FAMILIES.get(model_name, "—"), color=MID_GREY, size=8),
            cell(f"{ext} / {n}", bold=True, color=fg, size=8.5, align="CENTER"),
            cell(f"{high} / {n}", color=fg, size=8.5, align="CENTER"),
            cell(f"${cost_per_15:.5f}", color=SLATE, size=8, align="RIGHT"),
            cell(f"{avg_lat:.1f}s", color=SLATE, size=8, align="CENTER"),
            cell(str(n), color=MID_GREY, size=8, align="CENTER"),
        ])
        row_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                           bg if row_idx % 2 == 0 else WHITE))
        row_idx += 1

    col_w = [2.0*inch, 0.9*inch, 0.8*inch, 0.8*inch, 1.1*inch, 0.8*inch, 0.5*inch]
    results_t = Table(rows, colWidths=col_w, repeatRows=1)

    ts = [
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
    ]
    # Highlight top performers (13+ out of 15)
    for i, (mn, v) in enumerate(sorted_models, 1):
        n = v["total"]
        if n >= 14 and v["ext"] / n >= 13/15 - 0.01:
            ts.append(("BACKGROUND", (0, i), (1, i), TEAL_BG))

    results_t.setStyle(TableStyle(ts))
    story.append(results_t)

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(
        "* Fewer than 15 PDFs tested — preliminary result.  "
        "Cost / 15 PDFs is extrapolated for models with more or fewer runs.",
        SMALL))

    story.append(PageBreak())

    # ── PAGE: PROMPT & REASONING COMPARISON ─────────────────────────────────
    story.append(Paragraph("VARIABLE ANALYSIS", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Prompt Style Impact", H1))
    story.append(Paragraph(
        "Tested on Claude Sonnet 4.5, 15 PDFs, temperature 0, no reasoning:", BODY))

    prompt_cmp = [
        ["Prompt Style", "Extracted", "High Conf", "Notes"],
        ["relaxed", "13 / 15", "13 / 15", "Best — accepts any SF figure"],
        ["few_shot", "13 / 15", "13 / 15", "Same as relaxed but higher token cost"],
        ["strict", "1 / 15", "1 / 15", "Only formal lease clauses — useless on flyers"],
    ]
    pt = Table(prompt_cmp, colWidths=[1.3*inch, 1.0*inch, 1.0*inch, 2.9*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 3), (-1, 3), RED_BG),
    ]))
    story.append(pt)

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Reasoning Mode Impact", H1))
    story.append(Paragraph(
        "Claude extended thinking vs standard, relaxed prompt, temperature 0, 15 PDFs:", BODY))

    reason_cmp = [
        ["Mode", "Extracted", "Avg Cost / PDF", "Avg Latency", "Verdict"],
        ["off (standard)", "13 / 15", "$0.0044", "2.0s", "Use this"],
        ["on (extended thinking)", "13 / 15", "$0.0079", "5.3s", "Same accuracy, 1.8x cost, 2.7x slower"],
    ]
    rt = Table(reason_cmp, colWidths=[1.3*inch, 0.9*inch, 1.1*inch, 0.9*inch, 2.0*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, AMBER_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(rt)

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("DeepSeek Reasoning (R1) Impact", H2))
    reason_ds = [
        ["Mode", "Extracted", "Avg Latency", "Verdict"],
        ["off (deepseek-chat)", "13 / 15", "2.6s", "Best — accurate and fast"],
        ["on (deepseek-r1 swap)", "8 / 15", "30-100s", "Significantly worse, very slow"],
    ]
    rdt = Table(reason_ds, colWidths=[2.0*inch, 1.0*inch, 1.1*inch, 2.1*inch])
    rdt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, RED_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(rdt)

    story.append(PageBreak())

    # ── PAGE: KEY FINDINGS ───────────────────────────────────────────────────
    story.append(Paragraph("KEY FINDINGS", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Summary of Results", H1))

    findings = [
        ("green", "Mistral Nemo is the accuracy leader",
         "14-15/15 PDFs extracted correctly at $0.00007/call average — a tiny, "
         "cheap model that outperforms Claude Sonnet 4.5 on this specific task. "
         "Recommended for production if this accuracy holds on a larger sample."),
        ("green", "Gemma 3 12B is the best cost-accuracy tradeoff",
         "13/15 accuracy at $0.00003/call — 135x cheaper than Claude Sonnet 4.5 "
         "($0.0045/call) for the same extraction rate. Strong candidate for "
         "high-volume production use."),
        ("green", "Gemini 2.5 Flash Lite is the fastest and nearly free",
         "12-13/15 accuracy, 0.6s average latency, $0.00007/call. Best option "
         "if throughput and speed are the priority."),
        ("amber", "Relaxed prompt is mandatory for this dataset",
         "The strict prompt (formal lease clauses only) produces near-zero "
         "extraction on marketing flyers and site plans — which make up the "
         "majority of the dataset. Always use the relaxed prompt."),
        ("amber", "Reasoning modes add cost with no accuracy benefit",
         "Extended thinking (Claude) and DeepSeek R1 produce identical or worse "
         "accuracy at 2-3x the cost and latency. Disable reasoning for this task."),
        ("amber", "Temperature has zero effect",
         "T=0.0 and T=0.3 produced identical results across all models. Lock at 0."),
        ("red", "Gemini 2.5 Pro is not suitable for this task",
         "0/15 extraction rate across a full 15-PDF run ($0.08 total). "
         "Consistently returns null with low confidence. The model may need "
         "a very different prompt format or more pages to perform."),
        ("red", "Baidu ERNIE not available on OpenRouter",
         "All Baidu model IDs return 400 errors on this platform. Would require "
         "a direct integration with the Baidu Qianfan API — a separate project."),
        ("blue", "First-2-pages effort matches full-document accuracy",
         "The effort level (how much text to send) had no impact on accuracy "
         "for this dataset — square footage figures appear early in documents. "
         "Use first2 in production to minimise token cost."),
        ("blue", "Vision extraction implemented but not fully benchmarked",
         "Claude Vision and other vision-capable models can process rasterised "
         "page images. Full vision benchmarking across 15+ PDFs is pending."),
    ]

    colors_map = {"green": (GREEN, GREEN_BG), "amber": (AMBER, AMBER_BG),
                  "red": (RED, RED_BG), "blue": (TEAL, TEAL_BG)}

    for colour, title, body in findings:
        fg, bg = colors_map[colour]
        find_data = [[
            Paragraph(f'<b>{title}</b><br/>{body}',
                      ParagraphStyle("fd", fontName="Helvetica", fontSize=9,
                                     leading=13, textColor=SLATE,
                                     spaceAfter=0))
        ]]
        ft = Table(find_data, colWidths=[6.25 * inch])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("LINEBEFORE",    (0, 0), (0, -1), 3, fg),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.06 * inch))

    story.append(PageBreak())

    # ── PAGE: RECOMMENDATIONS ────────────────────────────────────────────────
    story.append(Paragraph("RECOMMENDATIONS", EYEBROW))
    story.append(rule())
    story.append(Paragraph("Next Steps", H1))

    rec_data = [
        ["Priority", "Action", "Rationale"],
        ["High", "Run Mistral Nemo on 50+ PDFs",
         "Confirm 14/15 result holds at scale before recommending as production model"],
        ["High", "Run Gemma 3 12B on 50+ PDFs",
         "Validate 13/15 result — already cheapest model at that accuracy tier"],
        ["Medium", "Full vision benchmark",
         "Test vision extraction on the scanned-heavy PDFs where Tesseract OCR struggles"],
        ["Medium", "LlamaParse integration",
         "Dedicated PDF parser (cloud.llamaindex.ai) — separate API key required. "
         "Strong on scanned docs and complex layouts"],
        ["Low", "Connect Tier 1 + Tier 3 outputs",
         "Merge AI-extracted SF figures with Overture Maps building footprint "
         "data into a single enrichment table"],
        ["Low", "Baidu ERNIE direct integration",
         "Would require Baidu Qianfan API key — separate project outside OpenRouter"],
    ]
    recs_t = Table(rec_data, colWidths=[0.8*inch, 1.8*inch, 3.7*inch])
    recs_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTNAME",      (0, 1), (1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), WHITE),
        ("BACKGROUND",    (0, 1), (0, 3), colors.HexColor("#C0392B")),
        ("BACKGROUND",    (0, 4), (0, 5), colors.HexColor("#C97A0A")),
        ("BACKGROUND",    (0, 6), (0, -1), colors.HexColor("#6C849A")),
        ("TEXTCOLOR",     (1, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
    ]))
    story.append(recs_t)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Cost Projection", H2))
    story.append(Paragraph(
        "Based on benchmark results, estimated cost to run extraction on 1,000 PDFs:", BODY))

    cost_proj = [
        ["Model", "Cost / 1,000 PDFs", "Accuracy (est.)", "Notes"],
        ["Mistral Nemo", "$0.006", "~93%", "New leader — needs validation"],
        ["Gemma 3 12B", "$0.030", "~87%", "Best validated cheap option"],
        ["Gemini 2.5 Flash Lite", "$0.007", "~80%", "Fastest, nearly free"],
        ["GPT-4o Mini", "$0.161", "~87%", "Previous benchmark — well established"],
        ["Claude Sonnet 4.5", "$4.50", "~87%", "Original model — gold standard"],
    ]
    cpt = Table(cost_proj, colWidths=[1.7*inch, 1.3*inch, 1.2*inch, 2.1*inch])
    cpt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREEN_BG, GREEN_BG, WHITE,
                                              LIGHT_BG, AMBER_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(cpt)

    # ── BUILD ────────────────────────────────────────────────────────────────
    def first_page(canvas, doc):
        on_cover(canvas, doc)

    def later_pages(canvas, doc):
        on_page(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    build_report()
