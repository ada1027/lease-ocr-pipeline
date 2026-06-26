# Lease Document OCR Extraction Pipeline

Automated pipeline to ingest lease PDFs, extract candidate square footage text, and write results to a CSV enrichment table.

## What it does

1. **Ingest** — opens a PDF, validates it is 2–200 pages
2. **Classify** — marks each page as `searchable` (has real text) or `scanned` (image-only)
3. **Extract** — pulls text from searchable pages matching GLA / sq ft / SF keywords
4. **OCR** — runs Tesseract on scanned pages to recover text from images
5. **Output** — appends one row per PDF to `data/output/enrichment_table.csv`

## Setup

### System dependencies

```bash
brew install tesseract
brew install --cask temurin   # Java — required by opendataloader-pdf
```

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY when ready for the AI step
```

## Usage

```bash
python pipeline.py "data/your_lease.pdf" STORE-001
```

Results are appended to `data/output/enrichment_table.csv`.

## Running tests

```bash
pytest tests/ -v
```

Drop any `.pdf` into `data/` and the integration tests pick it up automatically.

## Project structure

```
pipeline.py                  Entry point
src/
  ingestion/loader.py        PDF loading, page count validation, searchable/scanned classification
  extraction/searchable.py   Keyword-filtered text extraction from searchable pages
  extraction/ocr.py          Tesseract OCR for scanned pages
  ai/extractor.py            Claude text extraction (requires ANTHROPIC_API_KEY)
  ai/vision.py               Claude Vision fallback for diagrams/floorplans
  persistence/enrichment.py  CSV write for AI-extracted structured results
  utils/logging.py           Shared logger
tests/
  test_loader.py             Unit tests (no PDF required)
  test_real_pdfs.py          Integration tests against PDFs in data/
data/                        Input PDFs
data/output/                 enrichment_table.csv (git-ignored)
```

## What's next

- Wire in `src/ai/extractor.py` once `ANTHROPIC_API_KEY` is available — Claude reads the candidate text and returns `{square_footage, unit, confidence, evidence_snippet}`
- Activate `src/ai/vision.py` for low-confidence results where square footage is shown in a diagram
