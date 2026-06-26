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


