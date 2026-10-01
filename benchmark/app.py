"""Gradio web UI for the vision extraction benchmark.

Security contract:
- OPENROUTER_API_KEY read from env server-side only; never in HTML/JS/logs shown to users
- All errors shown to users are short generic messages; full details go to app.log only
- PDF uploads validated via magic bytes, saved with UUID filenames, deleted after processing
- Folder browse locked to data/ subdirectory only
- share=False, localhost-only launch
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging — full details to file only, NOT to stderr (protect screen-sharers)
# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).parent / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
logger = logging.getLogger("benchmark.app")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

sys.path.insert(0, str(_REPO_ROOT))

from benchmark.vision_run import (  # noqa: E402
    VISION_MODELS,
    VisionModel,
    call_openrouter_native_pdf,
    call_openrouter_vision,
    get_openrouter_client,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = _REPO_ROOT / "data"
_PDF_MAGIC = b"%PDF"
_MAX_UPLOAD_MB = 50

# ---------------------------------------------------------------------------
# OpenRouter model fetcher (cached per session)
# ---------------------------------------------------------------------------
_or_models_cache: list[dict] | None = None


def _fetch_or_vision_models() -> list[dict]:
    global _or_models_cache
    if _or_models_cache is not None:
        return _or_models_cache
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set; skipping live model fetch")
        return []
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        vision = [
            m for m in models
            if "image" in (m.get("architecture") or {}).get("input_modalities", [])
        ]
        _or_models_cache = vision
        logger.info("Fetched %d vision models from OpenRouter", len(vision))
        return vision
    except Exception as exc:
        logger.error("OpenRouter model fetch failed: %s", exc)
        return []


def _or_model_choices() -> list[tuple[str, str]]:
    models = _fetch_or_vision_models()
    choices = []
    for m in sorted(models, key=lambda x: x.get("id", "")):
        mid = m.get("id", "")
        name = m.get("name") or mid
        pricing = m.get("pricing") or {}
        try:
            inp = float(pricing.get("prompt", 0)) * 1000
            out = float(pricing.get("completion", 0)) * 1000
            label = f"{name} — ${inp:.4f}/${ out:.4f} per 1k"
        except (TypeError, ValueError):
            label = name
        choices.append((label, mid))
    return choices


# ---------------------------------------------------------------------------
# PDF validation
# ---------------------------------------------------------------------------
def _validate_pdf_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PDF_MAGIC
    except OSError:
        return False


def _safe_temp_copy(upload_path: str) -> Path:
    src = Path(upload_path)
    if src.stat().st_size > _MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {_MAX_UPLOAD_MB} MB limit")
    if not _validate_pdf_magic(src):
        raise ValueError("File does not appear to be a valid PDF")
    tmp_dir = Path(tempfile.gettempdir()) / "lease_ocr_uploads"
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4()}.pdf"
    dest.write_bytes(src.read_bytes())
    return dest


# ---------------------------------------------------------------------------
# Data-directory browser (locked to data/)
# ---------------------------------------------------------------------------
def _list_data_pdfs() -> list[str]:
    if not _DATA_DIR.exists():
        return []
    pdfs = sorted(_DATA_DIR.rglob("*.pdf"))
    return [str(p.relative_to(_REPO_ROOT)) for p in pdfs]


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------
def _model_from_id(model_id: str) -> VisionModel:
    for m in VISION_MODELS:
        if m.model_id == model_id:
            return m
    live = {m["id"]: m for m in _fetch_or_vision_models()}
    entry = live.get(model_id, {})
    pricing = entry.get("pricing") or {}
    try:
        inp_per_1k = float(pricing.get("prompt", 0)) * 1000
        out_per_1k = float(pricing.get("completion", 0)) * 1000
    except (TypeError, ValueError):
        inp_per_1k = out_per_1k = 0.0
    name = (entry.get("name") or model_id).split("/")[-1]
    return VisionModel(
        name=name,
        backend="openrouter",
        model_id=model_id,
        native_pdf=False,
        cost_per_1k_input=inp_per_1k,
        cost_per_1k_output=out_per_1k,
    )


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------
def _rows_to_summary_table(rows: list[dict]) -> list[list]:
    out = []
    for r in rows:
        status = "✓" if r["json_valid"] else "✗"
        error_hint = r.get("schema_error", "")[:120] if not r["json_valid"] else ""
        out.append([
            r["model_name"],
            status,
            len(r.get("units", [])),
            r.get("doc_type", "—"),
            r.get("input_tokens", 0),
            r.get("output_tokens", 0),
            f"${r.get('cost_usd', 0):.4f}",
            f"{r.get('latency_sec', 0):.1f}s",
            r.get("input_method", "—"),
            error_hint,
        ])
    return out


def _rows_to_unit_table(rows: list[dict]) -> list[list]:
    """Flatten all units across all models into per-unit rows, sorted by tenant then page."""
    out = []
    for r in rows:
        model_name = r["model_name"]
        for u in r.get("units", []):
            tenant = u.get("tenant_name") or ""
            unit_num = u.get("unit_number") or ""
            floor = u.get("floor_number") or ""
            unit_type = u.get("unit_type") or ""
            sqft = u.get("square_footage")  # may be null
            sqft_str = str(int(sqft)) if isinstance(sqft, (int, float)) else "null"
            conf = u.get("confidence", 0)
            page = u.get("page_number")
            page_str = str(page) if page is not None else ""
            evidence = u.get("evidence_quote", "")
            out.append([
                model_name,
                tenant,
                unit_num,
                floor,
                unit_type,
                sqft_str,
                f"{conf:.2f}" if isinstance(conf, float) else str(conf),
                page_str,
                evidence,
            ])

    def _sort_key(row):
        tenant = row[1].lower()
        try:
            page = int(row[7]) if row[7] else 9999
        except ValueError:
            page = 9999
        return (tenant, page)
    out.sort(key=_sort_key)
    return out


# ---------------------------------------------------------------------------
# Extraction runner
# ---------------------------------------------------------------------------
def run_extraction(
    upload_file,
    data_pdf_choice: str,
    model_ids: list[str],
    max_pages: int,
):
    """Run extraction; yield (status, summary_table, unit_table, gallery_images) progressively."""
    tmp_path: Path | None = None
    img_dir: Path | None = None
    try:
        # -- resolve PDF source --
        if upload_file is not None:
            try:
                tmp_path = _safe_temp_copy(upload_file.name)
            except ValueError as exc:
                logger.warning("Upload rejected: %s — file=%s", exc, upload_file.name)
                yield "Upload rejected: not a valid PDF file.", [], [], []
                return
            pdf_path = str(tmp_path)
            display_name = Path(upload_file.name).name
        elif data_pdf_choice:
            full = _REPO_ROOT / data_pdf_choice
            if not full.resolve().is_relative_to(_DATA_DIR.resolve()):
                logger.error("Path traversal attempt: %s", data_pdf_choice)
                yield "Invalid file selection.", [], [], []
                return
            if not full.exists():
                yield "Selected file not found.", [], [], []
                return
            pdf_path = str(full)
            display_name = data_pdf_choice
        else:
            yield "Please upload a PDF or select one from the data folder.", [], [], []
            return

        if not model_ids:
            yield "Please select at least one model.", [], [], []
            return

        if not os.environ.get("OPENROUTER_API_KEY"):
            yield "OPENROUTER_API_KEY is not configured on this server.", [], [], []
            return

        try:
            client = get_openrouter_client()
        except Exception:
            logger.exception("Failed to create OpenRouter client")
            yield "Could not connect to OpenRouter. Check the server log.", [], [], []
            return

        # -- create temp dir for page images (shared across all models) --
        img_dir = Path(tempfile.gettempdir()) / "lease_ocr_pages" / uuid.uuid4().hex
        img_dir.mkdir(parents=True, exist_ok=True)
        images_saved = False

        rows: list[dict] = []
        total = len(model_ids)
        gallery: list[tuple[str, str]] = []

        yield f"Starting extraction for **{display_name}** with {total} model(s)…", [], [], []

        for i, model_id in enumerate(model_ids, 1):
            model = _model_from_id(model_id)
            yield (
                f"Running {i}/{total}: {model.name}…",
                _rows_to_summary_table(rows),
                _rows_to_unit_table(rows),
                gallery,
            )

            try:
                if model.native_pdf:
                    result = call_openrouter_native_pdf(client, model, pdf_path, max_pages)
                    # native PDF path doesn't rasterize — save images on demand via fitz
                    if not images_saved:
                        _save_page_images_fitz(pdf_path, max_pages, img_dir)
                        gallery = _build_gallery(img_dir)
                        images_saved = True
                else:
                    result = call_openrouter_vision(
                        client, model, pdf_path, max_pages,
                        image_save_dir=img_dir if not images_saved else None,
                    )
                    if not images_saved:
                        gallery = _build_gallery(img_dir)
                        images_saved = True

                rows.append(result)
                logger.info(
                    "Completed %s on %s: valid=%s units=%d cost=$%.5f error=%r",
                    model.name, display_name,
                    result["json_valid"], len(result["units"]), result["cost_usd"],
                    result.get("schema_error", "") if not result["json_valid"] else "",
                )
            except Exception:
                logger.exception("Extraction failed: model=%s pdf=%s", model.model_id, display_name)
                rows.append({
                    "model_name": model.name,
                    "model_id": model.model_id,
                    "json_valid": False,
                    "schema_error": "Extraction error — see server log",
                    "units": [],
                    "doc_type": "unknown",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": 0.0,
                    "raw_response": "",
                    "input_method": "error",
                    "n_pages": 0,
                })

        total_cost = sum(r["cost_usd"] for r in rows)
        out_path = _save_run_results(rows, display_name, max_pages)
        yield (
            f"Done — {total} model(s) finished on **{display_name}**. "
            f"Total cost: ${total_cost:.4f}. "
            f"Raw responses saved to `{out_path.name}`.",
            _rows_to_summary_table(rows),
            _rows_to_unit_table(rows),
            gallery,
        )

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("Could not delete temp file: %s", tmp_path)
        if img_dir and img_dir.exists():
            try:
                shutil.rmtree(img_dir)
            except OSError:
                logger.warning("Could not delete image dir: %s", img_dir)


def _save_page_images_fitz(pdf_path: str, max_pages: int, img_dir: Path) -> None:
    """Rasterize pages via fitz for native-PDF models that don't call call_openrouter_vision."""
    try:
        import fitz
        from src.extraction.ocr import rasterize_page
        from benchmark.vision_run import _dpi_for_pages
        doc = fitz.open(pdf_path)
        n_pages = min(doc.page_count, max_pages)
        doc.close()
        dpi = _dpi_for_pages(n_pages)
        for pn in range(1, n_pages + 1):
            png = rasterize_page(pdf_path, pn, dpi=dpi)
            (img_dir / f"page_{pn:03d}.png").write_bytes(png)
    except Exception:
        logger.exception("Failed to save page images for gallery")


_RESULTS_DIR = _REPO_ROOT / "benchmark" / "results"


def _save_run_results(rows: list[dict], pdf_display_name: str, max_pages: int) -> Path:
    """Write one JSON file per app-UI run to benchmark/results/, mirroring the CLI output."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("app_%Y%m%d_%H%M%S")
    out_path = _RESULTS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "pdf": pdf_display_name,
        "max_pages": max_pages,
        "models": [
            {
                "model_name": r["model_name"],
                "model_id": r["model_id"],
                "json_valid": r["json_valid"],
                "schema_error": r.get("schema_error", ""),
                "doc_type": r.get("doc_type", ""),
                "units": r.get("units", []),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "cost_usd": r.get("cost_usd", 0.0),
                "latency_sec": r.get("latency_sec", 0.0),
                "input_method": r.get("input_method", ""),
                "n_pages": r.get("n_pages", 0),
                "raw_response": r.get("raw_response", ""),
            }
            for r in rows
        ],
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved run results to %s", out_path)
    except OSError:
        logger.exception("Failed to write run results to %s", out_path)
    return out_path


def _build_gallery(img_dir: Path) -> list[tuple[str, str]]:
    """Return sorted (path, label) pairs for Gradio Gallery."""
    images = sorted(img_dir.glob("page_*.png"))
    return [(str(p), f"Page {int(p.stem.split('_')[1])}") for p in images]


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def _build_ui() -> gr.Blocks:
    live_choices: list = []
    preset_ids = [m.model_id for m in VISION_MODELS]
    preset_labels = [m.name for m in VISION_MODELS]

    with gr.Blocks(title="Lease Vision Benchmark") as demo:
        gr.Markdown("## Lease Vision Benchmark\nExtract tenant units from commercial lease PDFs.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Source PDF")
                upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                gr.Markdown("**— or pick from data/ folder —**")
                data_choices = _list_data_pdfs()
                data_dropdown = gr.Dropdown(
                    label="data/ PDFs",
                    choices=data_choices,
                    value=None,
                    interactive=True,
                )
                refresh_btn = gr.Button("↻ Refresh list", size="sm")

                gr.Markdown("### Models")
                use_preset = gr.Checkbox(label="Use benchmark presets", value=True)
                preset_checkboxes = gr.CheckboxGroup(
                    label="Preset models",
                    choices=[(label, mid) for label, mid in zip(preset_labels, preset_ids)],
                    value=preset_ids,
                    visible=True,
                )
                live_dropdown = gr.Dropdown(
                    label="OpenRouter vision models (live)",
                    choices=live_choices,
                    multiselect=True,
                    value=[],
                    visible=False,
                    interactive=True,
                )
                refresh_models_btn = gr.Button("↻ Refresh model list", size="sm", visible=False)

                max_pages = gr.Slider(
                    label="Max pages per PDF",
                    minimum=1, maximum=100, step=1, value=50,
                )
                run_btn = gr.Button("Run Extraction", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Markdown("Ready.")
                results_table = gr.Dataframe(
                    headers=[
                        "Model", "Valid", "Units", "Doc type",
                        "In tokens", "Out tokens", "Cost", "Latency",
                        "Input method", "Error",
                    ],
                    datatype=["str"] * 10,
                    label="Results summary",
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(f"_Logs written to `{_LOG_PATH}`_")

        # -- Unit detail table --
        gr.Markdown("### Per-unit detail")
        unit_table = gr.Dataframe(
            headers=["Model", "Tenant", "Unit #", "Floor", "Unit Type", "SqFt", "Confidence", "Page", "Evidence / Pairing note"],
            datatype=["str"] * 9,
            label="Extracted units (sorted by tenant, then page)",
            interactive=False,
            wrap=True,
        )

        # -- Page gallery --
        gr.Markdown("### Source pages")
        page_gallery = gr.Gallery(
            label="PDF pages (rendered during extraction)",
            columns=3,
            height=500,
            object_fit="contain",
        )

        # -- Event wiring --

        def toggle_model_source(use_preset_val):
            return (
                gr.update(visible=use_preset_val),
                gr.update(visible=not use_preset_val),
                gr.update(visible=not use_preset_val),
            )

        use_preset.change(
            toggle_model_source,
            inputs=[use_preset],
            outputs=[preset_checkboxes, live_dropdown, refresh_models_btn],
        )

        def refresh_data():
            return gr.update(choices=_list_data_pdfs())

        refresh_btn.click(refresh_data, outputs=[data_dropdown])

        def refresh_models():
            global _or_models_cache
            _or_models_cache = None
            return gr.update(choices=_or_model_choices())

        refresh_models_btn.click(refresh_models, outputs=[live_dropdown])

        def get_model_ids(use_preset_val, preset_val, live_val):
            return (preset_val or []) if use_preset_val else (live_val or [])

        def _on_run(upload, data_choice, use_p, preset_val, live_val, mp):
            yield from run_extraction(
                upload, data_choice, get_model_ids(use_p, preset_val, live_val), mp
            )

        run_btn.click(
            fn=_on_run,
            inputs=[upload, data_dropdown, use_preset, preset_checkboxes, live_dropdown, max_pages],
            outputs=[status_box, results_table, unit_table, page_gallery],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY is not set. Extraction will fail.")

    demo = _build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=False,
    )
