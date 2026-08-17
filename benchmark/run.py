"""
Benchmark — compare models, prompt styles, effort levels, extraction modes,
temperature, and reasoning across a fixed set of lease PDFs.

Variables tested:
  --models    claude gpt-4o gpt-4o-mini deepseek llama
  --prompts   relaxed strict few_shot
  --effort    first2 half full          (how much text to send)
  --mode      text vision both          (extraction method)
  --temps     0 0.3                     (temperature)
  --reasoning off on                    (extended thinking / reasoning)

Usage:
    python benchmark/run.py                                   # sensible defaults
    python benchmark/run.py --models claude --prompts relaxed strict few_shot
    python benchmark/run.py --mode both --models claude
    python benchmark/run.py --reasoning off on --models claude deepseek
    python benchmark/run.py --temps 0 0.3 --models claude
    python benchmark/run.py --pdf path/to/file.pdf            # single PDF
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.models import MODELS, ModelConfig
from benchmark.prompts import PROMPTS, PromptConfig
from src.ai.extractor import _strip_code_block
from src.extraction.ocr import extract_scanned_text, rasterize_page
from src.extraction.searchable import extract_candidate_text
from src.ingestion.loader import load_pdf

# ---------------------------------------------------------------------------
# Fixed test set — 15 PDFs: mix of page counts, searchable, scanned, mixed
# ---------------------------------------------------------------------------
TEST_PDFS = [
    "data/SiteFiles_20260625_part01/SiteFiles/10099/20240912131552443975.pdf",   # 7pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/10439/20240912132106205145.pdf",   # 4pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/10455/20240912132122486809.pdf",   # 2pp no text (edge case)
    "data/SiteFiles_20260625_part01/SiteFiles/10103/20240918142905086890.pdf",   # 4pp mixed scan
    "data/SiteFiles_20260625_part01/SiteFiles/10195/20240918134820671739.pdf",   # 4pp mixed scan
    "data/SiteFiles_20260625_part01/SiteFiles/10350/20240912131941034699.pdf",   # 10pp heavy scan
    "data/SiteFiles_20260625_part01/SiteFiles/1001/20240912104716746251.pdf",    # 9pp heavy scan
    "data/SiteFiles_20260625_part01/SiteFiles/10713/20240912132524295488.pdf",   # 2pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/11097/20240918135005349219.pdf",   # 2pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/11524/20240918140345125332.pdf",   # 2pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/10977/20240912133001408947.pdf",   # 7pp mixed scan
    "data/SiteFiles_20260625_part01/SiteFiles/10800/20240912132708411685.pdf",   # 4pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/1150/20240912103837517060.pdf",    # 3pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/10657/20240912132441386579.pdf",   # 6pp searchable
    "data/SiteFiles_20260625_part01/SiteFiles/11163/20240912133302950116.pdf",   # 11pp searchable
]

EFFORT_LEVELS = {
    "first2": lambda text: "\n".join(text.split("\n")[:40]),
    "half":   lambda text: text[: len(text) // 2],
    "full":   lambda text: text,
}

# Models that support extended reasoning via OpenRouter
REASONING_MODELS = {
    "anthropic/claude-sonnet-4-5",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",          # already a reasoning model — no swap needed
    "qwen/qwen3-235b-a22b",          # Qwen 3 supports thinking mode
    "qwen/qwen3-30b-a3b",
}

# DeepSeek's dedicated reasoning model (separate model ID)
DEEPSEEK_REASONER_ID = "deepseek/deepseek-r1"

OUTPUT_DIR = Path("benchmark/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_PATH = Path("benchmark/ground_truth.csv")


def load_ground_truth() -> dict[str, int | None]:
    """Load ground truth SF values keyed by PDF filename."""
    if not GROUND_TRUTH_PATH.exists():
        return {}
    gt = {}
    with GROUND_TRUTH_PATH.open() as fh:
        for row in csv.DictReader(fh):
            val = row.get("correct_sf", "").strip()
            gt[row["pdf"]] = int(val) if val else None
    return gt


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def calc_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * model.cost_per_1k_input
        + output_tokens / 1000 * model.cost_per_1k_output
    )


def _parse_response(raw: str) -> dict:
    try:
        return json.loads(_strip_code_block(raw))
    except json.JSONDecodeError:
        return {
            "square_footage": None, "unit": None,
            "confidence": "low", "evidence_snippet": f"JSON parse error: {raw[:120]}",
        }


def call_model_text(
    client: OpenAI,
    model: ModelConfig,
    prompt: PromptConfig,
    text: str,
    temperature: float = 0,
    reasoning: bool = False,
) -> dict:
    """Send text to a model. Supports temperature and reasoning flags."""
    if not text.strip():
        return _empty_result("no text extracted from document")

    # For DeepSeek reasoning, swap to the reasoner model
    model_id = DEEPSEEK_REASONER_ID if (reasoning and "deepseek" in model.model_id) else model.model_id

    kwargs: dict = dict(
        model=model_id,
        max_tokens=1024 if reasoning else 512,
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": text},
        ],
    )

    # Claude extended thinking via OpenRouter
    if reasoning and "anthropic" in model.model_id:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 512}}

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        return _error_result(str(e))
    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content or ""
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    data = _parse_response(raw)

    return {
        "square_footage": data.get("square_footage"),
        "unit": data.get("unit"),
        "confidence": data.get("confidence", "low"),
        "evidence_snippet": data.get("evidence_snippet", ""),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": calc_cost(model, in_tok, out_tok),
        "latency_sec": round(elapsed, 2),
        "raw_response": raw,
    }


def call_model_vision(
    client: OpenAI,
    model: ModelConfig,
    prompt: PromptConfig,
    pdf_path: str,
    page_numbers: list[int],
    temperature: float = 0,
) -> dict:
    """Send page images to a vision-capable model."""
    if not model.supports_vision:
        return _empty_result("model does not support vision")
    if not page_numbers:
        return _empty_result("no pages for vision")

    content = []
    for pn in page_numbers[:3]:
        png_bytes = rasterize_page(pdf_path, pn)
        b64 = base64.standard_b64encode(png_bytes).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt.system_prompt})

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model.model_id,
            max_tokens=512,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        return _error_result(str(e))
    elapsed = time.perf_counter() - start

    raw = response.choices[0].message.content or ""
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    data = _parse_response(raw)

    return {
        "square_footage": data.get("square_footage"),
        "unit": data.get("unit"),
        "confidence": data.get("confidence", "low"),
        "evidence_snippet": data.get("evidence_snippet", ""),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": calc_cost(model, in_tok, out_tok),
        "latency_sec": round(elapsed, 2),
        "raw_response": raw,
    }


def _empty_result(reason: str) -> dict:
    return {
        "square_footage": None, "unit": None, "confidence": "low",
        "evidence_snippet": reason, "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0, "latency_sec": 0, "raw_response": "",
    }


def _error_result(msg: str) -> dict:
    return {
        "square_footage": None, "unit": None, "confidence": "error",
        "evidence_snippet": msg[:200], "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0, "latency_sec": 0, "raw_response": "",
    }


def benchmark_pdf(
    pdf_path: str,
    models: list[ModelConfig],
    prompts: list[PromptConfig],
    effort_names: list[str],
    modes: list[str],
    temperatures: list[float],
    reasoning_options: list[bool],
    client: OpenAI,
) -> list[dict]:
    path = Path(pdf_path)
    print(f"\n  PDF: {path.name}")

    try:
        doc = load_pdf(pdf_path)
        candidate_text = extract_candidate_text(doc.pages)
        ocr_text = extract_scanned_text(doc.pages, pdf_path)
        if ocr_text:
            candidate_text = candidate_text + "\n\n" + ocr_text
        searchable = sum(1 for p in doc.pages if p.mode == "searchable")
        scanned = sum(1 for p in doc.pages if p.mode == "scanned")
        all_pages = [p.page_number for p in doc.pages]
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return []

    rows = []

    for model in models:
        for prompt in prompts:
            for temp in temperatures:
                for reasoning in reasoning_options:
                    # Skip reasoning=True for models that don't support it
                    if reasoning and model.model_id not in REASONING_MODELS and "deepseek" not in model.model_id:
                        continue

                    reasoning_label = "reason=on" if reasoning else "reason=off"
                    temp_label = f"t={temp}"

                    # TEXT modes
                    if "text" in modes or "both" in modes:
                        for effort_name in effort_names:
                            trimmed = EFFORT_LEVELS[effort_name](candidate_text)
                            label = f"{model.name} | {prompt.name} | {effort_name} | {temp_label} | {reasoning_label}"
                            print(f"    → {label}...", end=" ", flush=True)
                            result = call_model_text(client, model, prompt, trimmed, temp, reasoning)
                            print(f"{result['square_footage']} ({result['confidence']}) | ${result['cost_usd']:.5f} | {result['latency_sec']}s")

                            rows.append(_make_row(
                                path, doc, searchable, scanned, candidate_text, trimmed,
                                model, prompt, f"text/{effort_name}", temp, reasoning, result,
                            ))

                    # VISION mode
                    if ("vision" in modes or "both" in modes) and model.supports_vision and not reasoning:
                        label = f"{model.name} | {prompt.name} | vision | {temp_label}"
                        print(f"    → {label}...", end=" ", flush=True)
                        result = call_model_vision(client, model, prompt, pdf_path, all_pages[:3], temp)
                        print(f"{result['square_footage']} ({result['confidence']}) | ${result['cost_usd']:.5f} | {result['latency_sec']}s")

                        rows.append(_make_row(
                            path, doc, searchable, scanned, candidate_text, "",
                            model, prompt, "vision", temp, False, result,
                        ))

    return rows


def _make_row(
    path, doc, searchable, scanned, full_text, sent_text,
    model, prompt, extraction_mode, temperature, reasoning, result,
) -> dict:
    return {
        "pdf": path.name,
        "store_folder": path.parent.name,
        "total_pages": doc.page_count,
        "searchable_pages": searchable,
        "scanned_pages": scanned,
        "text_chars_full": len(full_text),
        "text_chars_sent": len(sent_text),
        "model_name": model.name,
        "model_id": model.model_id,
        "prompt_style": prompt.name,
        "extraction_mode": extraction_mode,
        "temperature": temperature,
        "reasoning": reasoning,
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_results(rows: list[dict], run_id: str) -> Path:
    out = OUTPUT_DIR / f"benchmark_{run_id}.csv"
    if not rows:
        return out
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def is_correct(extracted: int | None, ground_truth: int | None, tolerance: float = 0.02) -> bool | None:
    """True if within 2% of ground truth. None if ground truth unknown."""
    if ground_truth is None:
        return None
    if extracted is None:
        return False
    return abs(extracted - ground_truth) / ground_truth <= tolerance


def print_summary(rows: list[dict]) -> None:
    gt = load_ground_truth()
    has_gt = bool(gt)

    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    if has_gt:
        gt_total = sum(1 for v in gt.values() if v is not None)
        print(f"Ground truth loaded: {gt_total} PDFs verified  (✓=correct  ✗=wrong  ?=unverified)")
    header = f"{'Combination':<55} {'Ext':>5} {'High':>5}"
    if has_gt:
        header += f" {'Correct':>8}"
    header += f" {'AvgCost':>10} {'AvgLat':>8} {'Total':>10}"
    print(header)
    print(f"{'-'*100}")

    def row_key(r):
        return (
            f"{r['model_name']} | {r['prompt_style']} | {r['extraction_mode']} | "
            f"t={r['temperature']} | reason={'on' if r['reasoning'] else 'off'}"
        )

    for combo, group in groupby(sorted(rows, key=row_key), key=row_key):
        group = list(group)
        n = len(group)
        extracted = sum(1 for r in group if r["square_footage"] is not None)
        high_conf = sum(1 for r in group if r["confidence"] == "high")
        avg_cost = sum(r["cost_usd"] for r in group) / n
        avg_lat = sum(r["latency_sec"] for r in group) / n
        total = sum(r["cost_usd"] for r in group)

        line = f"{combo:<55} {extracted}/{n:>2} {high_conf}/{n:>2}"

        if has_gt:
            correct = sum(
                1 for r in group
                if is_correct(r["square_footage"], gt.get(r["pdf"])) is True
            )
            verifiable = sum(1 for r in group if gt.get(r["pdf"]) is not None)
            line += f" {correct}/{verifiable:>2}    "

        line += f" ${avg_cost:>8.5f} {avg_lat:>6.2f}s ${total:>8.5f}"
        print(line)

    print(f"{'='*100}")
    print(f"Total spend this run: ${sum(r['cost_usd'] for r in rows):.4f}")
    print(f"{'='*100}\n")

    if has_gt:
        # Per-PDF breakdown for this run
        unverified = [pdf for pdf, v in gt.items() if v is None]
        if unverified:
            print(f"PDFs still needing manual verification ({len(unverified)}):")
            for pdf in unverified:
                print(f"  ✗  {pdf}")
            print(f"\nEdit benchmark/ground_truth.csv and fill in the correct_sf column.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lease PDF extraction across all variables")
    parser.add_argument("--models",    nargs="+", help="Model name keywords (e.g. claude gpt deepseek)")
    parser.add_argument("--prompts",   nargs="+", help="relaxed strict few_shot")
    parser.add_argument("--effort",    nargs="+", default=["full"], help="first2 half full")
    parser.add_argument("--mode",      nargs="+", default=["text"], help="text vision both")
    parser.add_argument("--temps",     nargs="+", default=["0"], help="Temperature values e.g. 0 0.3")
    parser.add_argument("--reasoning", nargs="+", default=["off"], help="off on")
    parser.add_argument("--pdf",       help="Run on a single PDF instead of the test set")
    args = parser.parse_args()

    models = MODELS
    if args.models:
        kws = [k.lower() for k in args.models]
        models = [m for m in MODELS if any(kw in m.name.lower() or kw in m.model_id.lower() for kw in kws)]
        if not models:
            print(f"No models matched: {args.models}"); sys.exit(1)

    prompts = PROMPTS
    if args.prompts:
        prompts = [p for p in PROMPTS if p.name in args.prompts]
        if not prompts:
            print(f"No prompts matched: {args.prompts}"); sys.exit(1)

    effort_names  = [e for e in args.effort if e in EFFORT_LEVELS]
    modes         = args.mode
    temperatures  = [float(t) for t in args.temps]
    reasoning_opts = [r == "on" for r in args.reasoning]

    pdfs = [args.pdf] if args.pdf else TEST_PDFS
    pdfs = [p for p in pdfs if Path(p).exists()]
    if not pdfs:
        print("No PDFs found."); sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = get_client()

    print(f"\nBenchmark run: {run_id}")
    print(f"Models:      {', '.join(m.name for m in models)}")
    print(f"Prompts:     {', '.join(p.name for p in prompts)}")
    print(f"Effort:      {', '.join(effort_names)}")
    print(f"Modes:       {', '.join(modes)}")
    print(f"Temperature: {', '.join(str(t) for t in temperatures)}")
    print(f"Reasoning:   {', '.join(args.reasoning)}")
    print(f"PDFs:        {len(pdfs)}")
    print(f"{'-'*50}")

    all_rows = []
    for pdf in pdfs:
        rows = benchmark_pdf(pdf, models, prompts, effort_names, modes, temperatures, reasoning_opts, client)
        all_rows.extend(rows)

    out = write_results(all_rows, run_id)
    print_summary(all_rows)
    print(f"Full results → {out}")


if __name__ == "__main__":
    main()
