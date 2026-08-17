"""Tier runner — run individual cascade tiers from the command line.

Usage:
    python tiers.py tier2 --store STORE-001 --chain "Chipotle" --sf 2400
    python tiers.py tier3 --store STORE-001 --lat 33.749 --lng -84.388
    python tiers.py scrape                        # runs Tier 0.5 batch
    python tiers.py scrape --chain "Chipotle"     # filter by chain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def run_tier2(args):
    from src.tiers.tier2_bounds import load_chain_averages, check_bounds, fallback_from_average

    averages = load_chain_averages()
    if not averages:
        print("No chain_averages.csv found. Drop your averages file at data/chain_averages.csv")
        print("Template: data/chain_averages_template.csv")
        sys.exit(1)

    result = check_bounds(args.chain, args.sf, averages)
    print(json.dumps(result.as_dict(), indent=2))

    if not result.within_bounds:
        fallback = fallback_from_average(args.chain, averages)
        print(f"\n⚠  Bounds violation ({result.flag}). Chain average fallback: {fallback} sq ft")


def run_tier3(args):
    from src.tiers.tier3_overture import lookup_footprint
    import dataclasses

    result = lookup_footprint(lat=args.lat, lng=args.lng, store_id=args.store)
    print(json.dumps(dataclasses.asdict(result), indent=2))


def run_scrape(args):
    from src.tiers.tier0_5_scraper import run_batch
    from pathlib import Path

    centers_path = Path("data/shopping_centers.csv")
    if not centers_path.exists():
        print("No shopping_centers.csv found. Copy the template:")
        print("  cp data/shopping_centers_template.csv data/shopping_centers.csv")
        print("Then fill in your center list.")
        sys.exit(1)

    results = run_batch(chain_filter=args.chain if hasattr(args, "chain") else None)
    ok    = sum(1 for r in results if r.status == "ok")
    pdfs  = sum(len(r.pdfs_downloaded) for r in results)
    with_sf = sum(1 for r in results if r.sf_from_directory)
    print(f"\nScrape complete: {len(results)} centers, {ok} ok, {pdfs} PDFs, {with_sf} direct SF found")


def main():
    parser = argparse.ArgumentParser(description="Cascade tier runner")
    sub = parser.add_subparsers(dest="tier", required=True)

    p2 = sub.add_parser("tier2", help="Chain bounds check")
    p2.add_argument("--store", required=True)
    p2.add_argument("--chain", required=True, help="Chain name (must match chain_averages.csv)")
    p2.add_argument("--sf", required=True, type=int, help="Extracted SF to check")

    p3 = sub.add_parser("tier3", help="Overture building footprint lookup")
    p3.add_argument("--store", required=True)
    p3.add_argument("--lat", required=True, type=float)
    p3.add_argument("--lng", required=True, type=float)

    ps = sub.add_parser("scrape", help="Tier 0.5 browser-harness leasing scraper")
    ps.add_argument("--chain", help="Filter to a specific chain name")

    args = parser.parse_args()
    {"tier2": run_tier2, "tier3": run_tier3, "scrape": run_scrape}[args.tier](args)


if __name__ == "__main__":
    main()
