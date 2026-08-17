"""Tier 2 — Chain min/max/avg bounds check.

Loads a chain_averages CSV and flags any extracted SF value that falls outside
the per-brand envelope [0.3 × min, 3 × max]. Does NOT produce a new SF value —
it annotates existing records from other tiers.

Expected CSV columns: chain_name, avg_sf, min_sf, max_sf
(boss confirmed this already exists; drop it at data/chain_averages.csv)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PATH = Path("data/chain_averages.csv")

# How far outside the envelope before we flag it
LOWER_MULTIPLIER = 0.30   # below 30% of min → suspicious
UPPER_MULTIPLIER = 3.00   # above 300% of max → suspicious


@dataclass
class ChainBounds:
    chain_name: str
    avg_sf: int
    min_sf: int
    max_sf: int

    @property
    def lower_bound(self) -> float:
        return self.min_sf * LOWER_MULTIPLIER

    @property
    def upper_bound(self) -> float:
        return self.max_sf * UPPER_MULTIPLIER


@dataclass
class BoundsResult:
    chain_name: str
    extracted_sf: int
    avg_sf: int
    min_sf: int
    max_sf: int
    within_bounds: bool
    flag: str   # "ok" | "below_floor" | "above_ceiling" | "no_chain_data"

    def as_dict(self) -> dict:
        return {
            "chain_name": self.chain_name,
            "avg_sf": self.avg_sf,
            "min_sf": self.min_sf,
            "max_sf": self.max_sf,
            "within_bounds": self.within_bounds,
            "bounds_flag": self.flag,
        }


def load_chain_averages(path: Path = DEFAULT_PATH) -> dict[str, ChainBounds]:
    """Load chain_averages.csv → dict keyed by lowercased chain name."""
    if not path.exists():
        logger.warning("chain_averages.csv not found at %s — Tier 2 will skip bounds checks.", path)
        return {}

    bounds: dict[str, ChainBounds] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                name = row["chain_name"].strip().lower()
                bounds[name] = ChainBounds(
                    chain_name=row["chain_name"].strip(),
                    avg_sf=int(float(row["avg_sf"])),
                    min_sf=int(float(row["min_sf"])),
                    max_sf=int(float(row["max_sf"])),
                )
            except (KeyError, ValueError) as e:
                logger.warning("Skipping bad row in chain_averages.csv: %s — %s", row, e)

    logger.info("Loaded %d chain bounds from %s", len(bounds), path)
    return bounds


def check_bounds(
    chain_name: str,
    extracted_sf: int,
    chain_averages: dict[str, ChainBounds],
) -> BoundsResult:
    """Check whether extracted_sf is within the chain's expected envelope."""
    key = chain_name.strip().lower()
    bounds = chain_averages.get(key)

    if bounds is None:
        return BoundsResult(
            chain_name=chain_name, extracted_sf=extracted_sf,
            avg_sf=0, min_sf=0, max_sf=0,
            within_bounds=True, flag="no_chain_data",
        )

    if extracted_sf < bounds.lower_bound:
        flag = "below_floor"
        within = False
    elif extracted_sf > bounds.upper_bound:
        flag = "above_ceiling"
        within = False
    else:
        flag = "ok"
        within = True

    if not within:
        logger.warning(
            "Bounds violation: chain=%s extracted=%s outside [%.0f, %.0f]",
            chain_name, extracted_sf, bounds.lower_bound, bounds.upper_bound,
        )

    return BoundsResult(
        chain_name=chain_name, extracted_sf=extracted_sf,
        avg_sf=bounds.avg_sf, min_sf=bounds.min_sf, max_sf=bounds.max_sf,
        within_bounds=within, flag=flag,
    )


def fallback_from_average(chain_name: str, chain_averages: dict[str, ChainBounds]) -> Optional[int]:
    """Return chain average SF as a fallback value (source: brand_average_fallback)."""
    bounds = chain_averages.get(chain_name.strip().lower())
    return bounds.avg_sf if bounds else None
