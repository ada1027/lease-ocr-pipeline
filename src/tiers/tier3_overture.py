"""Tier 3 — Overture / OSM building footprint area lookup via DuckDB.

For freestanding single-tenant locations, queries the Overture Maps buildings
layer to find the building polygon at a given (lat, lng) and computes its area
in square feet.

Overture data is accessed via DuckDB's httpfs extension reading Parquet files
directly from S3 — no local download needed.

Usage:
    result = lookup_footprint(lat=33.749, lng=-84.388, store_id="STORE-001")

The boss noted that ChainXY POI coordinates may not be on the rooftop, so we
do a radius search (default 50 m) and return the closest matching building.
Multi-tenant / inline stores will get a building that is larger than the tenant
space — caller should apply a freestanding filter before trusting the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Overture Maps S3 path (public, no auth needed via httpfs)
OVERTURE_BUILDINGS_PATH = (
    "s3://overturemaps-us-west-2/release/2026-06-17.0/theme=buildings/type=building/*"
)

# Search radius in metres around the supplied coordinate
DEFAULT_RADIUS_M = 50

# Approximate sq ft per sq m
SQM_TO_SQFT = 10.7639


@dataclass
class FootprintResult:
    store_id: str
    lat: float
    lng: float
    building_id: Optional[str]
    area_sqm: Optional[float]
    area_sqft: Optional[int]
    confidence: float
    source: str   # "overture_footprint" | "osm_footprint" | "no_match"
    evidence: str


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance between two WGS-84 points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lookup_footprint(
    lat: float,
    lng: float,
    store_id: str,
    radius_m: float = DEFAULT_RADIUS_M,
) -> FootprintResult:
    """Query Overture buildings layer for the building closest to (lat, lng).

    Returns a FootprintResult with area in both sq m and sq ft.
    source is "overture_footprint" on success, "no_match" if nothing found.
    """
    try:
        import duckdb
    except ImportError:
        raise RuntimeError("duckdb is required for Tier 3: pip install duckdb")

    # Bounding box for the radius search (approximate degrees)
    deg_lat = radius_m / 111_000
    deg_lng = radius_m / (111_000 * math.cos(math.radians(lat)))

    bbox_sql = f"""
        bbox.xmin >= {lng - deg_lng} AND bbox.xmax <= {lng + deg_lng}
        AND bbox.ymin >= {lat - deg_lat} AND bbox.ymax <= {lat + deg_lat}
    """

    logger.info("Tier 3: querying Overture for store=%s at (%.4f, %.4f) r=%.0fm", store_id, lat, lng, radius_m)

    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
        con.execute("SET s3_region = 'us-west-2';")

        # ST_Area on geographic degrees; we convert to sqm using local scale factor below
        query = f"""
            SELECT
                id,
                ST_Area(geometry)              AS area_deg2,
                bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax,
                ST_X(ST_Centroid(geometry))    AS centroid_lng,
                ST_Y(ST_Centroid(geometry))    AS centroid_lat
            FROM read_parquet('{OVERTURE_BUILDINGS_PATH}')
            WHERE bbox.xmin >= {lng - deg_lng} AND bbox.xmax <= {lng + deg_lng}
              AND bbox.ymin >= {lat - deg_lat} AND bbox.ymax <= {lat + deg_lat}
            ORDER BY
                (ST_X(ST_Centroid(geometry)) - {lng})^2
                + (ST_Y(ST_Centroid(geometry)) - {lat})^2
            LIMIT 5
        """
        rows = con.execute(query).fetchall()
        con.close()
    except Exception as e:
        logger.error("Overture query failed for store=%s: %s", store_id, e)
        return FootprintResult(
            store_id=store_id, lat=lat, lng=lng,
            building_id=None, area_sqm=None, area_sqft=None,
            confidence=0.0, source="no_match",
            evidence=f"Query error: {e}",
        )

    if not rows:
        logger.info("No Overture building found within %.0fm of store=%s", radius_m, store_id)
        return FootprintResult(
            store_id=store_id, lat=lat, lng=lng,
            building_id=None, area_sqm=None, area_sqft=None,
            confidence=0.0, source="no_match",
            evidence=f"No building within {radius_m}m in Overture dataset",
        )

    building_id, area_deg2, xmin, xmax, ymin, ymax, c_lng, c_lat = rows[0]
    dist_m = _haversine_m(lat, lng, c_lat, c_lng)

    # Convert degrees² → metres².
    # ST_Area returns NaN for some geometries; fall back to bbox area.
    import math as _math
    mid_lat = (ymin + ymax) / 2 if (ymin and ymax) else lat
    m_per_deg_lat = 111_000
    m_per_deg_lng = 111_000 * _math.cos(_math.radians(mid_lat))
    if area_deg2 and not _math.isnan(area_deg2) and area_deg2 > 0:
        area_sqm = area_deg2 * m_per_deg_lat * m_per_deg_lng
    elif xmin and xmax and ymin and ymax:
        area_sqm = (xmax - xmin) * m_per_deg_lng * (ymax - ymin) * m_per_deg_lat
    else:
        area_sqm = None

    # Confidence degrades with distance from the POI coordinate
    if dist_m <= 10:
        confidence = 0.90
    elif dist_m <= 25:
        confidence = 0.75
    elif dist_m <= 50:
        confidence = 0.60
    else:
        confidence = 0.40

    area_sqft = int(area_sqm * SQM_TO_SQFT) if (area_sqm and not math.isnan(area_sqm)) else None

    logger.info(
        "Overture match: store=%s building=%s area=%.0f sqm (%.0f sqft) dist=%.1fm conf=%.2f",
        store_id, building_id, area_sqm or 0, area_sqft or 0, dist_m, confidence,
    )

    return FootprintResult(
        store_id=store_id, lat=lat, lng=lng,
        building_id=str(building_id),
        area_sqm=round(area_sqm, 1) if area_sqm else None,
        area_sqft=area_sqft,
        confidence=confidence,
        source="overture_footprint",
        evidence=f"Overture building {building_id}, centroid {dist_m:.1f}m from POI, {area_sqm:.0f} sqm footprint",
    )


def batch_lookup(
    records: list[dict],   # each needs: store_id, lat, lng
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[FootprintResult]:
    """Run footprint lookup for a list of store records."""
    results = []
    for rec in records:
        result = lookup_footprint(
            lat=float(rec["lat"]),
            lng=float(rec["lng"]),
            store_id=str(rec["store_id"]),
            radius_m=radius_m,
        )
        results.append(result)
    return results
