"""Prompt variants to test in benchmarking."""

from dataclasses import dataclass


@dataclass
class PromptConfig:
    name: str
    system_prompt: str
    description: str


PROMPTS = [
    PromptConfig(
        name="relaxed",
        description="Accept any SF figure — current production approach",
        system_prompt="""\
You are a commercial real estate analyst. Extract the primary square footage from the document.

Rules:
- Accept any square footage figure: total center size, GLA, demised premises, suite size, or building size
- If multiple sizes appear, pick the largest or most prominent one
- Marketing flyers and tenant rosters are valid sources

Return ONLY raw JSON — no markdown, no code blocks:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – exact text you used (200 chars max)
""",
    ),

    PromptConfig(
        name="strict",
        description="Only formal lease language — demised premises, GLA, rentable area",
        system_prompt="""\
You are a commercial real estate lease analyst. Extract the primary leasable square footage
from the lease document provided.

Rules:
- Only use square footage from formal lease clauses: Demised Premises, Premises, GLA, Rentable Area
- Do NOT use square footage from marketing text, site plans, or tenant rosters
- If no formal lease clause is found, return square_footage as null with confidence "low"

Return ONLY raw JSON — no markdown, no code blocks:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – exact verbatim clause you relied on (200 chars max)
""",
    ),

    PromptConfig(
        name="few_shot",
        description="Includes 3 worked examples before the task",
        system_prompt="""\
You are a commercial real estate analyst. Extract the primary square footage from the document.
Accept any square footage figure: total center size, GLA, demised premises, suite size, or building size.
If multiple sizes appear, pick the largest or most prominent one.

Here are examples of correct extractions:

EXAMPLE 1:
Text: "Village of Blaine Shopping Center — 221,239 square foot center anchored by Cub Foods"
Output: {"square_footage": 221239, "unit": "sq ft", "confidence": "high", "evidence_snippet": "221,239 square foot center anchored by Cub Foods"}

EXAMPLE 2:
Text: "AVAILABLE SPACE: 3,500 SF | Suite 20057 | $18/SF NNN"
Output: {"square_footage": 3500, "unit": "sq ft", "confidence": "high", "evidence_snippet": "AVAILABLE SPACE: 3,500 SF | Suite 20057"}

EXAMPLE 3:
Text: "Parking ratio: 4.5 per 1,000 SF. Call for pricing."
Output: {"square_footage": null, "unit": null, "confidence": "low", "evidence_snippet": "No square footage found — only a parking ratio mentioned"}

Now extract from the document below.
Return ONLY raw JSON — no markdown, no code blocks:
  square_footage   – integer or null
  unit             – "sq ft" | "sq m" | null
  confidence       – "high" | "medium" | "low"
  evidence_snippet – exact text you used (200 chars max)
""",
    ),
]
