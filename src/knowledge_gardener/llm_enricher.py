"""Gemini LLM enrichment layer for Knowledge Gardener analysis outputs.

Runs after the statistical pipeline as an optional presentation layer.
If GEMINI_API_KEY is not set, or if the API call fails for any reason,
the pipeline degrades gracefully to the existing template-based reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from knowledge_gardener.models import ClusterIndex, ConceptCluster, InsightReport

logger = logging.getLogger(__name__)

ENRICHMENT_SCHEMA_VERSION = 1
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_CLUSTERS_IN_PAYLOAD = 15  # cap payload size for large vaults


@dataclass
class ClusterConnection:
    """A structured cross-cluster link identified by the LLM."""

    cluster_a: str
    cluster_b: str
    reason: str
    confidence: float = 1.0


@dataclass
class LLMEnrichment:
    """All LLM-generated insights for one analysis run."""

    schema_version: int
    model: str
    cluster_descriptions: dict[str, str]   # cluster_id -> 1-2 sentence description
    cluster_confidence: dict[str, float]   # cluster_id -> 0.0-1.0
    connections: list[ClusterConnection]   # structured cross-cluster observations
    questions: list[str]                   # 3-5 reflective questions
    weekly_narrative: str | None           # prose summary of the diff (or None)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    payload_hash: str = ""


def generate_human_label(cluster: ConceptCluster) -> str:
    """Deterministic readable label using top-3 members (no LLM needed).

    Produces "A & B" or "A, B & C" instead of just the centroid name,
    giving the LLM more context about the cluster's content.
    """
    top = cluster.members[:3]
    if len(top) == 1:
        return top[0].title()
    if len(top) == 2:
        return f"{top[0].title()} & {top[1].title()}"
    return f"{top[0].title()}, {top[1].title()} & {top[2].title()}"


def build_payload(
    report: InsightReport,
    clusters: ClusterIndex,
    diff_data: dict | None = None,
) -> dict:
    """Build the compact JSON context sent to Gemini.

    Capped at MAX_CLUSTERS_IN_PAYLOAD to prevent prompt bloat on large vaults.
    Clusters are ranked by size × density to prioritise the most coherent ones.
    """
    non_singletons = [c for c in clusters.clusters if c.size > 1]
    sorted_clusters = sorted(
        non_singletons,
        key=lambda c: c.size * c.internal_density,
        reverse=True,
    )
    top_clusters = sorted_clusters[:MAX_CLUSTERS_IN_PAYLOAD]
    omitted = max(0, len(non_singletons) - MAX_CLUSTERS_IN_PAYLOAD)

    cluster_data = [
        {
            "id": c.id,
            "label": c.label,
            "human_label": generate_human_label(c),
            "size": c.size,
            "density": round(c.internal_density, 3),
            "top_members": c.members[:10],
        }
        for c in top_clusters
    ]

    bridge_data = [
        {
            "concept": b.concept,
            "connects": [b.home_cluster_id] + b.bridged_cluster_ids[:3],
            "bridge_score": round(b.bridge_score, 3),
        }
        for b in report.bridge_concepts[:10]
    ]

    trend_data = {
        "emerging": [t.concept for t in report.concept_trends if t.label == "emerging"][:10],
        "declining": [
            t.concept for t in report.concept_trends
            if t.label in ("dormant", "declining")
        ][:10],
    }

    top_insights = [
        {"category": i.category, "headline": i.headline}
        for i in sorted(report.insights, key=lambda x: x.rank, reverse=True)[:5]
    ]

    payload: dict = {
        "stats": {
            "total_notes": report.total_notes,
            "total_concepts": report.total_concepts,
            "cluster_count": len(non_singletons),
            "clusters_omitted_from_payload": omitted,
        },
        "clusters": cluster_data,
        "bridges": bridge_data,
        "trends": trend_data,
        "top_insights": top_insights,
    }

    if diff_data:
        payload["weekly_diff"] = diff_data

    return payload


def payload_hash(p: dict) -> str:
    """Stable SHA-256 hash of a payload dict (first 16 hex chars)."""
    canonical = json.dumps(p, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


_SYSTEM_PROMPT = """\
You are a knowledge-graph analyst helping a user understand the structure
of their personal Obsidian vault. Your role is descriptive, not prescriptive.

STRICT RULES:
- Only use information present in the supplied JSON. Do not infer the user's
  goals, emotions, beliefs, career plans, or personal circumstances beyond
  what the provided concepts directly state.
- Do not introduce concepts or connections absent from the data.
- If you have insufficient evidence, say so briefly instead of guessing.
- Write in second person ("Your notes on X suggest...") and keep language
  neutral and factual.
- Descriptions must be 1-2 sentences maximum. Do not pad.
"""

_USER_PROMPT_TEMPLATE = """\
Analyze the following knowledge vault data and return a JSON object that \
matches this schema EXACTLY — no markdown, no code fences, no commentary \
outside the JSON object:

{{
  "cluster_descriptions": {{
    "<cluster_id>": "<1-2 sentence factual description based ONLY on the provided members>",
    ... one entry per cluster in the data ...
  }},
  "cluster_confidence": {{
    "<cluster_id>": <float 0.0-1.0 — how clearly the members form a coherent theme>,
    ...
  }},
  "connections": [
    {{
      "cluster_a": "<cluster_id from the data>",
      "cluster_b": "<cluster_id from the data>",
      "reason": "<one sentence: what specifically links these themes, based on the data>",
      "confidence": <float 0.0-1.0>
    }}
    ... limit to the 3-5 most evidenced cross-cluster links ...
  ],
  "questions": [
    "<3-5 questions grounded in specific concepts from the data, not generic advice>"
  ],
  "weekly_narrative": "<1-2 sentence prose summary of what changed this week, \
or null if no weekly_diff data is present in the payload>"
}}

Rules:
- Only reference cluster_ids that appear in the "clusters" list below.
- Return ONLY valid JSON. No markdown. No code fences. No text before or after.

VAULT DATA:
{payload_json}
"""


def enrich(
    report: InsightReport,
    clusters: ClusterIndex,
    diff_data: dict | None = None,
) -> LLMEnrichment | None:
    """Call Gemini to generate enriched insights.

    Returns None if GEMINI_API_KEY is not set, or if the call fails for
    any reason. The statistical pipeline is unaffected either way.
    Override the model via the GEMINI_MODEL environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        from google import genai  # type: ignore[import]
    except Exception:
        logger.warning(
            "google-genai not available; skipping LLM enrichment. "
            "Run: pip install 'knowledge-gardener[llm]'"
        )
        return None

    payload = build_payload(report, clusters, diff_data)
    p_hash = payload_hash(payload)
    prompt = _USER_PROMPT_TEMPLATE.format(payload_json=json.dumps(payload, indent=2))

    t0 = time.monotonic()
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "temperature": 0.2,
            },
        )
    except Exception as exc:
        logger.warning("Gemini API call failed: %s", exc)
        return None

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw_text = (response.text or "").strip()

    # Strip markdown code fences if Gemini wraps the output anyway
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse Gemini response as JSON: %s", exc)
        logger.debug("Raw response (first 500 chars): %s", raw_text[:500])
        return None

    # Extract token telemetry safely across SDK versions
    tokens_in = tokens_out = 0
    usage = getattr(response, "usage_metadata", None)
    if usage:
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0

    # Parse structured connections
    connections: list[ClusterConnection] = []
    for conn in data.get("connections", []):
        try:
            connections.append(ClusterConnection(
                cluster_a=conn["cluster_a"],
                cluster_b=conn["cluster_b"],
                reason=conn["reason"],
                confidence=float(conn.get("confidence", 1.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue

    logger.info(
        "LLM enrichment complete: model=%s tokens=%d+%d latency=%dms",
        model, tokens_in, tokens_out, latency_ms,
    )

    return LLMEnrichment(
        schema_version=ENRICHMENT_SCHEMA_VERSION,
        model=model,
        cluster_descriptions=data.get("cluster_descriptions", {}),
        cluster_confidence=data.get("cluster_confidence", {}),
        connections=connections,
        questions=data.get("questions", []),
        weekly_narrative=data.get("weekly_narrative"),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        payload_hash=p_hash,
    )


# ---------------------------------------------------------------------------
# Snapshot serialisation helpers
# ---------------------------------------------------------------------------

def enrichment_to_dict(enrichment: LLMEnrichment) -> dict:
    """Serialise enrichment content (the LLM output) for snapshot storage."""
    return {
        "cluster_descriptions": enrichment.cluster_descriptions,
        "cluster_confidence": enrichment.cluster_confidence,
        "connections": [
            {
                "cluster_a": c.cluster_a,
                "cluster_b": c.cluster_b,
                "reason": c.reason,
                "confidence": c.confidence,
            }
            for c in enrichment.connections
        ],
        "questions": enrichment.questions,
        "weekly_narrative": enrichment.weekly_narrative,
    }


def enrichment_meta_dict(enrichment: LLMEnrichment) -> dict:
    """Serialise enrichment metadata (versioning + telemetry) for snapshot storage."""
    return {
        "schema_version": enrichment.schema_version,
        "model": enrichment.model,
        "tokens_in": enrichment.tokens_in,
        "tokens_out": enrichment.tokens_out,
        "latency_ms": enrichment.latency_ms,
        "payload_hash": enrichment.payload_hash,
    }


def enrichment_from_snapshot(
    enrichment_data: dict,
    meta: dict,
) -> LLMEnrichment | None:
    """Reconstruct LLMEnrichment from cached snapshot data."""
    if not enrichment_data or not meta:
        return None
    try:
        connections = [
            ClusterConnection(
                cluster_a=c["cluster_a"],
                cluster_b=c["cluster_b"],
                reason=c["reason"],
                confidence=float(c.get("confidence", 1.0)),
            )
            for c in enrichment_data.get("connections", [])
        ]
        return LLMEnrichment(
            schema_version=meta.get("schema_version", ENRICHMENT_SCHEMA_VERSION),
            model=meta.get("model", ""),
            cluster_descriptions=enrichment_data.get("cluster_descriptions", {}),
            cluster_confidence=enrichment_data.get("cluster_confidence", {}),
            connections=connections,
            questions=enrichment_data.get("questions", []),
            weekly_narrative=enrichment_data.get("weekly_narrative"),
            tokens_in=meta.get("tokens_in", 0),
            tokens_out=meta.get("tokens_out", 0),
            latency_ms=meta.get("latency_ms", 0),
            payload_hash=meta.get("payload_hash", ""),
        )
    except (KeyError, TypeError, ValueError):
        return None
