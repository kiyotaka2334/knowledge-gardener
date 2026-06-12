"""Knowledge insights engine: bridge detection, trend analysis, narrative generation."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from knowledge_gardener.models import (
    BridgeConcept,
    ClusterIndex,
    ClusterSummary,
    ConceptEdge,
    ConceptGraph,
    ConceptIndex,
    ConceptTrend,
    EvergreenConcept,
    Insight,
    InsightReport,
    NarrativeEvent,
    VaultModel,
)

_WIKILINK_UNIT = 0.5  # identical to concept_clusterer.py


def analyze(
    vault: VaultModel,
    index: ConceptIndex,
    graph: ConceptGraph,
    clusters: ClusterIndex,
    recent_window_days: int = 90,
) -> InsightReport:
    """Generate a full InsightReport from the four core data structures."""
    vault_start, vault_end, vault_age_days = _vault_timespan(vault)

    bridge_concepts = _analyze_bridges(graph, clusters)
    evergreen_concepts = _analyze_evergreen(
        index, clusters, vault, vault_end, vault_age_days, recent_window_days
    )
    concept_trends = _analyze_trends(index, clusters, vault, vault_end, recent_window_days)
    cluster_summaries = _summarize_clusters(
        clusters, graph, bridge_concepts, concept_trends
    )
    narrative = _build_narrative(index, clusters, vault_start, vault_end, vault_age_days)
    insights = _render_insights(
        bridge_concepts,
        evergreen_concepts,
        concept_trends,
        cluster_summaries,
        narrative,
        clusters,
    )

    return InsightReport(
        version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        vault_root=graph.vault_root,
        vault_age_days=vault_age_days,
        total_notes=len(vault.notes),
        total_concepts=len(index.concepts),
        total_clusters=len(clusters.clusters),
        recent_window_days=recent_window_days,
        bridge_concepts=bridge_concepts,
        evergreen_concepts=evergreen_concepts,
        concept_trends=concept_trends,
        cluster_summaries=cluster_summaries,
        narrative=narrative,
        insights=insights,
    )


# ---------------------------------------------------------------------------
# Bridge detection
# ---------------------------------------------------------------------------

def _analyze_bridges(graph: ConceptGraph, clusters: ClusterIndex) -> list[BridgeConcept]:
    if not graph.nodes:
        return []

    # Accumulators keyed by node
    internal_weights: dict[str, float] = {n: 0.0 for n in graph.nodes}
    external_edge_map: dict[str, list[tuple[float, str, str]]] = {
        n: [] for n in graph.nodes
    }

    for edge in graph.edges:
        src_cid = clusters.node_cluster.get(edge.source)
        tgt_cid = clusters.node_cluster.get(edge.target)
        if src_cid is None or tgt_cid is None:
            continue
        w = _combined_weight(edge)
        if src_cid == tgt_cid:
            internal_weights[edge.source] += w
            internal_weights[edge.target] += w
        else:
            external_edge_map[edge.source].append((w, edge.target, tgt_cid))
            external_edge_map[edge.target].append((w, edge.source, src_cid))

    results: list[BridgeConcept] = []
    for node in sorted(graph.nodes):
        ext_edges = external_edge_map[node]
        if not ext_edges:
            continue
        int_w = internal_weights[node]
        ext_w = sum(e[0] for e in ext_edges)
        total_w = int_w + ext_w
        bridge_score = round(ext_w / total_w, 6) if total_w > 0 else 0.0
        bridged = sorted(set(e[2] for e in ext_edges))
        top = sorted(ext_edges, key=lambda e: (-e[0], e[1]))[:3]
        results.append(
            BridgeConcept(
                concept=node,
                home_cluster_id=clusters.node_cluster[node],
                bridged_cluster_ids=bridged,
                bridge_score=bridge_score,
                internal_weight=round(int_w, 6),
                external_weight=round(ext_w, 6),
                bridge_breadth=len(bridged),
                top_bridge_edges=[
                    {"concept": e[1], "cluster_id": e[2], "edge_weight": round(e[0], 6)}
                    for e in top
                ],
            )
        )

    results.sort(key=lambda b: (-b.bridge_score, -b.bridge_breadth, -b.external_weight, b.concept))
    return results


# ---------------------------------------------------------------------------
# Evergreen concepts
# ---------------------------------------------------------------------------

def _analyze_evergreen(
    index: ConceptIndex,
    clusters: ClusterIndex,
    vault: VaultModel,
    vault_end: datetime,
    vault_age_days: int,
    recent_window_days: int,
) -> list[EvergreenConcept]:
    total_notes = max(len(vault.notes), 1)
    results: list[EvergreenConcept] = []

    for name, concept in sorted(index.concepts.items()):
        if not concept.first_seen or not concept.last_seen:
            continue
        try:
            first = _parse_ts(concept.first_seen)
            last = _parse_ts(concept.last_seen)
        except ValueError:
            continue

        longevity_days = max((last - first).days, 0)
        days_since = max((vault_end - last).days, 0)

        breadth = round(concept.source_count / total_notes, 6)
        longevity_norm = (
            round(longevity_days / vault_age_days, 6) if vault_age_days > 0 else 0.0
        )
        if days_since <= recent_window_days:
            recency_norm = 1.0
        else:
            decay = (days_since - recent_window_days) / max(vault_age_days, 1)
            recency_norm = round(max(0.0, 1.0 - decay), 6)

        evergreen_score = round(
            0.4 * breadth + 0.4 * longevity_norm + 0.2 * recency_norm, 6
        )
        results.append(
            EvergreenConcept(
                concept=name,
                cluster_id=clusters.node_cluster.get(name, ""),
                source_count=concept.source_count,
                first_seen=concept.first_seen,
                last_seen=concept.last_seen,
                longevity_days=longevity_days,
                days_since_last_seen=days_since,
                breadth=breadth,
                longevity_norm=longevity_norm,
                recency_norm=recency_norm,
                evergreen_score=evergreen_score,
            )
        )

    results.sort(key=lambda e: (-e.evergreen_score, -e.source_count, e.concept))
    return results


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def _analyze_trends(
    index: ConceptIndex,
    clusters: ClusterIndex,
    vault: VaultModel,
    vault_end: datetime,
    recent_window_days: int,
) -> list[ConceptTrend]:
    cutoff = vault_end - timedelta(days=recent_window_days)
    results: list[ConceptTrend] = []

    for name, concept in sorted(index.concepts.items()):
        recent_count = 0
        historical_count = 0
        for note_id in concept.sources:
            note = vault.notes.get(note_id)
            if note and note.modified:
                try:
                    ts = _parse_ts(note.modified)
                    if ts >= cutoff:
                        recent_count += 1
                    else:
                        historical_count += 1
                except ValueError:
                    historical_count += 1
            else:
                historical_count += 1

        trend_ratio = round(recent_count / max(historical_count, 1), 6)
        trend_confidence = round(min(concept.source_count, 10) / 10, 6)

        # Label assignment — first match wins
        if concept.source_count <= 1:
            label = "insufficient data"
        elif recent_count > 0 and historical_count == 0:
            label = "emerging"
        elif trend_ratio >= 1.5 and historical_count > 0:
            label = "rising"
        elif 0.5 <= trend_ratio < 1.5:
            label = "stable"
        elif trend_ratio < 0.5 and recent_count > 0:
            label = "declining"
        else:
            label = "dormant"

        try:
            last_dt = _parse_ts(concept.last_seen) if concept.last_seen else vault_end
        except ValueError:
            last_dt = vault_end
        days_since = max((vault_end - last_dt).days, 0)

        results.append(
            ConceptTrend(
                concept=name,
                cluster_id=clusters.node_cluster.get(name, ""),
                label=label,
                recent_count=recent_count,
                historical_count=historical_count,
                trend_ratio=trend_ratio,
                trend_confidence=trend_confidence,
                first_seen=concept.first_seen,
                last_seen=concept.last_seen,
                days_since_last_seen=days_since,
            )
        )

    results.sort(key=lambda t: (t.label, -t.trend_confidence, t.concept))
    return results


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------

def _summarize_clusters(
    clusters: ClusterIndex,
    graph: ConceptGraph,
    bridge_concepts: list[BridgeConcept],
    concept_trends: list[ConceptTrend],
) -> list[ClusterSummary]:
    if not clusters.clusters:
        return []

    bridge_set = {b.concept for b in bridge_concepts}
    trend_map = {t.concept: t.label for t in concept_trends}

    # Count edges touching each cluster as internal vs external
    internal_edge_counts: dict[str, int] = {c.id: 0 for c in clusters.clusters}
    external_counts: dict[str, int] = {c.id: 0 for c in clusters.clusters}

    for edge in graph.edges:
        src_cid = clusters.node_cluster.get(edge.source)
        tgt_cid = clusters.node_cluster.get(edge.target)
        if src_cid is None or tgt_cid is None:
            continue
        if src_cid == tgt_cid:
            internal_edge_counts[src_cid] = internal_edge_counts.get(src_cid, 0) + 1
        else:
            external_counts[src_cid] = external_counts.get(src_cid, 0) + 1
            external_counts[tgt_cid] = external_counts.get(tgt_cid, 0) + 1

    results: list[ClusterSummary] = []
    for cluster in clusters.clusters:
        member_set = set(cluster.members)
        ext_count = external_counts.get(cluster.id, 0)
        int_count = internal_edge_counts.get(cluster.id, 0)
        total = int_count + ext_count
        isolation = round(1.0 - ext_count / total, 6) if total > 0 else 1.0

        bridge_members = sum(1 for m in cluster.members if m in bridge_set)

        if cluster.internal_edge_count > 0:
            centroid_int_deg = sum(
                1 for n in graph.neighbors(cluster.centroid) if n in member_set
            )
            hub_conc = round(
                centroid_int_deg / (2 * cluster.internal_edge_count), 6
            )
        else:
            hub_conc = 0.0

        labels = [trend_map.get(m, "insufficient data") for m in cluster.members]
        dominant = Counter(labels).most_common(1)[0][0] if labels else "insufficient data"
        centroid_trend = trend_map.get(cluster.centroid, "insufficient data")

        results.append(
            ClusterSummary(
                cluster_id=cluster.id,
                label=cluster.label,
                size=cluster.size,
                internal_density=cluster.internal_density,
                external_edge_count=ext_count,
                isolation_score=isolation,
                bridge_member_count=bridge_members,
                hub_concentration=hub_conc,
                dominant_trend=dominant,
                centroid_trend=centroid_trend,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(
    index: ConceptIndex,
    clusters: ClusterIndex,
    vault_start: datetime,
    vault_end: datetime,
    vault_age_days: int,
) -> list[NarrativeEvent]:
    distinct_dates = {
        _parse_ts(c.first_seen).date()
        for c in index.concepts.values()
        if c.first_seen
    }

    if len(distinct_dates) < 2:
        return [
            NarrativeEvent(
                period_label="Entire vault",
                period_start=vault_start.isoformat(),
                period_end=vault_end.isoformat(),
                statement_type="insufficient_temporal_data",
                statement=(
                    "Insufficient temporal data for narrative analysis. "
                    "All notes share the same modification timestamp — original "
                    "modification times may have been lost (e.g. via git clone)."
                ),
                dominant_cluster_id=None,
                new_concept_count=0,
                supporting_signals={"distinct_dates": len(distinct_dates)},
            )
        ]

    num_periods = min(4, len(distinct_dates))
    period_secs = (vault_end - vault_start).total_seconds() / num_periods
    periods: list[tuple[datetime, datetime]] = []
    for i in range(num_periods):
        p_start = vault_start + timedelta(seconds=i * period_secs)
        p_end = (
            vault_start + timedelta(seconds=(i + 1) * period_secs)
            if i < num_periods - 1
            else vault_end
        )
        periods.append((p_start, p_end))

    # Map each concept's first_seen to a period bucket
    period_data: list[dict[str, list[str]]] = [{} for _ in periods]
    for name, concept in sorted(index.concepts.items()):
        if not concept.first_seen:
            continue
        try:
            first = _parse_ts(concept.first_seen)
        except ValueError:
            continue
        cluster_id = clusters.node_cluster.get(name, "")
        for i, (p_start, p_end) in enumerate(periods):
            if first <= p_end or i == num_periods - 1:
                period_data[i].setdefault(cluster_id, []).append(name)
                break

    events: list[NarrativeEvent] = []

    for i, (p_start, p_end) in enumerate(periods):
        bucket = period_data[i]
        if not bucket:
            continue
        new_count = sum(len(v) for v in bucket.values())
        dominant = max(bucket, key=lambda k: len(bucket[k]))
        cluster_label = _cluster_label(dominant, clusters)

        if i == 0:
            stmt_type = "vault_origin"
            stmt = (
                f"The vault began in {_period_label(p_start, p_end)} "
                f"with concepts primarily from the '{cluster_label}' cluster "
                f"({len(bucket.get(dominant, []))} concepts)."
            )
        elif i == num_periods - 1:
            days = max((vault_end - p_start).days, 1)
            stmt_type = "current_focus"
            stmt = (
                f"The vault has most recently focused on the '{cluster_label}' cluster "
                f"({new_count} concepts added in the last {days} days)."
            )
        else:
            stmt_type = "cluster_dominance"
            stmt = (
                f"In {_period_label(p_start, p_end)}, the '{cluster_label}' cluster "
                f"was the most active theme ({len(bucket.get(dominant, []))} new concepts)."
            )

        events.append(
            NarrativeEvent(
                period_label=_period_label(p_start, p_end),
                period_start=p_start.isoformat(),
                period_end=p_end.isoformat(),
                statement_type=stmt_type,
                statement=stmt,
                dominant_cluster_id=dominant,
                new_concept_count=new_count,
                supporting_signals={
                    "concepts_per_cluster": {k: len(v) for k, v in bucket.items()}
                },
            )
        )

    return events


# ---------------------------------------------------------------------------
# Insight rendering
# ---------------------------------------------------------------------------

def _render_insights(
    bridge_concepts: list[BridgeConcept],
    evergreen_concepts: list[EvergreenConcept],
    concept_trends: list[ConceptTrend],
    cluster_summaries: list[ClusterSummary],
    narrative: list[NarrativeEvent],
    clusters: ClusterIndex,
) -> list[Insight]:
    insights: list[Insight] = []

    # Bridge insights
    for b in bridge_concepts[:10]:
        home_label = _cluster_label(b.home_cluster_id, clusters)
        if b.bridge_breadth == 1:
            other_label = _cluster_label(b.bridged_cluster_ids[0], clusters)
            headline = (
                f"'{b.concept}' connects the '{home_label}' and '{other_label}' clusters."
            )
        else:
            headline = (
                f"'{b.concept}' bridges {b.bridge_breadth + 1} clusters "
                f"(bridge score: {b.bridge_score:.2f})."
            )
        bridge_partners = ", ".join(e["concept"] for e in b.top_bridge_edges)
        explanation = (
            f"Internal weight: {b.internal_weight:.3f}. "
            f"External weight: {b.external_weight:.3f}. "
            f"Strongest external links: {bridge_partners}."
        )
        insights.append(
            Insight(
                id=_make_id("bridge", headline),
                category="bridge",
                headline=headline,
                explanation=explanation,
                concepts=[b.concept],
                clusters=sorted({b.home_cluster_id} | set(b.bridged_cluster_ids)),
                confidence=round(b.bridge_score, 3),
                supporting_signals={
                    "bridge_score": b.bridge_score,
                    "internal_weight": b.internal_weight,
                    "external_weight": b.external_weight,
                    "bridge_breadth": b.bridge_breadth,
                },
                rank=b.bridge_score,
            )
        )

    # Evergreen insights (top 10 by score)
    for ev in evergreen_concepts[:10]:
        headline = (
            f"'{ev.concept}' is an evergreen concept "
            f"({ev.source_count} notes, {ev.longevity_days}-day span, "
            f"score {ev.evergreen_score:.2f})."
        )
        explanation = (
            f"Breadth: {ev.breadth:.3f}. "
            f"Longevity: {ev.longevity_norm:.3f} ({ev.longevity_days} days). "
            f"Recency: {ev.recency_norm:.3f} ({ev.days_since_last_seen} days since last seen)."
        )
        insights.append(
            Insight(
                id=_make_id("evergreen", headline),
                category="evergreen",
                headline=headline,
                explanation=explanation,
                concepts=[ev.concept],
                clusters=[ev.cluster_id] if ev.cluster_id else [],
                confidence=round(ev.evergreen_score, 3),
                supporting_signals={
                    "source_count": ev.source_count,
                    "longevity_days": ev.longevity_days,
                    "breadth": ev.breadth,
                    "longevity_norm": ev.longevity_norm,
                    "recency_norm": ev.recency_norm,
                },
                rank=ev.evergreen_score,
            )
        )

    # Emerging concepts (all recent-only, capped at 10)
    emerging = sorted(
        [t for t in concept_trends if t.label == "emerging"],
        key=lambda t: (-t.recent_count, t.concept),
    )[:10]
    for t in emerging:
        headline = f"'{t.concept}' is an emerging concept ({t.recent_count} recent notes)."
        confidence = round(min(t.recent_count, 5) / 5, 3)
        insights.append(
            Insight(
                id=_make_id("emerging", headline),
                category="emerging",
                headline=headline,
                explanation=f"First seen: {t.first_seen}. No historical appearances.",
                concepts=[t.concept],
                clusters=[t.cluster_id] if t.cluster_id else [],
                confidence=confidence,
                supporting_signals={
                    "recent_count": t.recent_count,
                    "trend_confidence": t.trend_confidence,
                },
                rank=round(t.recent_count * t.trend_confidence, 6),
            )
        )

    # Declining / dormant concepts (capped at 10)
    fading = sorted(
        [t for t in concept_trends if t.label in ("dormant", "declining")],
        key=lambda t: (-t.days_since_last_seen, t.concept),
    )[:10]
    for t in fading:
        headline = (
            f"'{t.concept}' is {t.label} "
            f"({t.days_since_last_seen} days since last seen)."
        )
        confidence = round(min(t.historical_count, 5) / 5 * max(0.0, 1.0 - t.trend_ratio), 3)
        insights.append(
            Insight(
                id=_make_id("declining", headline),
                category="declining",
                headline=headline,
                explanation=(
                    f"Historical notes: {t.historical_count}. "
                    f"Recent notes: {t.recent_count}."
                ),
                concepts=[t.concept],
                clusters=[t.cluster_id] if t.cluster_id else [],
                confidence=confidence,
                supporting_signals={
                    "historical_count": t.historical_count,
                    "recent_count": t.recent_count,
                    "trend_ratio": t.trend_ratio,
                    "days_since_last_seen": t.days_since_last_seen,
                },
                rank=round(t.days_since_last_seen * t.trend_confidence, 6),
            )
        )

    # Cluster-level insights
    if cluster_summaries:
        non_singletons = [c for c in cluster_summaries if c.size > 1]

        # Largest cluster
        largest = max(cluster_summaries, key=lambda c: (c.size, c.cluster_id))
        headline = f"The '{largest.label}' cluster is the largest theme ({largest.size} concepts)."
        insights.append(
            Insight(
                id=_make_id("cluster", headline),
                category="cluster",
                headline=headline,
                explanation=(
                    f"Internal density: {largest.internal_density:.3f}. "
                    f"Bridge members: {largest.bridge_member_count}."
                ),
                concepts=[],
                clusters=[largest.cluster_id],
                confidence=round(min(largest.size, 20) / 20, 3),
                supporting_signals={
                    "size": largest.size,
                    "density": largest.internal_density,
                    "bridge_member_count": largest.bridge_member_count,
                },
                rank=float(largest.size),
            )
        )

        # Densest non-singleton cluster (if different from largest)
        if non_singletons:
            densest = max(non_singletons, key=lambda c: (c.internal_density, c.cluster_id))
            if densest.cluster_id != largest.cluster_id:
                headline = (
                    f"The '{densest.label}' cluster is the most tightly connected "
                    f"(density {densest.internal_density:.2f}, {densest.size} concepts)."
                )
                insights.append(
                    Insight(
                        id=_make_id("cluster", headline),
                        category="cluster",
                        headline=headline,
                        explanation="Every concept in this cluster is richly interconnected.",
                        concepts=[],
                        clusters=[densest.cluster_id],
                        confidence=round(
                            densest.internal_density * min(densest.size, 10) / 10, 3
                        ),
                        supporting_signals={
                            "density": densest.internal_density,
                            "size": densest.size,
                        },
                        rank=densest.internal_density,
                    )
                )

        # Most isolated cluster with meaningful isolation
        most_isolated = max(
            cluster_summaries, key=lambda c: (c.isolation_score, c.cluster_id)
        )
        if most_isolated.isolation_score > 0.8 and most_isolated.size > 1:
            headline = (
                f"The '{most_isolated.label}' cluster is relatively isolated "
                f"({most_isolated.external_edge_count} external connections)."
            )
            insights.append(
                Insight(
                    id=_make_id("cluster", headline),
                    category="cluster",
                    headline=headline,
                    explanation=f"Isolation score: {most_isolated.isolation_score:.3f}.",
                    concepts=[],
                    clusters=[most_isolated.cluster_id],
                    confidence=round(most_isolated.isolation_score, 3),
                    supporting_signals={
                        "isolation_score": most_isolated.isolation_score,
                        "external_edge_count": most_isolated.external_edge_count,
                    },
                    rank=most_isolated.isolation_score,
                )
            )

    # Narrative insights
    for event in narrative:
        if event.statement_type == "insufficient_temporal_data":
            insights.append(
                Insight(
                    id=_make_id("narrative", event.statement),
                    category="narrative",
                    headline="Temporal analysis unavailable: all notes share the same timestamp.",
                    explanation=event.statement,
                    concepts=[],
                    clusters=[],
                    confidence=0.0,
                    supporting_signals=event.supporting_signals,
                    rank=0.0,
                )
            )
        else:
            confidence = round(min(event.new_concept_count, 10) / 10, 3)
            insights.append(
                Insight(
                    id=_make_id("narrative", event.statement),
                    category="narrative",
                    headline=event.statement,
                    explanation=(
                        f"Period: {event.period_label}. "
                        f"New concepts: {event.new_concept_count}."
                    ),
                    concepts=[],
                    clusters=[event.dominant_cluster_id] if event.dominant_cluster_id else [],
                    confidence=confidence,
                    supporting_signals=event.supporting_signals,
                    rank=0.0,
                )
            )

    return insights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combined_weight(edge: ConceptEdge) -> float:
    return edge.wikilink_count * _WIKILINK_UNIT + edge.co_occurrence_weight


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp, always returning a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _vault_timespan(vault: VaultModel) -> tuple[datetime, datetime, int]:
    """Return (vault_start, vault_end, age_days) derived from note modified times."""
    timestamps: list[datetime] = []
    for note in vault.notes.values():
        if note.modified:
            try:
                timestamps.append(_parse_ts(note.modified))
            except ValueError:
                pass
    if not timestamps:
        now = datetime.now(timezone.utc)
        return now, now, 1
    vault_start = min(timestamps)
    vault_end = max(timestamps)
    age_days = max((vault_end - vault_start).days, 1)
    return vault_start, vault_end, age_days


def _make_id(category: str, text: str) -> str:
    return hashlib.sha256(f"{category}:{text}".encode()).hexdigest()[:8]


def _cluster_label(cluster_id: str, clusters: ClusterIndex) -> str:
    for c in clusters.clusters:
        if c.id == cluster_id:
            return c.label
    return cluster_id


def _period_label(start: datetime, end: datetime) -> str:
    if start.year == end.year and start.month == end.month:
        return start.strftime("%b %Y")
    elif start.year == end.year:
        return f"{start.strftime('%b')}-{end.strftime('%b %Y')}"
    return f"{start.strftime('%b %Y')}-{end.strftime('%b %Y')}"
