"""Full experiment / knowledge ledger for hypothesize (not top-k retrieval)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.shared.experiments.hypothesis import (
    BASELINE_HYPOTHESIS_ID,
    HypothesisStore,
)
from labpilot.research_engine.shared.experiments.models import Hypothesis, HypothesisStatus
from labpilot.research_engine.shared.labels import is_record_reference

_META_TAGS = frozenset(
    {
        "baseline",
        "technique",
        "pipeline_diff",
        "transfer",
        "failure_fix",
        "improvement",
        "stacked",
        "combination",
        "ablation",
        "follow-up",
        "execution",
        "submit",
        "belief",
        "unused_belief",
        "unused_claim",
        "untried",
    }
)


class TechniqueRecord(BaseModel):
    name: str
    label: str
    category: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    status: str = "untried"  # worked | failed | untried


class ExperimentLedger(BaseModel):
    """Competition-wide inventory for candidate generation and ranking."""

    competition: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    techniques: list[TechniqueRecord] = Field(default_factory=list)
    techniques_worked: list[str] = Field(default_factory=list)
    techniques_failed: list[str] = Field(default_factory=list)
    techniques_untried: list[str] = Field(default_factory=list)
    beliefs_unused: list[dict[str, Any]] = Field(default_factory=list)
    claims_unused: list[dict[str, Any]] = Field(default_factory=list)
    winning_hypothesis_id: str | None = None
    winning_stack: list[str] = Field(default_factory=list)
    winning_confidence: float = 0.5
    winning_gain: float = 0.0
    avoid_pairs: list[tuple[str, str]] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)

    def technique_names(self) -> list[str]:
        return [t.name for t in self.techniques]

    def is_failed(self, name: str) -> bool:
        label = normalize_label(name)
        return label in {normalize_label(x) for x in self.techniques_failed}

    def is_worked(self, name: str) -> bool:
        label = normalize_label(name)
        return label in {normalize_label(x) for x in self.techniques_worked}


def build_experiment_ledger(
    knowledge_dir: Path,
    competition: str,
) -> ExperimentLedger:
    """Scan all artifacts, techniques, hyps, beliefs, claims without top-k cut."""
    hyp_store = HypothesisStore(knowledge_dir, competition)
    hyps = hyp_store.list()

    artifacts: list[dict[str, Any]] = []
    technique_index: dict[str, TechniqueRecord] = {}
    claims_all: list[dict[str, Any]] = []
    beliefs_all: list[dict[str, Any]] = []

    with KnowledgeStore(knowledge_dir, competition) as store:
        for art in store.list_artifacts():
            artifacts.append(
                {
                    "id": art.id,
                    "type": str(art.type),
                    "title": art.title,
                    "techniques": list(art.techniques),
                    "claims": list(art.claims),
                    "feature_recipes": (art.metadata or {}).get("feature_recipes") or [],
                }
            )
            for tech in art.techniques:
                _index_technique(technique_index, tech, artifact_id=art.id)
            for recipe in (art.metadata or {}).get("feature_recipes") or []:
                if isinstance(recipe, dict) and recipe.get("name"):
                    _index_technique(
                        technique_index,
                        str(recipe["name"]),
                        artifact_id=art.id,
                        category="feature_engineering",
                    )
            for claim in art.claims:
                text = str(claim).strip()
                if text:
                    claims_all.append({"text": text, "artifact_id": art.id})

        for row in store.list_techniques():
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            _index_technique(
                technique_index,
                name,
                category=str(row.get("category") or ""),
            )

        for belief in store.list_beliefs():
            beliefs_all.append(dict(belief) if not isinstance(belief, dict) else belief)

    worked, failed = _technique_outcomes(hyps)
    for label, record in technique_index.items():
        if label in worked:
            record.status = "worked"
        elif label in failed:
            record.status = "failed"
        else:
            record.status = "untried"

    used_claim_labels = _hyp_text_labels(hyps)
    used_belief_labels = used_claim_labels | {
        normalize_label(h.technique or "") for h in hyps if h.technique
    }
    for tag in (t for h in hyps for t in h.tags):
        used_belief_labels.add(normalize_label(tag))

    beliefs_unused = []
    for belief in beliefs_all:
        tech = str(belief.get("technique") or belief.get("name") or "").strip()
        status = str(belief.get("status") or "").lower()
        if not tech:
            continue
        if status in {"deprecated", "contradicted", "rejected"}:
            failed.add(normalize_label(tech))
            continue
        if normalize_label(tech) in used_belief_labels:
            continue
        beliefs_unused.append(
            {
                "technique": tech,
                "status": status,
                "id": str(belief.get("id") or ""),
                "summary": str(belief.get("summary") or belief.get("effect") or ""),
            }
        )

    claims_unused = [
        c
        for c in claims_all
        if normalize_label(c["text"][:80]) not in used_claim_labels
        and normalize_label(c["text"][:80]) not in worked
    ]

    winning = _resolve_winning(hyps)
    winning_stack = list(winning.technique_stack) if winning else []
    if winning and winning.technique and winning.technique not in winning_stack:
        winning_stack = [*winning_stack, winning.technique]

    untried = [
        t.name
        for t in technique_index.values()
        if t.status == "untried" and t.label not in _META_TAGS and t.name.lower() not in _META_TAGS
    ]
    avoid_pairs: list[tuple[str, str]] = []
    for hyp in hyps:
        if hyp.status != HypothesisStatus.REJECTED:
            continue
        if hyp.technique and winning_stack:
            for parent_tech in winning_stack:
                avoid_pairs.append((parent_tech, hyp.technique))
        # Failed combination: avoid re-trying the same member pairs.
        combo = list(hyp.combo_techniques or [])
        if len(combo) >= 2:
            for a, b in combinations(combo, 2):
                avoid_pairs.append((a, b))
        elif hyp.technique and "+" in hyp.technique:
            parts = [p.strip() for p in hyp.technique.split("+") if p.strip()]
            if len(parts) >= 2:
                for a, b in combinations(parts, 2):
                    avoid_pairs.append((a, b))

    return ExperimentLedger(
        competition=competition,
        artifacts=artifacts,
        techniques=list(technique_index.values()),
        techniques_worked=sorted(
            {technique_index[k].name for k in worked if k in technique_index}
            | {h.technique for h in hyps if h.technique and normalize_label(h.technique) in worked}
        ),
        techniques_failed=sorted(
            {technique_index[k].name for k in failed if k in technique_index}
            | {
                h.technique
                for h in hyps
                if h.technique and normalize_label(h.technique) in failed
            }
        ),
        techniques_untried=untried,
        beliefs_unused=beliefs_unused,
        claims_unused=claims_unused[:50],
        winning_hypothesis_id=winning.id if winning else None,
        winning_stack=winning_stack,
        winning_confidence=float(winning.confidence) if winning else 0.5,
        winning_gain=float(winning.expected_impact or 0.0) if winning else 0.0,
        avoid_pairs=avoid_pairs,
        hypotheses=hyps,
    )


def _index_technique(
    index: dict[str, TechniqueRecord],
    name: str,
    *,
    artifact_id: str = "",
    category: str = "",
) -> None:
    label = normalize_label(name)
    if not label or label in _META_TAGS or name.strip().lower() in _META_TAGS:
        return
    # Checked against the *raw* name: `normalize_label` strips the colon these
    # prefixes depend on, so testing the normalised label — as this did — can
    # never match. That is why five `hyp:*` rows reached `techniques.name`.
    if is_record_reference(name):
        return
    existing = index.get(label)
    if existing is None:
        index[label] = TechniqueRecord(
            name=name.strip(),
            label=label,
            category=category,
            artifact_ids=[artifact_id] if artifact_id else [],
        )
        return
    if artifact_id and artifact_id not in existing.artifact_ids:
        existing.artifact_ids.append(artifact_id)
    if category and not existing.category:
        existing.category = category


def _technique_outcomes(hyps: list[Hypothesis]) -> tuple[set[str], set[str]]:
    worked: set[str] = set()
    failed: set[str] = set()
    for hyp in hyps:
        is_combo = bool(hyp.combo_techniques) or "combination" in {
            t.lower() for t in hyp.tags
        }
        if is_combo and hyp.status == HypothesisStatus.REJECTED:
            # Combo loss → avoid_pairs for members; do not blacklist each member.
            joined = normalize_label(hyp.technique or "")
            if joined:
                failed.add(joined)
            continue
        labels = {
            normalize_label(hyp.technique or ""),
            *[normalize_label(t) for t in hyp.technique_stack],
            *[normalize_label(t) for t in hyp.tags if t.lower() not in _META_TAGS],
        }
        if is_combo and hyp.combo_techniques:
            labels |= {normalize_label(t) for t in hyp.combo_techniques}
        labels.discard("")
        if hyp.status == HypothesisStatus.CONFIRMED:
            worked |= labels
        elif hyp.status == HypothesisStatus.REJECTED:
            failed |= labels
        elif hyp.status == HypothesisStatus.INCONCLUSIVE:
            outcome = (hyp.actual_outcome or "").lower()
            if "gain" in outcome or "improv" in outcome:
                worked |= labels
            elif "loss" in outcome or "overfit" in outcome or "worse" in outcome:
                if is_combo:
                    joined = normalize_label(hyp.technique or "")
                    if joined:
                        failed.add(joined)
                else:
                    failed |= labels
    return worked, failed


def _hyp_text_labels(hyps: list[Hypothesis]) -> set[str]:
    labels: set[str] = set()
    for hyp in hyps:
        for field in (hyp.observation, hyp.reason, hyp.prediction):
            if field:
                labels.add(normalize_label(field[:80]))
        for tag in hyp.tags:
            labels.add(normalize_label(tag))
        if hyp.technique:
            labels.add(normalize_label(hyp.technique))
    return labels


def _resolve_winning(hyps: list[Hypothesis]) -> Hypothesis | None:
    confirmed = [
        h
        for h in hyps
        if h.status == HypothesisStatus.CONFIRMED and h.id != BASELINE_HYPOTHESIS_ID
    ]
    if confirmed:
        return max(
            confirmed,
            key=lambda h: (
                float(h.public_score) if h.public_score is not None else -1.0,
                float(h.expected_impact or 0.0),
                float(h.confidence),
            ),
        )
    gained = [
        h
        for h in hyps
        if h.status in {HypothesisStatus.INCONCLUSIVE, HypothesisStatus.TESTING}
        and h.id != BASELINE_HYPOTHESIS_ID
        and (
            "gain" in (h.actual_outcome or "").lower()
            or "improv" in (h.actual_outcome or "").lower()
            or (h.expected_impact or 0.0) > 0
        )
    ]
    if gained:
        return max(gained, key=lambda h: (float(h.expected_impact or 0.0), h.confidence))
    return None
