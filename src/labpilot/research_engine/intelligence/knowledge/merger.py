"""Merge flagship — collapse alias mentions into one concept per knowledge unit.

Grouping is deterministic (normalized keys, a small alias seed table, then a
conservative containment pass) so the hub works with the Micro Agent disabled.
When a ``ConceptNormalizerAgent`` is available it only chooses the canonical
label for an already-formed cluster; it never invents or splits clusters.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.feature_recipes import (
    FEATURE_ENGINEERING_CATEGORY,
    looks_like_feature_engineering,
)
from labpilot.research_engine.intelligence.knowledge.extractor import ConceptCandidate
from labpilot.research_engine.intelligence.knowledge.models import EntityType, EvidenceRef
from labpilot.research_engine.intelligence.micro_agents.concept_normalizer import (
    ConceptNormalizerAgent,
)

logger = logging.getLogger("labpilot.research_engine.intelligence.knowledge.merger")

#: Well-known families where string similarity alone cannot merge variants.
#: Values are normalized keys (see :func:`normalize_key`).
ALIAS_SEEDS: dict[str, tuple[str, ...]] = {
    "SpecAugment": (
        "specaugment",
        "specaug",
        "spectrogram augmentation",
        "time masking",
        "frequency masking",
        "freq masking",
        "time frequency masking",
    ),
    "Mixup": ("mixup", "mix up", "input mixup"),
    "CutMix": ("cutmix", "cut mix"),
    "EMA": ("ema", "exponential moving average", "weight averaging ema"),
    "SWA": ("swa", "stochastic weight averaging"),
    "Mixed Precision": ("amp", "mixed precision", "automatic mixed precision", "fp16"),
    "Focal Loss": ("focal loss", "focalloss"),
    "Test-Time Augmentation": ("tta", "test time augmentation", "test-time augmentation"),
    "Label Smoothing": ("label smoothing", "smoothed labels"),
}

# Containment merging below this key length is unsafe ("ema" inside "schema").
_MIN_CONTAINMENT_LEN = 6


def normalize_key(name: str) -> str:
    """Case/punctuation-insensitive key used to group mentions."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.strip().lower())
    return " ".join(cleaned.split())


def _build_seed_index() -> dict[str, str]:
    """Index seeds under spaced and de-spaced keys ("spec augment"/"specaugment")."""
    index: dict[str, str] = {}
    for canonical, variants in ALIAS_SEEDS.items():
        for variant in (*variants, canonical):
            key = normalize_key(variant)
            index.setdefault(key, canonical)
            index.setdefault(key.replace(" ", ""), canonical)
    return index


_SEED_BY_KEY: dict[str, str] = _build_seed_index()


def _seed_canonical(key: str) -> str | None:
    return _SEED_BY_KEY.get(key) or _SEED_BY_KEY.get(key.replace(" ", ""))


class ConceptCluster(BaseModel):
    """One merged concept: canonical name, aliases, and all supporting evidence."""

    entity_type: EntityType
    canonical: str
    aliases: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    normalized_by: str = "rule_engine"
    category: str = ""

    @property
    def mention_count(self) -> int:
        return len(self.evidence)


class KnowledgeMerger:
    """Group concept candidates into clusters, one per canonical concept."""

    def __init__(self, *, llm_client: object | None = None) -> None:
        self.llm_client = llm_client

    def merge(self, candidates: list[ConceptCandidate]) -> list[ConceptCluster]:
        clusters: list[ConceptCluster] = []
        by_entity: dict[EntityType, list[ConceptCandidate]] = {}
        for candidate in candidates:
            by_entity.setdefault(candidate.entity_type, []).append(candidate)
        for entity_type, group in by_entity.items():
            clusters.extend(self._merge_one_entity(entity_type, group))
        return sorted(clusters, key=lambda c: (-c.mention_count, c.canonical.lower()))

    def _merge_one_entity(
        self,
        entity_type: EntityType,
        candidates: list[ConceptCandidate],
    ) -> list[ConceptCluster]:
        # 1. Bucket by seed canonical when known, else by normalized key.
        buckets: dict[str, list[ConceptCandidate]] = {}
        seeded: set[str] = set()
        for candidate in candidates:
            key = normalize_key(candidate.name)
            if not key:
                continue
            seed = _seed_canonical(key)
            bucket_key = normalize_key(seed) if seed else key
            if seed:
                seeded.add(bucket_key)
            buckets.setdefault(bucket_key, []).append(candidate)

        # 2. Conservative containment pass: fold longer keys into shorter ones.
        for key in sorted(buckets, key=len):
            if key not in buckets or len(key) < _MIN_CONTAINMENT_LEN:
                continue
            for other in [k for k in buckets if k != key]:
                if other not in buckets or key not in buckets:
                    continue
                if len(other) <= len(key):
                    continue
                if key in other:
                    buckets[key].extend(buckets.pop(other))
                    seeded.discard(other)

        clusters: list[ConceptCluster] = []
        for key, group in buckets.items():
            variants = [candidate.name for candidate in group]
            canonical, normalized_by, category = self._canonical_name(
                key, variants, seeded=key in seeded
            )
            # Aliases keep every surface form that differs from the canonical
            # label, including case-only variants, so lookups stay exhaustive.
            aliases = [variant for variant in dict.fromkeys(variants) if variant != canonical]
            clusters.append(
                ConceptCluster(
                    entity_type=entity_type,
                    canonical=canonical,
                    aliases=aliases,
                    evidence=_dedupe_evidence([c.evidence for c in group]),
                    normalized_by=normalized_by,
                    category=category,
                )
            )
        return clusters

    def _canonical_name(
        self,
        key: str,
        variants: list[str],
        *,
        seeded: bool,
    ) -> tuple[str, str, str]:
        """Return ``(canonical, normalized_by, category)`` for one cluster."""
        if seeded:
            for canonical in ALIAS_SEEDS:
                if normalize_key(canonical) == key:
                    return canonical, "alias_seed", _fe_category(canonical, variants)
        distinct = list(dict.fromkeys(variants))
        if len(distinct) > 1 and self.llm_client is not None:
            named = self._ask_agent(distinct)
            if named is not None:
                canonical, by, category = named
                return canonical, by, category or _fe_category(canonical, variants)
        canonical = _most_common_variant(distinct, variants)
        return canonical, "rule_engine", _fe_category(canonical, variants)

    def _ask_agent(self, distinct: list[str]) -> tuple[str, str, str] | None:
        agent = ConceptNormalizerAgent(llm_client=self.llm_client)
        try:
            result = agent.run(StructuredContext(items=distinct))
        except Exception as exc:  # normalization is an optional upgrade
            logger.debug("concept normalization failed: %s", exc)
            return None
        canonical = str(getattr(result, "canonical", "")).strip()
        # Guard: the agent may only relabel within the cluster it was given.
        if not canonical or normalize_key(canonical) not in {
            normalize_key(variant) for variant in distinct
        }:
            return None
        category = str(getattr(result, "category", "")).strip()
        return canonical, "llm" if agent.last_used_llm else "rule_engine", category


def _most_common_variant(distinct: list[str], variants: list[str]) -> str:
    counts = Counter(variants)
    return sorted(distinct, key=lambda name: (-counts[name], len(name), name))[0]


def _fe_category(canonical: str, variants: list[str] | None = None) -> str:
    haystack = " ".join([canonical, *(variants or [])])
    if looks_like_feature_engineering(haystack):
        return FEATURE_ENGINEERING_CATEGORY
    return ""


def _dedupe_evidence(refs: list[EvidenceRef]) -> list[EvidenceRef]:
    merged: dict[str, EvidenceRef] = {}
    for ref in refs:
        existing = merged.get(ref.artifact_id)
        if existing is None:
            merged[ref.artifact_id] = ref.model_copy(deep=True)
            continue
        if ref.weight > existing.weight:
            existing.weight = ref.weight
    return sorted(merged.values(), key=lambda ref: ref.artifact_id)
