"""Per-competition knowledge base (Milestone 2, Plan 5).

Comparator-driven updates are the primary signal. Optional reflection enrichment
adds UNKNOWN, low-confidence entries for technique tags not yet seen from
comparisons.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from labpilot.experiments.models import (
    ConfigChange,
    ExperimentComparison,
    KnowledgeEffect,
    KnowledgeEntry,
    StructuredReflection,
)

logger = logging.getLogger(__name__)

# Confidence / effect heuristics — named so they can be revisited without a
# design change (Plan 5).
_CONF_PER_SAMPLE = 0.1
_CONF_BASE = 0.5
_CONF_CAP = 0.95
_CONF_PENALTY = 0.15
_CONF_FLOOR = 0.3
_REFLECTION_CONF_CAP = 0.4
_EFFECT_EPSILON = 0.001

_WHITESPACE = re.compile(r"\s+")


def normalize_technique(name: str) -> str:
    """Lowercase + collapse whitespace for stable matching."""
    return _WHITESPACE.sub(" ", name.strip().lower())


def technique_tags_from_changes(changes: list[ConfigChange]) -> list[str]:
    """Derive technique tags from config changes (field / recipe short names)."""
    tags: list[str] = []
    seen: set[str] = set()
    for change in changes:
        raw: str | None = None
        if change.field == "feature_recipes":
            value = change.compare_value if change.compare_value is not None else change.base_value
            if value is not None:
                raw = str(value)
        elif change.field.startswith("model_params."):
            raw = change.field.removeprefix("model_params.")
        elif change.field in {"template_name", "problem_type"}:
            raw = change.field
        else:
            raw = change.field
        if not raw:
            continue
        normalized = normalize_technique(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(normalized)
    return tags


def _now() -> datetime:
    return datetime.now()


def _effect_from_delta(delta: float, *, epsilon: float = _EFFECT_EPSILON) -> KnowledgeEffect:
    if delta > epsilon:
        return KnowledgeEffect.IMPROVES
    if delta < -epsilon:
        return KnowledgeEffect.HURTS
    return KnowledgeEffect.NEUTRAL


class KnowledgeBase:
    """File-backed knowledge store at `knowledge/<slug>/knowledge_base.json`."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.path = self.knowledge_dir / competition / "knowledge_base.json"
        self._entries: dict[tuple[str, str], KnowledgeEntry] = {}
        self._load()

    def _load(self) -> None:
        self._entries = {}
        if not self.path.is_file():
            return
        try:
            raw = self.path.read_text()
            if not raw.strip():
                return
            payload = json.loads(raw)
            items = payload.get("entries", payload) if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                return
            for item in items:
                entry = KnowledgeEntry.model_validate(item)
                key = (normalize_technique(entry.technique), entry.metric_key)
                self._entries[key] = entry.model_copy(
                    update={"technique": normalize_technique(entry.technique)}
                )
        except (OSError, ValueError) as exc:
            logger.warning("Could not load knowledge base %s: %s", self.path, exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(
            self._entries.values(),
            key=lambda e: (e.technique, e.metric_key),
        )
        payload = {
            "competition": self.competition,
            "entries": [e.model_dump(mode="json") for e in entries],
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    def get(self, technique: str, metric_key: str) -> KnowledgeEntry | None:
        return self._entries.get((normalize_technique(technique), metric_key))

    def list_entries(
        self,
        *,
        technique: str | None = None,
        effect: KnowledgeEffect | None = None,
    ) -> list[KnowledgeEntry]:
        items = list(self._entries.values())
        if technique is not None:
            needle = normalize_technique(technique)
            items = [e for e in items if e.technique == needle]
        if effect is not None:
            items = [e for e in items if e.effect == effect]
        return sorted(items, key=lambda e: (e.technique, e.metric_key))

    def top_discoveries(self, n: int = 5) -> list[KnowledgeEntry]:
        improves = [e for e in self._entries.values() if e.effect == KnowledgeEffect.IMPROVES]
        improves.sort(key=lambda e: e.delta_estimate * e.confidence, reverse=True)
        return improves[:n]

    def known_failures(self, n: int = 5) -> list[KnowledgeEntry]:
        hurts = [e for e in self._entries.values() if e.effect == KnowledgeEffect.HURTS]
        hurts.sort(key=lambda e: e.delta_estimate * e.confidence)
        return hurts[:n]

    def update_from_comparison(
        self,
        comparison: ExperimentComparison,
        *,
        maximize: bool = True,
        technique_tags: list[str] | None = None,
    ) -> list[KnowledgeEntry]:
        """Update entries from a comparator result. Returns updated entries."""
        metric_key = comparison.primary_metric_key
        if metric_key is None or metric_key not in comparison.metric_deltas:
            return []

        raw_delta = comparison.metric_deltas[metric_key]
        signed = float(raw_delta) if maximize else -float(raw_delta)
        tags = technique_tags if technique_tags is not None else technique_tags_from_changes(
            comparison.changes
        )
        if not tags:
            return []

        run_id = comparison.compare_id
        updated: list[KnowledgeEntry] = []
        for tag in tags:
            entry = self._update_from_delta(
                technique=tag,
                metric_key=metric_key,
                delta=signed,
                run_id=run_id,
            )
            updated.append(entry)
        self._save()
        return updated

    def update_from_reflection(
        self,
        reflection: StructuredReflection,
        *,
        metric_key: str,
    ) -> list[KnowledgeEntry]:
        """Add UNKNOWN low-confidence entries for draft tags not yet in the KB.

        Tags already covered by a comparator-derived entry (any effect other
        than UNKNOWN with sample evidence) are skipped — they are corroborated.
        """
        if reflection.generated_by != "llm":
            return []

        tags: list[str] = []
        seen: set[str] = set()
        for draft in reflection.new_hypotheses:
            for tag in draft.tags:
                normalized = normalize_technique(tag)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tags.append(normalized)
        if not tags or not metric_key:
            return []

        updated: list[KnowledgeEntry] = []
        for tag in tags:
            key = (tag, metric_key)
            existing = self._entries.get(key)
            if existing is not None and existing.effect != KnowledgeEffect.UNKNOWN:
                # Already corroborated by comparator signal — leave alone.
                continue
            if existing is not None and existing.effect == KnowledgeEffect.UNKNOWN:
                # Refresh evidence + modest confidence bump, still capped.
                evidence = list(existing.evidence_run_ids)
                if reflection.run_id not in evidence:
                    evidence.append(reflection.run_id)
                conf = min(
                    _REFLECTION_CONF_CAP,
                    max(existing.confidence, min(reflection.confidence, _REFLECTION_CONF_CAP)),
                )
                entry = existing.model_copy(
                    update={
                        "sample_size": existing.sample_size + 1,
                        "evidence_run_ids": evidence,
                        "confidence": conf,
                        "updated_at": _now(),
                    }
                )
            else:
                entry = KnowledgeEntry(
                    technique=tag,
                    metric_key=metric_key,
                    effect=KnowledgeEffect.UNKNOWN,
                    delta_estimate=0.0,
                    confidence=min(_REFLECTION_CONF_CAP, max(0.0, reflection.confidence)),
                    sample_size=1,
                    evidence_run_ids=[reflection.run_id],
                    updated_at=_now(),
                )
            self._entries[key] = entry
            updated.append(entry)

        if updated:
            self._save()
        return updated

    def _update_from_delta(
        self,
        *,
        technique: str,
        metric_key: str,
        delta: float,
        run_id: str,
    ) -> KnowledgeEntry:
        key = (normalize_technique(technique), metric_key)
        existing = self._entries.get(key)
        # Idempotent: same run_id must not inflate sample_size when comparison
        # is rewritten (finally + write_reflection).
        if existing is not None and run_id in existing.evidence_run_ids:
            return existing

        n = (existing.sample_size if existing else 0) + 1
        prior_avg = existing.delta_estimate if existing else 0.0
        new_avg = prior_avg + (delta - prior_avg) / n
        consistent = existing is None or (prior_avg >= 0) == (delta >= 0)
        if consistent:
            confidence = min(_CONF_CAP, _CONF_BASE + _CONF_PER_SAMPLE * n)
        else:
            prior_conf = existing.confidence if existing else _CONF_BASE
            confidence = max(_CONF_FLOOR, prior_conf - _CONF_PENALTY)

        evidence = list(existing.evidence_run_ids) if existing else []
        evidence.append(run_id)

        entry = KnowledgeEntry(
            technique=key[0],
            metric_key=metric_key,
            effect=_effect_from_delta(new_avg),
            delta_estimate=new_avg,
            confidence=confidence,
            sample_size=n,
            evidence_run_ids=evidence,
            updated_at=_now(),
        )
        self._entries[key] = entry
        return entry
