"""Stage 2 Symbolic Retrieval + Stage 4 Evidence Expansion (SQLite joins).

No embeddings. Discussions are an empty stub until Plan F. Pipeline-diff
(similar pipelines → missing techniques) is deferred past Plan 9 v1.
"""

from __future__ import annotations

import json
import re
from typing import Any

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import ResearchArtifactType
from labpilot.research_engine.execution.technique.status_constants import (
    PLANNER_VISIBLE_STATUSES,
)
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryPlan,
    RetrievalHit,
    RetrievalIntent,
    SymbolicBundle,
)

_FAILURE_MARKERS = (
    "hurt",
    "hurts",
    "regress",
    "failed",
    "failure",
    "worse",
    "decreased",
    "drop",
    "negative",
)


class SymbolicFetcher:
    """Query knowledge.db for techniques and expand along evidence links."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def fetch(
        self,
        intent: RetrievalIntent,
        plan: QueryPlan,
        *,
        technique_ids: list[str] | None = None,
        expand: bool = True,
    ) -> SymbolicBundle:
        limits = plan.limits
        tech_limit = int(limits.get("techniques", 12))
        paper_limit = int(limits.get("papers", 8))
        exp_limit = int(limits.get("experiments", 8))
        repo_limit = int(limits.get("repositories", 5))
        fail_limit = int(limits.get("failures", 6))

        techniques = self._select_techniques(intent, limit=tech_limit)
        if technique_ids is not None:
            allowed = set(technique_ids)
            techniques = [row for row in techniques if str(row["id"]) in allowed]
            # Preserve caller order when restricting to survivors.
            order = {tid: index for index, tid in enumerate(technique_ids)}
            techniques.sort(key=lambda row: order.get(str(row["id"]), 10_000))

        bundle = SymbolicBundle(techniques=techniques)
        if not techniques:
            bundle.notes.append("symbolic: no techniques matched intent filters.")
            bundle.discussions = []
            bundle.notes.append("discussions: stub empty until Plan F.")
            return bundle

        if expand:
            for technique in techniques:
                tid = str(technique["id"])
                self._expand_technique(
                    bundle,
                    tid,
                    technique_name=str(technique.get("name") or tid),
                    intent=intent,
                    paper_limit=paper_limit,
                    exp_limit=exp_limit,
                    repo_limit=repo_limit,
                )

            bundle.papers = _dedupe_hits(bundle.papers)[:paper_limit]
            bundle.experiments = _dedupe_hits(bundle.experiments)[:exp_limit]
            bundle.repositories = _dedupe_hits(bundle.repositories)[:repo_limit]
            bundle.failures = self._collect_failures(techniques, bundle, limit=fail_limit)
        else:
            bundle.notes.append("symbolic: core selection only (expansion deferred).")

        bundle.discussions = []
        bundle.notes.append("discussions: stub empty until Plan F.")
        bundle.notes.append(
            f"symbolic: {len(techniques)} technique(s), "
            f"{len(bundle.papers)} paper(s), {len(bundle.experiments)} experiment(s), "
            f"{len(bundle.repositories)} repo(s), {len(bundle.failures)} failure(s)."
        )
        return bundle

    def _select_techniques(
        self, intent: RetrievalIntent, *, limit: int
    ) -> list[dict[str, Any]]:
        rows = self.store.list_techniques(domain=intent.domain, limit=None)
        if not rows:
            rows = self.store.list_techniques(limit=None)

        scored: list[tuple[float, dict[str, Any]]] = []
        pipeline = {normalize_label(item) for item in intent.current_pipeline}
        question_terms = _terms(intent.question)
        for row in rows:
            status = str(row.get("status") or "candidate")
            if status not in PLANNER_VISIBLE_STATUSES:
                continue
            score = float(row.get("confidence") or 0.5)
            axes: list[str] = []
            name = str(row.get("name") or "")
            domain = str(row.get("domain") or "")
            if intent.domain and domain and intent.domain.lower() in domain.lower():
                score += 0.2
                axes.append("domain")
            if normalize_label(name) in pipeline:
                score += 0.15
                axes.append("pipeline")
            if question_terms and any(term in name.lower() for term in question_terms):
                score += 0.1
                axes.append("question")
            meta = _parse_json(row.get("metadata"), {})
            aliases = [str(a) for a in meta.get("aliases", [])] if isinstance(meta, dict) else []
            if question_terms and any(
                any(term in alias.lower() for term in question_terms) for alias in aliases
            ):
                score += 0.05
                axes.append("alias")
            # Evidence richness bonus.
            n_evidence = len(self.store.artifacts_for_technique(str(row["id"])))
            score += min(0.2, 0.03 * n_evidence)
            enriched = dict(row)
            enriched["_axes"] = axes
            enriched["_score"] = min(1.0, score)
            scored.append((score, enriched))

        scored.sort(key=lambda item: (-item[0], str(item[1].get("name") or "").lower()))
        return [row for _score, row in scored[:limit]]

    def _expand_technique(
        self,
        bundle: SymbolicBundle,
        tid: str,
        *,
        technique_name: str,
        intent: RetrievalIntent,
        paper_limit: int,
        exp_limit: int,
        repo_limit: int,
    ) -> None:
        for artifact_id in self.store.artifacts_for_technique(tid):
            artifact = self.store.get_artifact(artifact_id)
            if artifact is None:
                continue
            hit = RetrievalHit(
                kind=_hit_kind(artifact.type),
                document_id=artifact.id,
                label=artifact.title or artifact.id,
                score=min(1.0, max(0.1, artifact.confidence)),
                axes_matched=["technique"],
                knowledge_ids=[tid],
                why=f"linked to {technique_name}",
                summary=(artifact.summary or "")[:240],
            )
            if artifact.type is ResearchArtifactType.PAPER and intent.need_papers:
                if len([h for h in bundle.papers if tid in h.knowledge_ids]) < paper_limit:
                    bundle.papers.append(hit)
            elif artifact.type is ResearchArtifactType.EXPERIMENT and intent.need_experiments:
                if len([h for h in bundle.experiments if tid in h.knowledge_ids]) < exp_limit:
                    bundle.experiments.append(hit)
            elif artifact.type is ResearchArtifactType.REPOSITORY and intent.need_repositories:
                if len([h for h in bundle.repositories if tid in h.knowledge_ids]) < repo_limit:
                    bundle.repositories.append(hit)

    def _collect_failures(
        self,
        techniques: list[dict[str, Any]],
        bundle: SymbolicBundle,
        *,
        limit: int,
    ) -> list[RetrievalHit]:
        failures: list[RetrievalHit] = []

        for belief in self.store.list_beliefs():
            status = str(belief.get("status") or "").lower()
            if status != "deprecated":
                continue
            failures.append(
                RetrievalHit(
                    kind="failure",
                    document_id=str(belief.get("id") or ""),
                    label=f"deprecated belief: {belief.get('technique')}",
                    score=0.7,
                    axes_matched=["belief"],
                    knowledge_ids=[],
                    why="belief status deprecated",
                    summary=str(belief.get("effect") or ""),
                )
            )

        for technique in techniques:
            known = str(technique.get("known_issues") or "").strip()
            if known:
                failures.append(
                    RetrievalHit(
                        kind="failure",
                        document_id=str(technique.get("id") or ""),
                        label=f"known issue: {technique.get('name')}",
                        score=0.55,
                        axes_matched=["technique"],
                        knowledge_ids=[str(technique.get("id") or "")],
                        why="technique known_issues",
                        summary=known[:240],
                    )
                )

        for hit in [*bundle.experiments, *bundle.papers]:
            text = f"{hit.label} {hit.summary} {hit.why}".lower()
            if any(marker in text for marker in _FAILURE_MARKERS):
                failures.append(
                    hit.model_copy(
                        update={
                            "kind": "failure",
                            "why": hit.why or "failure marker in evidence",
                        }
                    )
                )

        return _dedupe_hits(failures)[:limit]


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _terms(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token]


def _hit_kind(artifact_type: ResearchArtifactType) -> str:
    mapping = {
        ResearchArtifactType.PAPER: "paper",
        ResearchArtifactType.EXPERIMENT: "experiment",
        ResearchArtifactType.REPOSITORY: "repository",
        ResearchArtifactType.DISCUSSION: "discussion",
    }
    return mapping.get(artifact_type, "paper")


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: dict[str, RetrievalHit] = {}
    for hit in hits:
        key = hit.document_id or f"{hit.kind}:{hit.label}"
        existing = seen.get(key)
        if existing is None or hit.score > existing.score:
            if existing is not None:
                hit = hit.model_copy(
                    update={
                        "knowledge_ids": list(
                            dict.fromkeys([*existing.knowledge_ids, *hit.knowledge_ids])
                        )
                    }
                )
            seen[key] = hit
    return sorted(seen.values(), key=lambda item: (-item.score, item.label.lower()))


def _parse_json(raw: object, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not raw:
        return default
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return default
