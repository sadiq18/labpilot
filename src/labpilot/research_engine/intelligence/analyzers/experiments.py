"""ExperimentAnalyzer — local M2 experiment memory → artifacts (design §3.4).

Fully local and deterministic: reads the M2 experiment graph and the
per-competition knowledge base **read-only** (no network, no LLM). Failures are
first-class (design success-criterion Q3) — ``HURTS`` technique/metric entries
surface as notes and are flagged on the experiments that produced them.
"""

from __future__ import annotations

from labpilot.research_engine.shared.experiments.graph import build_graph
from labpilot.research_engine.shared.experiments.knowledge import KnowledgeBase, normalize_technique
from labpilot.research_engine.shared.experiments.models import Experiment
from labpilot.research_engine.intelligence.analyzers.base import BaseAnalyzer
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)


def _experiment_techniques(exp: Experiment) -> list[str]:
    """Deterministic technique tags for one experiment (recipes + template)."""
    tags: list[str] = []
    seen: set[str] = set()
    for raw in [*exp.feature_recipes, exp.template_name or ""]:
        normalized = normalize_technique(raw) if raw else ""
        if normalized and normalized not in seen:
            seen.add(normalized)
            tags.append(normalized)
    return tags


class ExperimentAnalyzer(BaseAnalyzer):
    name = "experiments"
    default_enabled = True

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        graph = build_graph(
            context.runs_dir, context.competition, knowledge_dir=context.knowledge_dir
        )
        if not graph.nodes:
            return self._empty(f"No runs found for '{context.competition}'.")

        kb = KnowledgeBase(context.knowledge_dir, context.competition)
        failures = kb.known_failures(n=100)
        failed_by_run: dict[str, list[str]] = {}
        for entry in failures:
            for run_id in entry.evidence_run_ids:
                failed_by_run.setdefault(run_id, []).append(entry.technique)

        artifacts: list[ResearchArtifact] = []
        technique_rollup: list[str] = []
        seen_tech: set[str] = set()
        for exp in sorted(graph.nodes.values(), key=lambda e: e.created_at):
            techniques = _experiment_techniques(exp)
            for tag in techniques:
                if tag not in seen_tech:
                    seen_tech.add(tag)
                    technique_rollup.append(tag)

            regressions = failed_by_run.get(exp.id, [])
            artifacts.append(
                ResearchArtifact(
                    id=f"exp:{exp.id}",
                    type=ResearchArtifactType.EXPERIMENT,
                    source="m2",
                    title=exp.description or exp.id,
                    summary=f"{exp.description} [{exp.status}]".strip(),
                    techniques=techniques,
                    claims=[f"'{t}' regressed the primary metric" for t in regressions],
                    competition_slug=context.competition,
                    metadata={
                        "status": exp.status,
                        "iteration": exp.iteration,
                        "parent_id": exp.parent_id,
                        "metrics": exp.metrics,
                        "public_score": exp.public_score,
                        "runtime_seconds": exp.runtime_seconds,
                        "regressions": regressions,
                    },
                )
            )

        notes = [
            f"failure: '{entry.technique}' hurts {entry.metric_key} "
            f"(delta {entry.delta_estimate:+.4f}, n={entry.sample_size}, "
            f"confidence={entry.confidence:.2f})"
            for entry in failures
        ]
        return ResearchArtifacts(
            analyzer=self.name,
            items=artifacts,
            notes=notes,
            techniques=technique_rollup,
        )
