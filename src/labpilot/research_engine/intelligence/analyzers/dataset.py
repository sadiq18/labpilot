"""DatasetAnalyzer — deterministic dataset profile → artifact (design §3.4).

Reads the most recent local run's ``profile.json`` (Pandas/NumPy stats computed
upstream by the profiler). Never calls an LLM and never touches the network
(§2.4 Hard No: statistics and distributions are deterministic).
"""

from __future__ import annotations

from labpilot.research_engine.shared.experiments.graph import build_graph
from labpilot.accessor.profiler.report import load_profile
from labpilot.accessor.profiler.tabular import DatasetProfile
from labpilot.research_engine.intelligence.analyzers.base import BaseAnalyzer
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)

# A column is flagged "null-heavy" past this share of missing values.
_NULL_HEAVY_PCT = 20.0


class DatasetAnalyzer(BaseAnalyzer):
    name = "dataset"
    default_enabled = True

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        found = self._latest_profile(context)
        if found is None:
            return self._empty(
                f"No dataset profile found for '{context.competition}' "
                "(run the pipeline to produce profile.json first)."
            )
        run_id, profile = found

        null_heavy = sorted(c.name for c in profile.columns if c.null_pct >= _NULL_HEAVY_PCT)
        summary = (
            f"{profile.modality} dataset — {profile.row_count} train rows, "
            f"{profile.column_count} columns, target={profile.target_column or 'unknown'}"
        )
        artifact = ResearchArtifact(
            id=f"dataset:{context.competition}",
            type=ResearchArtifactType.DATASET,
            source="m2",
            title=f"{context.competition} dataset profile",
            summary=summary,
            datasets=[profile.competition or context.competition],
            competition_slug=context.competition,
            metadata={
                "run_id": run_id,
                "modality": profile.modality,
                "row_count": profile.row_count,
                "test_row_count": profile.test_row_count,
                "column_count": profile.column_count,
                "target_column": profile.target_column,
                "id_column": profile.id_column,
                "null_heavy_columns": null_heavy,
                "warnings": profile.warnings,
                # Partition facts drive validation design downstream: rows in a
                # partitioned dataset are not iid, so a shuffled row-level split
                # scores a near-duplicate of every training row.
                "partitioned": profile.partitioned,
                "partition_key": profile.partition_key,
                "partition_kinds": profile.partition_kinds,
                "train_partition_count": profile.train_partition_count,
                "test_partition_count": profile.test_partition_count,
                "row_count_estimated": profile.row_count_estimated,
            },
        )

        notes = list(profile.warnings)
        if null_heavy:
            notes.append(
                f"{len(null_heavy)} column(s) >={_NULL_HEAVY_PCT:g}% null: {', '.join(null_heavy)}"
            )
        return ResearchArtifacts(analyzer=self.name, items=[artifact], notes=notes)

    def _latest_profile(self, context: AnalyzeContext) -> tuple[str, DatasetProfile] | None:
        graph = build_graph(
            context.runs_dir, context.competition, knowledge_dir=context.knowledge_dir
        )
        for exp in sorted(graph.nodes.values(), key=lambda e: e.created_at, reverse=True):
            run_dir = context.runs_dir / exp.id
            if not (run_dir / "profile.json").is_file():
                continue
            try:
                return exp.id, load_profile(run_dir)
            except (OSError, ValueError):
                continue
        return self._profile_raw_data(context)

    def _profile_raw_data(self, context: AnalyzeContext) -> tuple[str, DatasetProfile] | None:
        """Profile ``data/raw`` directly when no prior run has produced one.

        Without this, a fresh workspace analyzes (and therefore plans) with no
        knowledge of the data at all — the profile only appeared after a run,
        which is exactly backwards from what planning needs.
        """
        data_dir = context.data_dir
        if data_dir is None or not data_dir.is_dir():
            return None
        from labpilot.config import ProfilerConfig
        from labpilot.accessor.profiler.tabular import TabularProfiler

        try:
            profile = TabularProfiler(ProfilerConfig()).profile_directory(
                data_dir, context.competition
            )
        except Exception as exc:  # noqa: BLE001 — analysis degrades, never aborts
            self._raw_profile_error = str(exc)
            return None
        return "raw", profile
