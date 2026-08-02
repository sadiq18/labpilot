"""Stage 2 FacetPipeline / merger / extractors."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.memory.facets import (
    FacetContext,
    FacetPipeline,
    log_confidence_histogram,
    merge_facet_hits,
)
from labpilot.research_engine.memory.facets.code import CodeExtractor
from labpilot.research_engine.memory.facets.dataset import DatasetExtractor
from labpilot.research_engine.memory.facets.metadata import MetadataExtractor
from labpilot.research_engine.memory.facets.result import ResultExtractor
from labpilot.research_engine.memory.facets.rules import RulesExtractor
from labpilot.research_engine.memory.models import ExperienceFacet


def test_merge_max_confidence_source_priority_and_evidence_cap() -> None:
    hits = [
        ExperienceFacet(
            facet="audio",
            confidence=0.45,
            evidence=["bird"],
            source="rules",
        ),
        ExperienceFacet(
            facet="audio",
            confidence=0.8,
            evidence=["librosa", "melspectrogram"],
            source="code",
        ),
        ExperienceFacet(
            facet="audio",
            confidence=0.5,
            evidence=[f"e{i}" for i in range(10)],
            source="dataset",
        ),
    ]
    merged = merge_facet_hits(hits)
    assert len(merged) == 1
    assert merged[0].confidence == 0.8
    assert merged[0].source == "code"
    assert len(merged[0].evidence) <= 8
    assert "librosa" in merged[0].evidence


def test_pipeline_soft_fails_extractor() -> None:
    class Boom:
        name = "boom"

        def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
            del ctx
            raise RuntimeError("boom")

    pipeline = FacetPipeline(
        extractors=[
            Boom(),
            MetadataExtractor(),
        ]
    )
    facets = pipeline.extract(
        FacetContext(
            competition="x",
            payload={"problem_type": "audio"},
        )
    )
    assert any(f.facet == "audio" and f.source == "metadata" for f in facets)


def test_pipeline_all_extractors_fail_returns_empty() -> None:
    class Boom:
        name = "boom"

        def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
            del ctx
            raise RuntimeError("boom")

    pipeline = FacetPipeline(extractors=[Boom(), Boom()])
    assert pipeline.extract(FacetContext(competition="x")) == []


def test_merge_normalizes_facet_casing() -> None:
    merged = merge_facet_hits(
        [
            ExperienceFacet(
                facet="Audio",
                confidence=0.5,
                evidence=["A"],
                source="rules",
            ),
            ExperienceFacet(
                facet="AUDIO",
                confidence=0.9,
                evidence=["B"],
                source="code",
            ),
        ]
    )
    assert len(merged) == 1
    assert merged[0].facet == "audio"
    assert merged[0].source == "code"


def test_merge_reserves_evidence_slots_across_sources() -> None:
    """High-conf hit cannot fill all 8 slots alone — lower sources still contribute."""
    high = ExperienceFacet(
        facet="audio",
        confidence=0.9,
        evidence=[f"hi-{i}" for i in range(10)],
        source="code",
    )
    low = ExperienceFacet(
        facet="audio",
        confidence=0.4,
        evidence=["critical-low-signal"],
        source="rules",
    )
    merged = merge_facet_hits([high, low])
    assert "critical-low-signal" in merged[0].evidence
    assert len(merged[0].evidence) <= 8


def test_code_extractor_from_pipeline_train(tmp_path: Path) -> None:
    pipe = tmp_path / "pipeline"
    pipe.mkdir()
    (pipe / "train.py").write_text(
        "import librosa\nfrom torchaudio.transforms import MelSpectrogram\n",
        encoding="utf-8",
    )
    hits = CodeExtractor().extract(
        FacetContext(competition="bird", workspace_path=tmp_path)
    )
    by = {h.facet: h for h in hits}
    assert "audio" in by
    assert by["audio"].source == "code"
    assert "librosa" in by["audio"].evidence


def test_dataset_extractor_extensions(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.wav").write_bytes(b"RIFF")
    hits = DatasetExtractor().extract(
        FacetContext(competition="bird", workspace_path=tmp_path)
    )
    assert any(h.facet == "audio" and ".wav" in h.evidence for h in hits)


def test_result_extractor_from_reflection() -> None:
    hits = ResultExtractor().extract(
        FacetContext(
            competition="x",
            reflection={
                "observation": "SpecAugment helps minority classes",
                "likely_cause": "imbalance",
            },
        )
    )
    names = {h.facet for h in hits}
    assert "augmentation" in names or "imbalance" in names


def test_rules_and_metadata_still_work() -> None:
    meta = MetadataExtractor().extract(
        FacetContext(competition="x", payload={"problem_type": "audio"})
    )
    assert meta[0].source == "metadata"
    rules = RulesExtractor().extract(
        FacetContext(
            competition="birdclef-2026",
            payload={"description": "bird sound spectrogram"},
        )
    )
    assert any(h.facet == "audio" for h in rules)


def test_confidence_histogram_bands() -> None:
    facets = [
        ExperienceFacet(facet="a", confidence=0.4, evidence=["x"], source="rules"),
        ExperienceFacet(facet="b", confidence=0.6, evidence=["y"], source="rules"),
        ExperienceFacet(facet="c", confidence=0.9, evidence=["z"], source="metadata"),
    ]
    summary = log_confidence_histogram(facets, competition="demo")
    assert summary["low"] == 1
    assert summary["mid"] == 1
    assert summary["high"] == 1
