import json
from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionMetadata, CompetitionSpec
from labpilot.research_engine.intelligence.competition.parser import CompetitionParser
from labpilot.research_engine.intelligence.competition.submission_mode import apply_submission_mode, detect_kernel_only_from_rules


class FakeFetcher:
    def __init__(self, metadata: CompetitionMetadata | None) -> None:
        self.metadata = metadata

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return self.metadata


def test_parser_sets_submission_mode_kernel(tmp_path: Path):
    fetcher = FakeFetcher(
        CompetitionMetadata(
            slug="aerial-cactus",
            title="Aerial Cactus",
            is_kernels_submissions_only=True,
        )
    )
    spec = CompetitionParser(
        "aerial-cactus", configs_dir=tmp_path, metadata_fetcher=fetcher
    ).parse()

    assert spec.submission_mode == "kernel"
    assert spec.is_kernels_submissions_only is True
    assert "aerial-cactus/submissions" in spec.submissions_url


def test_parser_sets_submission_mode_csv(tmp_path: Path):
    fetcher = FakeFetcher(
        CompetitionMetadata(slug="titanic", title="Titanic", is_kernels_submissions_only=False)
    )
    spec = CompetitionParser("titanic", configs_dir=tmp_path, metadata_fetcher=fetcher).parse()

    assert spec.submission_mode == "csv"


def test_rules_fallback_detects_kernel_only():
    assert detect_kernel_only_from_rules("This is a code competition with notebook submission.")
    assert not detect_kernel_only_from_rules("Submit a CSV file to the leaderboard.")


def test_apply_submission_mode_rules_fallback():
    spec = CompetitionSpec(
        slug="mystery",
        raw_html="Kernels only submissions are required for this code competition.",
    )
    updated = apply_submission_mode(spec)
    assert updated.submission_mode == "kernel"
    assert updated.is_kernels_submissions_only is True


def test_local_yaml_can_override_submission_mode(tmp_path: Path):
    (tmp_path / "custom.yaml").write_text(
        "title: Custom\nsubmission_mode: kernel\nis_kernels_submissions_only: true\n"
    )
    fetcher = FakeFetcher(CompetitionMetadata(slug="custom", is_kernels_submissions_only=False))
    spec = CompetitionParser("custom", configs_dir=tmp_path, metadata_fetcher=fetcher).parse()
    assert spec.submission_mode == "kernel"
