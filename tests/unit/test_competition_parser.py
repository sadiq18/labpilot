from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionMetadata
from labpilot.research_engine.intelligence.competition.parser import CompetitionParser


class FakeFetcher:
    def __init__(self, metadata: CompetitionMetadata | None) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        self.calls.append(competition)
        return self.metadata


class RaisingFetcher:
    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        raise RuntimeError("no network in this test")


def test_local_file_wins_over_auto_resolution(tmp_path: Path):
    (tmp_path / "titanic.yaml").write_text("title: Titanic\nproblem_type: tabular_classification\n")
    fetcher = FakeFetcher(CompetitionMetadata(slug="titanic", title="Should not be used"))

    spec = CompetitionParser("titanic", configs_dir=tmp_path, metadata_fetcher=fetcher).parse()

    assert spec.title == "Titanic"
    assert fetcher.calls == []


def test_auto_resolves_metadata_when_no_local_file_exists(tmp_path: Path):
    fetcher = FakeFetcher(
        CompetitionMetadata(
            slug="house-prices",
            title="House Prices",
            description="Predict sale prices.",
            category="Getting Started",
            evaluation_metric_raw="Root-Mean-Squared-Error (RMSE)",
        )
    )

    spec = CompetitionParser("house-prices", configs_dir=tmp_path, metadata_fetcher=fetcher).parse()

    assert fetcher.calls == ["house-prices"]
    assert spec.title == "House Prices"
    assert spec.description == "Predict sale prices."
    assert spec.tags == ["Getting Started"]
    assert spec.evaluation_metric is not None
    assert spec.evaluation_metric.direction == "minimize"
    # RMSE in competition metadata maps to tabular_regression when no local YAML
    # sets problem_type (profiler can still refine later from data).
    assert spec.problem_type.value == "tabular_regression"


def test_falls_back_to_bare_contract_when_fetch_raises(tmp_path: Path):
    spec = CompetitionParser(
        "mystery-competition", configs_dir=tmp_path, metadata_fetcher=RaisingFetcher()
    ).parse()

    assert spec.slug == "mystery-competition"
    assert spec.title == ""
    assert spec.problem_type.value == "unknown"


def test_falls_back_to_bare_contract_without_a_fetcher(tmp_path: Path):
    spec = CompetitionParser("mystery-competition", configs_dir=tmp_path).parse()

    assert spec.slug == "mystery-competition"
    assert spec.data_url == "https://www.kaggle.com/competitions/mystery-competition/data"


def test_auto_resolution_with_no_metadata_found_returns_bare_contract(tmp_path: Path):
    fetcher = FakeFetcher(None)

    spec = CompetitionParser(
        "mystery-competition", configs_dir=tmp_path, metadata_fetcher=fetcher
    ).parse()

    assert fetcher.calls == ["mystery-competition"]
    assert spec.slug == "mystery-competition"
    assert spec.title == ""
