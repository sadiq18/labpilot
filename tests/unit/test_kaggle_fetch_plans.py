"""A campaign's Kaggle sweep is one call, not three.

`research analyze` pulls vote-sorted kernels, score-sorted kernels, and
discussions. A campaign runs the same tool every few steps, and each call is a
Kaggle round trip per item plus a Knowledge Hub ingest — so it takes the
kernels that actually scored and skips the rest.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.intelligence.models import AnalysisReport, AnalyzeContext
from labpilot.research_engine.intelligence.orchestrator import (
    KAGGLE_FETCH_PLANS,
    AnalyzeOrchestrator,
)
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry


class _RecordingFetchService:
    def __init__(self):
        self.calls = []

    def fetch(self, competition, *, sources, knowledge_dir, refresh, **kwargs):
        self.calls.append((sorted(sources), kwargs))

        class _Result:
            artifact_ids: list[str] = []
            written = 0
            skipped_existing = 0
            fetched = 0
            notes: list[str] = []

        return _Result()


def _run(plan, tmp_path):
    service = _RecordingFetchService()
    orchestrator = AnalyzeOrchestrator(
        AnalyzerRegistry(),
        fetch_kaggle=True,
        kaggle_fetch_plan=plan,
        kaggle_fetch_service=service,
    )
    context = AnalyzeContext(competition="demo", knowledge_dir=tmp_path, runs_dir=tmp_path / "runs")
    orchestrator._fetch_kaggle_run(AnalysisReport(competition={"slug": "demo"}), context)
    return service.calls


def test_best_score_makes_exactly_one_call(tmp_path):
    calls = _run("best_score", tmp_path)

    assert len(calls) == 1
    sources, kwargs = calls[0]
    assert sources == ["kernels"]
    assert kwargs["kernel_sort"] == "scoreDescending"


def test_best_score_fetches_no_discussions(tmp_path):
    """The saving is dropping calls, not shrinking one."""
    assert all(sources == ["kernels"] for sources, _ in _run("best_score", tmp_path))


def test_the_default_plan_is_unchanged(tmp_path):
    """`research analyze` keeps all three — this is the campaign's budget, not
    a claim that the other two are worthless."""
    calls = _run("all", tmp_path)

    assert [sources for sources, _ in calls] == [["kernels"], ["kernels"], ["discussions"]]
    assert [kwargs.get("kernel_sort") for _, kwargs in calls[:2]] == [
        "voteCount",
        "scoreDescending",
    ]


def test_an_unknown_plan_is_refused_at_construction():
    """A typo must not silently degrade to fetching nothing — a campaign would
    read that as 'no evidence exists' rather than 'nothing was asked for'."""
    with pytest.raises(ValueError, match="kaggle_fetch_plan"):
        AnalyzeOrchestrator(AnalyzerRegistry(), kaggle_fetch_plan="best-score")


def test_every_plan_names_only_real_sources():
    for plan, calls in KAGGLE_FETCH_PLANS.items():
        for sources, _kwargs in calls:
            assert sources <= {"kernels", "discussions"}, plan
