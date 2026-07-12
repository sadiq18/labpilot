from labpilot.brief.context import render_competition_context
from labpilot.competition.models import CompetitionSpec, MetricSpec


def test_render_competition_context_includes_rules_fields():
    competition = CompetitionSpec(
        slug="test",
        title="Test Competition",
        evaluation_metric=MetricSpec(
            name="auc", direction="maximize", key="auc", description="AUC"
        ),
        deadline="2026-12-31T00:00:00",
        max_daily_submissions=5,
        submissions_disabled=False,
        is_kernels_submissions_only=True,
        tags=["tabular"],
        raw_html="Rule one.\nRule two.",
    )
    text = render_competition_context(competition)
    assert "## Competition Context" in text
    assert "Metric key:** auc" in text
    assert "Daily submission limit:** 5" in text
    assert "Kernels-only submissions:** yes" in text
    assert "Rule one." in text


def test_render_competition_context_graceful_with_missing_fields():
    competition = CompetitionSpec(slug="bare")
    text = render_competition_context(competition)
    assert "**Metric:** unknown" in text
    assert "**Deadline:** unknown" in text
