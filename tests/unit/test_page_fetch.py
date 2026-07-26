"""Unit tests for competition page fetch (Plan 5b — Kaggle pages API)."""

from __future__ import annotations

from types import SimpleNamespace

from labpilot.research_engine.intelligence.competition.page_fetch import (
    fetch_competition_pages,
    looks_like_empty_shell,
    page_content_to_text,
    pages_from_api_payload,
)


def test_pages_from_api_payload_splits_overview_and_rules():
    pages = [
        SimpleNamespace(
            name="Description",
            content="Track cells in 3D microscopy over time.",
            mime_type="text/markdown",
        ),
        SimpleNamespace(
            name="Evaluation",
            content="score = adjusted_edge_jaccard + 0.1 * division_jaccard",
            mime_type="",
        ),
        SimpleNamespace(
            name="Code Requirements",
            content="CPU Notebook <= 12 hours. Internet disabled.",
            mime_type="",
        ),
        SimpleNamespace(
            name="rules",
            content="External data is not permitted. You must accept these rules.",
            mime_type="",
        ),
        SimpleNamespace(name="Prizes", content="1st Place - $18,000", mime_type=""),
    ]
    overview, rules = pages_from_api_payload(pages)
    assert "Track cells" in overview
    assert "adjusted_edge_jaccard" in overview
    assert "Internet disabled" in overview
    assert "External data is not permitted" in rules
    assert "18,000" not in overview
    assert "18,000" not in rules


def test_page_content_to_text_strips_html():
    html = "<html><body><h1>Eval</h1><p>Use F1 score.</p><script>x()</script></body></html>"
    text = page_content_to_text(html, mime_type="text/html")
    assert "F1 score" in text
    assert "script" not in text.lower() or "x()" not in text


def test_fetch_competition_pages_prefers_api(tmp_path):
    def fake_list(_slug: str):
        return [
            SimpleNamespace(
                name="Description",
                content="A" * 250,
                mime_type="",
            ),
            SimpleNamespace(
                name="rules",
                content="B" * 250,
                mime_type="",
            ),
        ]

    pages = fetch_competition_pages(
        "demo-comp",
        knowledge_dir=tmp_path,
        list_pages=fake_list,
    )
    assert pages.source == "api"
    assert not pages.is_empty_shell
    assert "A" * 50 in pages.overview_text
    assert "B" * 50 in pages.rules_text

    # Cached on second call without refresh / without hitting list_pages again.
    def boom(_slug: str):
        raise AssertionError("should use cache")

    cached = fetch_competition_pages(
        "demo-comp",
        knowledge_dir=tmp_path,
        list_pages=boom,
        refresh=False,
    )
    assert cached.source == "cache"
    assert not cached.is_empty_shell


def test_looks_like_empty_shell_short_spa_meta():
    assert looks_like_empty_shell("Kaggle · Competition")
    assert not looks_like_empty_shell("x" * 250)
