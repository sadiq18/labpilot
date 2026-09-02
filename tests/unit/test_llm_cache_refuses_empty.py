"""An empty response is the absence of an answer, not an answer.

Measured on playground-series-s6e8 (2026-08-30): while the free tier was
throttling, six calls returned empty bodies (429 with no choices, `content: ''`)
and were cached. Two later campaigns then failed identically on "Response did
not contain a JSON object. Got: ''" — served the cached emptiness, never
reaching a provider that was healthy again by then.

The tell was the bill: both runs cost exactly $0.00. A cache hit never bills, so
a campaign that fails at zero cost is a campaign that never made a request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fitroute.cache import PromptCache


@pytest.fixture
def cache(tmp_path: Path) -> PromptCache:
    return PromptCache(tmp_path / "llm.sqlite", enabled=True)


@pytest.mark.parametrize("body", ["", "   ", "\n\t "])
def test_an_empty_response_is_not_cached(cache: PromptCache, body: str) -> None:
    """Whitespace counts as empty: a body of spaces satisfies no parser
    downstream, and caching it fails the same way as `''`."""
    cache.set("k", body, model="m")

    assert cache.get("k") is None


def test_a_real_response_is_still_cached(cache: PromptCache) -> None:
    """The guard must not disable caching — that would turn every campaign
    into full price."""
    cache.set("k", '{"tool": "run_plan"}', model="m")

    assert cache.get("k") == '{"tool": "run_plan"}'


def test_an_empty_response_does_not_overwrite_a_good_one(cache: PromptCache) -> None:
    """The poisoning path in the wild: a key answered once, then a throttled
    retry returns empty. `set` upserts, so without the guard the good answer is
    replaced by the empty one and never recovers.
    """
    cache.set("k", '{"tool": "run_plan"}', model="m")
    cache.set("k", "", model="m")

    assert cache.get("k") == '{"tool": "run_plan"}'


def test_an_already_cached_empty_response_is_a_miss(cache: PromptCache, tmp_path: Path) -> None:
    """Refusing to write one only protects a cache that never had the problem.

    Every machine that hit this already holds the rows — six of them in the run
    it was found on — written before the guard existed. Without a read-side
    check they go on being served an empty answer, at $0.00, with the fix
    merged and apparently applied. A blank body is a miss, so the call reaches
    a provider that is healthy by now.
    """
    assert cache._conn is not None
    cache._conn.execute(
        "INSERT INTO llm_cache (cache_key, response, model) VALUES (?, ?, ?)",
        ("legacy", "", "m"),
    )
    cache._conn.commit()

    assert cache.get("legacy") is None
