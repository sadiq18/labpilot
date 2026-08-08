"""The prompt cache is reached from whatever thread makes an LLM call.

`BudgetLedger` already carries this fix, with the failure spelled out in its own
comment — *"without this every proxied request failed with 'SQLite objects
created in a thread can only be used in that same thread'"*. `PromptCache` is
its sibling: same package, same sqlite pattern, same gateway, and it was left
behind.

The cost of missing it is total rather than partial. `gateway._complete_once`
reads the cache *before* selecting a provider, so a cross-thread call dies at
the lookup and never reaches the model at all. Measured on rogii 2026-08-08:
CodeEngineerAgent, EvidenceSynthesisAgent and RecommendationAgent all reported
"skipped" for this one line, and codegen silently rendered a Jinja template
instead of calling the LLM — for four consecutive campaigns.
"""

from __future__ import annotations

import threading

from fitroute.cache import PromptCache, cache_key


def _in_thread(fn):
    """Run `fn` on another thread; re-raise whatever it raised."""
    box: dict[str, BaseException | object] = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            box["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=10)
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def test_a_read_from_another_thread_does_not_raise(tmp_path):
    """This is the exact call that killed every micro agent."""
    cache = PromptCache(tmp_path / "cache.db")
    try:
        assert _in_thread(lambda: cache.get("missing")) is None
    finally:
        cache.close()


def test_a_write_from_another_thread_is_visible_to_the_creator(tmp_path):
    cache = PromptCache(tmp_path / "cache.db")
    key = cache_key("m", "prompt", 0.3, "system")
    try:
        _in_thread(lambda: cache.set(key, "answer", model="m"))
        assert cache.get(key) == "answer"
    finally:
        cache.close()


def test_a_value_written_here_is_readable_there(tmp_path):
    cache = PromptCache(tmp_path / "cache.db")
    key = cache_key("m", "prompt", 0.3, "system")
    try:
        cache.set(key, "answer", model="m")
        assert _in_thread(lambda: cache.get(key)) == "answer"
    finally:
        cache.close()


def test_concurrent_writers_do_not_corrupt_the_cache(tmp_path):
    """`check_same_thread=False` permits cross-thread use; the lock is what
    makes it safe. sqlite tolerates cross-thread, not concurrent."""
    cache = PromptCache(tmp_path / "cache.db")
    try:
        threads = [
            threading.Thread(
                target=lambda i=i: cache.set(cache_key("m", f"p{i}", 0.3, "s"), f"a{i}", model="m")
            )
            for i in range(16)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for i in range(16):
            assert cache.get(cache_key("m", f"p{i}", 0.3, "s")) == f"a{i}"
    finally:
        cache.close()


def test_a_disabled_cache_is_still_safe_to_close(tmp_path):
    cache = PromptCache(tmp_path / "cache.db", enabled=False)
    assert cache.get("anything") is None
    cache.set("k", "v", model="m")
    cache.close()
