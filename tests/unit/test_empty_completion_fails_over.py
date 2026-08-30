"""An empty completion is a provider failure, not an answer.

The gateway already lists "NO MESSAGE.CONTENT" in `_RETRYABLE_TEXT`, added
because an HTTP 200 with nothing usable "is precisely the case another provider
can answer". But the adapter only raised when `content` was a non-string, and
`""` is a string — so a present-and-empty body sailed through as success.

Measured on playground-series-s6e8 (2026-08-30): the Conductor policy failed
twice in one campaign with "Response did not contain a JSON object. Got: ''"
and fell back to the offline engine, while the gateway logged no provider
failure at all. From its side the call had succeeded.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fitroute.adapters import OpenAICompatAdapter


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _adapter_returning(monkeypatch: pytest.MonkeyPatch, content: object) -> OpenAICompatAdapter:
    payload = {"choices": [{"message": {"content": content}}], "usage": {}}
    monkeypatch.setattr(
        "fitroute.adapters.urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(payload),
    )
    return OpenAICompatAdapter("https://example.test/v1", "k")


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_an_empty_body_raises_so_the_gateway_fails_over(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """The message must keep saying `no message.content` — `_RETRYABLE_TEXT`
    matches on that string, so rewording it silently disables failover."""
    adapter = _adapter_returning(monkeypatch, content)

    with pytest.raises(RuntimeError, match="no message.content"):
        adapter.complete("", "hi", model="m", temperature=0.0)


def test_a_real_body_still_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not reject working responses."""
    adapter = _adapter_returning(monkeypatch, '{"tool": "run_plan"}')

    assert adapter.complete("", "hi", model="m", temperature=0.0).text == '{"tool": "run_plan"}'


def test_the_raised_message_is_classified_retryable() -> None:
    """The end the fix exists for: this error must route to another provider,
    not end the call."""
    from fitroute.gateway import _is_retryable_upstream

    exc = RuntimeError("https://example.test/v1 returned no message.content for 'm'")

    assert _is_retryable_upstream(exc)
