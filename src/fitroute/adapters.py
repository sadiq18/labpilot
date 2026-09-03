"""Provider adapters — the HTTP shape of a completion, nothing more.

An adapter knows how to talk to one *family* of endpoint. It never decides
which provider to use; that is :func:`labpilot.llm.router.select_route`.

``OpenAICompatAdapter`` is the reason this module exists: Groq, GitHub Models,
OpenRouter, Mistral, Cerebras, vLLM, TGI and any self-hosted OpenAI-compatible
server are all one client differing only by ``base_url`` and key. Adding a
provider is then a config entry rather than a code change.

Router-core rule: this module imports nothing from ``labpilot`` (see
``docs/smart-router/DESIGN.md`` §13.1), so it can be extracted to the
standalone package by moving the file.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

_DEFAULT_TIMEOUT_SECONDS = 600.0

#: Sent on every request. Not cosmetic: Groq sits behind Cloudflare, which
#: rejects urllib's default ``Python-urllib/3.x`` agent with a 403 (error 1010)
#: on *every* endpoint — so without this the provider is unreachable rather
#: than merely rate-limited, and the failure looks like a bad key. Measured
#: 2026-08-06: default UA 403, ``fitroute/0.1`` 200. Identifying the client
#: honestly is also what providers ask for; do not impersonate a browser.
_USER_AGENT = "fitroute/0.1"


@dataclass(frozen=True)
class Completion:
    """One completion plus what it cost.

    ``prompt_tokens``/``completion_tokens`` are ``None`` when the endpoint
    reported no usage — deliberately distinct from ``0``, so a provider that
    reports nothing shows up as an unmetered call rather than silently
    understating spend.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def metered(self) -> bool:
        return self.prompt_tokens is not None or self.completion_tokens is not None


class OpenAICompatAdapter:
    """Any ``/chat/completions`` endpoint that speaks the OpenAI shape."""

    #: Whether ``json_mode`` can be enforced rather than merely requested.
    supports_json_mode = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            # Constrained decoding. Small models routinely ignore "reply with
            # JSON" and answer in prose; the reply is then discarded and the
            # agent degrades silently. Enforcing the grammar is far more
            # reliable than prompting harder or parsing more leniently.
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            # Status is preserved in the message because _is_transient_llm_error
            # classifies retryability by matching "429"/"503" in the text.
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"{self.base_url} timed out after {self.timeout_seconds:g}s on {model!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.base_url} unreachable: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.base_url} returned no choices for {model!r}")
        content = (choices[0].get("message") or {}).get("content")
        # `""` is a `str`, so an isinstance check alone accepts an empty body as
        # a valid answer: no raise, no failover, and the caller gets nothing.
        # `_RETRYABLE_TEXT` already lists "NO MESSAGE.CONTENT" precisely so this
        # case moves to the next provider — but it only ever fired for a missing
        # key, never for a present-and-empty one.
        #
        # Measured on playground-series-s6e8 (2026-08-30): the Conductor policy
        # failed twice in one campaign with "Response did not contain a JSON
        # object. Got: ''" and dropped to the offline engine, while the gateway
        # logged no provider failure at all — because from its side the call had
        # succeeded.
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"{self.base_url} returned no message.content for {model!r}")

        usage = body.get("usage") or {}
        return Completion(
            text=content,
            prompt_tokens=_maybe_int(usage.get("prompt_tokens")),
            completion_tokens=_maybe_int(usage.get("completion_tokens")),
        )


class OllamaAdapter:
    """Local inference. Same contract, so local competes on the same terms."""

    supports_json_mode = True

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout_seconds:g}s on model {model!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable at {self.base_url}: {exc}") from exc

        content = (body.get("message") or {}).get("content")
        # Same reasoning as the openai_compat adapter above: an empty string is
        # a `str`, and returning it hands the caller an answer that is not one.
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama response missing message.content")
        return Completion(
            text=content,
            prompt_tokens=_maybe_int(body.get("prompt_eval_count")),
            completion_tokens=_maybe_int(body.get("eval_count")),
        )


def build_adapter(
    kind: str,
    *,
    base_url: str,
    api_key: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
):
    """``ProviderSpec.kind`` -> adapter. The only place kinds are enumerated."""
    if kind == "ollama":
        return OllamaAdapter(base_url, timeout_seconds=timeout_seconds)
    if kind == "openai_compat":
        return OpenAICompatAdapter(base_url, api_key, timeout_seconds=timeout_seconds)
    raise ValueError(f"Unknown provider kind {kind!r} (expected: openai_compat, ollama)")


def _maybe_int(value: object) -> int | None:
    """``None`` for absent usage — see :class:`Completion`."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
