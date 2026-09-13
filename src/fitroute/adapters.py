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

#: The two ways an OpenAI-compatible endpoint can be made to emit JSON, and the
#: permitted values of ``ProviderSpec.json_via``. ``response_format`` is the
#: default because it is simpler and better tested; ``tools`` exists because
#: several capable models advertise ``tools`` but not ``response_format``, and
#: without it those models fail the ``structured_output`` precondition and are
#: rejected for exactly the roles they are best at.
_JSON_ROUTES = ("response_format", "tools")

#: The single property the forced call fills in. See :data:`_JSON_TOOL`.
_ENVELOPE_KEY = "json"

#: The function a ``tools``-route request forces the model to call.
#:
#: The payload's shape is set by the prompt, which the adapter never sees, so
#: the schema has to be shape-agnostic — but it cannot be an *open* object.
#: Providers validate arguments against the declared schema and drop anything
#: not in ``properties``: measured 2026-08-07 against gemini-3.5-flash-lite,
#: both ``{"type": "object", "additionalProperties": true}`` and the same with
#: an empty ``properties`` returned a forced call whose arguments were ``{}`` —
#: a well-formed, entirely empty answer, which is worse than a failure because
#: it parses. Declaring one required string property and carrying the payload
#: inside it is what survives validation; the same probe round-tripped a nested
#: object containing Python source with newlines intact.
_JSON_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_json",
        "description": "Return the answer to the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                _ENVELOPE_KEY: {
                    "type": "string",
                    "description": "The answer, as a JSON object serialised to a string.",
                }
            },
            "required": [_ENVELOPE_KEY],
        },
    },
}


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
        json_via: str = "response_format",
    ) -> None:
        if json_via not in _JSON_ROUTES:
            # Loudly, like an unknown name in routing.use: a typo that fell back
            # to the default would send response_format to an endpoint that
            # ignores it, and the symptom would be prose where JSON was required.
            raise ValueError(f"Unknown json_via {json_via!r} (expected: {', '.join(_JSON_ROUTES)})")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.json_via = json_via

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
            if self.json_via == "tools":
                payload["tools"] = [_JSON_TOOL]
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": _JSON_TOOL["function"]["name"]},
                }
            else:
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
        message = choices[0].get("message") or {}
        if json_mode and self.json_via == "tools":
            text = self._forced_tool_payload(message, model)
        else:
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"{self.base_url} returned no message.content for {model!r}")
            text = content

        usage = body.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=_maybe_int(usage.get("prompt_tokens")),
            completion_tokens=_maybe_int(usage.get("completion_tokens")),
        )

    def _forced_tool_payload(self, message: dict, model: str) -> str:
        """The JSON the forced call carried, unwrapped from its envelope."""
        calls = message.get("tool_calls") or []
        if not calls:
            # The call was forced, so an endpoint answering without it does not
            # have the capability its catalog entry claims. Raising says which
            # entry is wrong; falling back to message.content would hide it and
            # hand prose to a caller that is about to json.loads it.
            raise RuntimeError(
                f"{self.base_url} ignored the forced tool call for {model!r} — "
                f"json_via='tools' is not supported by this model"
            )
        arguments = (calls[0].get("function") or {}).get("arguments")
        # Per the OpenAI schema `arguments` is a JSON *string*, but several
        # compatible endpoints send the object itself; both mean the same thing.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{self.base_url} returned unparseable tool arguments for {model!r}: {exc}"
                ) from exc
        if not isinstance(arguments, dict) or _ENVELOPE_KEY not in arguments:
            raise RuntimeError(
                f"{self.base_url} returned a tool call without {_ENVELOPE_KEY!r} for {model!r}"
            )

        payload = arguments[_ENVELOPE_KEY]
        # The schema asks for a string, but a model that ignores the envelope and
        # nests the object directly has still answered the question.
        return payload if isinstance(payload, str) else json.dumps(payload)


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
        if not isinstance(content, str):
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
    json_via: str = "response_format",
):
    """``ProviderSpec.kind`` -> adapter. The only place kinds are enumerated."""
    if kind == "ollama":
        return OllamaAdapter(base_url, timeout_seconds=timeout_seconds)
    if kind == "openai_compat":
        return OpenAICompatAdapter(
            base_url, api_key, timeout_seconds=timeout_seconds, json_via=json_via
        )
    raise ValueError(f"Unknown provider kind {kind!r} (expected: openai_compat, ollama)")


def _maybe_int(value: object) -> int | None:
    """``None`` for absent usage — see :class:`Completion`."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
