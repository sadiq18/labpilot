"""Ollama local-model provider (stdlib HTTP — no extra dependency)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self, timeout_seconds: float = 5.0) -> list[str]:
        """Return locally pulled model names (empty when unreachable)."""
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — soft probe
            logger.debug("Ollama tags failed at %s (%s)", self.base_url, exc)
            return []
        return [m.get("name", "") for m in body.get("models") or [] if m.get("name")]

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            # Constrained decoding. Small local models routinely ignore "reply
            # with JSON" and answer in prose — observed as an analyzer
            # explaining competition rules in English and being discarded.
            # Asking the runtime to enforce the grammar is far more reliable
            # than prompting harder or parsing more leniently.
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
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout_seconds:g}s on model {model!r}. "
                "Raise llm.request_timeout_seconds or use a smaller model."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable at {self.base_url}: {exc}") from exc

        message = body.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response missing message.content")
        return content

    def is_reachable(self, timeout_seconds: float = 0.25) -> bool:
        """Cheap liveness probe used by auto mode."""
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds):
                return True
        except Exception as exc:  # noqa: BLE001 — soft probe
            logger.debug("Ollama not reachable at %s (%s)", self.base_url, exc)
            return False


class OllamaClient:
    """Legacy LLMClient-shaped wrapper (model/temperature bound at construct)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = OllamaProvider(base_url, timeout_seconds)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        return self._provider.complete(
            system,
            user,
            model=self.model,
            temperature=self.temperature,
            json_mode=json_mode,
        )
