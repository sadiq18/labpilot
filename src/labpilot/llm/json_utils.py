"""Shared JSON extraction for LLM responses (planner, reflection, …)."""

from __future__ import annotations

import json
import re
from typing import Any

# Trailing comma before a closing brace/bracket — the most common way a local
# model produces *nearly* valid JSON.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _candidate_spans(text: str) -> list[str]:
    """Return balanced ``{...}`` spans, longest first.

    A naive "first brace to last brace" slice breaks whenever the model wraps
    its answer in prose that itself contains braces, or emits two objects back
    to back — the slice then spans both and parses as neither. Scanning for
    balanced spans (respecting string literals and escapes) recovers the real
    objects instead.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start : index + 1])
    return sorted(spans, key=len, reverse=True)


def _loads(candidate: str) -> dict[str, Any] | None:
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            parsed = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Tolerates markdown fences, surrounding prose, several objects in one
    response, and a trailing comma. Raises ValueError when nothing parses.

    Robustness here matters disproportionately: every micro agent silently
    falls back to its rule engine when parsing fails, so a brittle parser
    quietly disables the whole intelligence layer on a local model rather than
    reporting an error.
    """
    text = text.strip()

    parsed = _loads(text)
    if parsed is not None:
        return parsed

    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        parsed = _loads(fence.group(1).strip())
        if parsed is not None:
            return parsed

    for candidate in _candidate_spans(text):
        parsed = _loads(candidate)
        if parsed is not None:
            return parsed

    snippet = text[:200].replace("\n", " ")
    raise ValueError(f"Response did not contain a JSON object. Got: {snippet!r}")
