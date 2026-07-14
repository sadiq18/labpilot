"""Shared JSON extraction for LLM responses (planner, reflection, …)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the outermost JSON object from an LLM response.

    Accepts optional markdown fences; raises ValueError if no object is found
    or JSON is invalid.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Response did not contain a JSON object.")
    return json.loads(text[start : end + 1])
