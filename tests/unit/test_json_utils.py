"""Unit tests for llm.json_utils.parse_json_object."""

import pytest

from labpilot.llm.json_utils import parse_json_object


def test_parse_raw_json_object():
    assert parse_json_object('{"a": 1, "b": [2]}') == {"a": 1, "b": [2]}


def test_parse_fenced_json_object():
    text = """Here you go:
```json
{"strategy": "tune", "actions": ["tune_hyperparams"]}
```
"""
    assert parse_json_object(text)["strategy"] == "tune"


def test_parse_object_embedded_in_prose():
    text = 'Result: {"ok": true, "n": 3} thanks'
    assert parse_json_object(text) == {"ok": True, "n": 3}


def test_parse_rejects_missing_object():
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("no braces here")
