"""Smoke tests for the optional ``deep`` extra (transformers stack).

Full deep Pipeline integration tests were retired with the Engineer cutover;
these keep ``pytest -m deep`` selectable so the CI deep job does not exit 5.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deep


def test_deep_extra_imports() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
