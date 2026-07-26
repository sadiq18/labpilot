"""Smoke tests for the optional ``image`` extra (torch / torchvision / pillow).

Full image Pipeline integration tests were retired with the Engineer cutover;
these keep ``pytest -m image`` selectable so the CI image job does not exit 5.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.image


def test_image_extra_imports() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
