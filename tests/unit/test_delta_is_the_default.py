"""M19 §3: delta is the default, and §4's removal shipped with it.

The flip needed a number rather than a preference. Measured on rogii
2026-08-09 after the codegen and retry fixes: 18 aider attempts, 17 usable, one
`aider_no_edit` — 5.6%, and 0% over the eight attempts with every fix in place.
The step 1c format comparison stands: `diff` at +18/-7 against `whole` at
+23/-7, for 19% fewer tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_default_strategy_is_delta():
    from labpilot.config import CodegenConfig

    assert CodegenConfig().strategy == "delta"


def test_the_handler_fallback_follows_that_default(tmp_path):
    """Two places naming a default is how they drift; the fallback reads it."""
    from labpilot.config import CodegenConfig
    from labpilot.research_engine.tools.handlers.run import _codegen_strategy

    class _WS:
        root = tmp_path  # no configs/default.yaml here, so the read fails

    assert _codegen_strategy(_WS()) == CodegenConfig().strategy


def test_the_template_pack_is_gone():
    """§4 ships with §3, not after it — a removal and the precondition that
    makes it safe travel together."""
    root = (
        Path(__file__).resolve().parents[2]
        / "src/labpilot/research_engine/execution/capabilities/code_engineering"
    )

    # Sources, not directories — a stale `__pycache__` left behind by a
    # `git rm` is noise, while a surviving `.j2` or renderer module is the
    # deletion not having happened.
    assert not list(root.rglob("*.j2"))
    assert not list((root / "offline_codegen").rglob("*.py"))


def test_baseline_selection_survives_the_deletion():
    """The pack was also the registry — `list_templates()` scanned it and kept
    the entries whose directory existed, so deleting it emptied the catalogue
    and took baseline selection with it."""
    from labpilot.research_engine.execution.baseline.registry import (
        get_template,
        list_templates,
    )

    assert list_templates()
    chosen = get_template("tabular_regression")
    assert chosen is not None
    assert chosen.model_family == "lightgbm"


@pytest.mark.parametrize(
    "problem_type",
    ["tabular_regression", "tabular_classification", "text_classification"],
)
def test_every_declared_problem_type_still_resolves(problem_type):
    from labpilot.research_engine.execution.baseline.registry import get_template

    assert get_template(problem_type) is not None
