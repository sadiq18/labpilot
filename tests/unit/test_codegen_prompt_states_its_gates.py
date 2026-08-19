"""A gate may only reject a file for a rule the prompt stated.

`_check_dependency_block` rejects `pipeline/train.py` whole when a third-party
import is missing from its PEP 723 block — no training, no metrics, no
evidence. The rule was written down in the code engineer's `skill.md`, which
nothing loads, and appeared in no prompt the model is actually sent.

Measured 2026-08-20 on two unrelated models, qwen2.5-coder:14b and
gemini-3.5-flash-lite: every generated `train.py` imported pandas/sklearn with
no block and was rejected, in campaigns that then ended having produced nothing.
Identical failures from unrelated models is the signature of an undeclared
requirement rather than weak codegen.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
    ApplyError,
    _check_dependencies_are_complete,
)

_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src/labpilot/research_engine/execution/micro_agents/code_engineer"
    / "prompts/code_engineer_system.md"
)


def test_the_prompt_states_the_dependency_rule_its_gate_enforces() -> None:
    prompt = _PROMPT.read_text(encoding="utf-8")

    assert "PEP 723" in prompt, "the gate rejects files for a rule the prompt never states"
    assert "# /// script" in prompt, "state the block's syntax, not just its name"
    assert "dependencies = [" in prompt


def test_the_prompt_names_the_imports_a_model_assumes_are_free() -> None:
    """`pandas` and `numpy` read as ambient to a model writing ML code, and
    they are exactly what the rejections named."""
    prompt = _PROMPT.read_text(encoding="utf-8").lower()

    assert "pandas" in prompt and "numpy" in prompt


def test_the_gate_still_rejects_an_incomplete_block() -> None:
    """The rule being documented does not soften it — a block that omits an
    import it uses must still be refused, or the documentation is decoration.

    Note what the gate does *not* do: a file with no block at all skips this
    check entirely (`_declared_dependencies` returns None). So the rejections
    measured were models writing a block and under-filling it, which is why the
    prompt has to name the ordinary-looking imports rather than only the rule.
    """
    incomplete = (
        '"""Train."""\n'
        "# /// script\n"
        '# dependencies = ["lightgbm>=4.0"]\n'
        "# ///\n\n"
        "import pandas as pd\nimport lightgbm as lgb\n\nprint(pd, lgb)\n"
    )

    with pytest.raises(ApplyError, match="does not declare"):
        _check_dependencies_are_complete(
            "pipeline/train.py", incomplete, ast.parse(incomplete)
        )


def test_the_gate_accepts_a_complete_block() -> None:
    complete = (
        '"""Train."""\n'
        "# /// script\n"
        '# dependencies = ["pandas>=2.0", "lightgbm>=4.0"]\n'
        "# ///\n\n"
        "import pandas as pd\nimport lightgbm as lgb\n\nprint(pd, lgb)\n"
    )

    _check_dependencies_are_complete("pipeline/train.py", complete, ast.parse(complete))
