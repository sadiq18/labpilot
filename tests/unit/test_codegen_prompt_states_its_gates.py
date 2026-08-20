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

The rule is stated in the runner's own terms — a script, its dependencies, and
what happens when they disagree — with no assumption that the script trains a
model. The seam this agent writes for is a validator, not a trainer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
    TRAIN_RELPATH,
    ApplyError,
    _check_dependencies_are_complete,
)
from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    _missing_entry_point,
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
    """`pandas` and `numpy` read as ambient to a model writing analysis code,
    and they are exactly what the rejections named — so they belong in the
    example the model reads, not only in prose about it."""
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


# -- the entry point the runner looks up by name ---------------------------


def test_the_prompt_states_the_entry_point_path() -> None:
    """`pipeline/train.py` appeared only as a placeholder inside the JSON
    schema example and in a branch that assumes prior code exists. Nothing said
    the from-scratch script must have that exact path, and the runner looks it
    up by name — so a model naming its script after the task produced a file
    that applied cleanly and was never run.

    Measured 2026-08-20: three consecutive runs emitted `pipeline/baseline.py`,
    matching the plan's own name and the `configs/baseline.yaml` beside it.
    """
    prompt = _PROMPT.read_text(encoding="utf-8")

    assert "The entry point is exactly `pipeline/train.py`" in prompt
    assert "baseline.py" in prompt, "name the wrong-but-natural choice, not just the right one"


def test_the_missing_entry_point_error_names_what_landed() -> None:
    """`"train.py missing after apply"` is true and unactionable: the proposal
    applied, so files *were* written, and which ones is the whole question."""
    message = _missing_entry_point([Path("/ws/pipeline/baseline.py"), Path("/ws/configs/b.yaml")])

    assert "baseline.py" in message
    assert TRAIN_RELPATH in message


def test_the_error_copes_with_a_proposal_that_wrote_nothing() -> None:
    assert "nothing" in _missing_entry_point([])
