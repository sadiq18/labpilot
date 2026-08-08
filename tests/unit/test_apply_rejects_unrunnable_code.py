"""Parsing is not running: `ast.parse` passes files that cannot execute.

The real record, rogii 2026-08-08. Codegen returned **624 bytes** — a module
docstring and half a `# requires-python = ` line, cut off mid-token. That is
valid Python (a docstring followed by comments), so the syntax gate accepted it,
`apply_proposal` wrote it, and seven consecutive executions failed on `uv`'s
complaint about an unterminated dependency block.

Two gates, both of which the truncated file passes today and must not:
  * a `# /// script` block that was opened and never closed
  * a `train.py` with no entry point to run
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
    ApplyError,
    apply_proposal,
)
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)

# Verbatim shape of what codegen actually produced.
_TRUNCATED = '"""Partition-aware regression baseline."""\n\n# /// script\n# requires-python = \\\n'

_GOOD = '''"""Baseline."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lightgbm>=4.0",
# ]
# ///

def main():
    pass


if __name__ == "__main__":
    main()
'''


def _apply(tmp_path, content: str, rel: str = "pipeline/train.py"):
    return apply_proposal(
        tmp_path,
        CodeProposal(files=[CodeFileSpec(path=rel, content=content)]),
    )


def test_the_truncated_file_that_cost_seven_executions_is_rejected(tmp_path):
    with pytest.raises(ApplyError):
        _apply(tmp_path, _TRUNCATED)


def test_ast_parse_alone_would_have_accepted_it(tmp_path):
    """The premise of this module — proves the old gate was insufficient."""
    import ast

    ast.parse(_TRUNCATED)  # must not raise; that is the whole problem


def test_an_unterminated_dependency_block_is_rejected(tmp_path):
    """uv refuses the whole script, so this is a run that cannot start."""
    content = _GOOD.replace("# ///\n\n", "\n")
    with pytest.raises(ApplyError, match="unterminated PEP 723"):
        _apply(tmp_path, content)


def test_a_train_script_with_no_entry_point_is_rejected(tmp_path):
    content = _GOOD.replace('if __name__ == "__main__":\n    main()\n', "")
    with pytest.raises(ApplyError, match="__main__"):
        _apply(tmp_path, content)


def test_the_workspace_is_untouched_when_a_proposal_is_rejected(tmp_path):
    """Propose-then-apply: a bad proposal must not leave a partial file."""
    with pytest.raises(ApplyError):
        _apply(tmp_path, _TRUNCATED)
    assert not (tmp_path / "pipeline" / "train.py").exists()


def test_a_complete_script_still_applies(tmp_path):
    """The carve-out must not cost the behaviour it guards."""
    written = _apply(tmp_path, _GOOD)
    assert (tmp_path / "pipeline" / "train.py").read_text(encoding="utf-8") == _GOOD
    assert len(written) == 1


def test_a_helper_module_needs_no_entry_point(tmp_path):
    """Only train.py is the entry point. Requiring `__main__` everywhere would
    reject correct helpers to catch a rare truncation."""
    helper = "def add(a, b):\n    return a + b\n"
    assert _apply(tmp_path, helper, rel="pipeline/helpers.py")


def test_a_script_without_any_dependency_block_is_fine(tmp_path):
    """Templates predating PEP 723 must keep working — declaring nothing is
    legal, opening a block and abandoning it is not."""
    content = '"""B."""\n\n\ndef main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n'
    assert _apply(tmp_path, content)
