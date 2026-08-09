"""M19 §2 exit criterion 2: the workspace is untouched when a proposal is rejected.

`apply_proposal` validated and wrote each file in the same loop, so a proposal
refused on its third file had already written the first two — a tree that is
neither the parent nor the proposal. The next experiment's parent is then
whatever the half-apply left behind.
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

_GOOD = (
    "# /// script\n"
    "# dependencies = []\n"
    "# ///\n"
    "def main():\n"
    "    return 1\n"
    "\n\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)


def _proposal(*files):
    return CodeProposal(files=[CodeFileSpec(path=p, content=c) for p, c in files])


def test_a_syntax_error_in_a_later_file_writes_nothing(tmp_path):
    """The first file is valid and must still not land."""
    with pytest.raises(ApplyError):
        apply_proposal(
            tmp_path,
            _proposal(("pipeline/helper.py", _GOOD), ("pipeline/train.py", "def broken(:\n")),
        )

    assert not (tmp_path / "pipeline" / "helper.py").exists()


def test_a_disallowed_path_in_a_later_file_writes_nothing(tmp_path):
    with pytest.raises(ApplyError):
        apply_proposal(
            tmp_path,
            _proposal(("pipeline/helper.py", _GOOD), ("../escape.py", _GOOD)),
        )

    assert not (tmp_path / "pipeline" / "helper.py").exists()


def test_an_existing_file_is_not_overwritten_by_a_rejected_proposal(tmp_path):
    """The case that matters: the parent must survive intact, or the next
    experiment diffs against something nobody proposed."""
    (tmp_path / "pipeline").mkdir()
    parent = tmp_path / "pipeline" / "train.py"
    parent.write_text("PARENT\n", encoding="utf-8")

    with pytest.raises(ApplyError):
        apply_proposal(
            tmp_path,
            _proposal(("pipeline/train.py", _GOOD), ("pipeline/bad.py", "def broken(:\n")),
        )

    assert parent.read_text() == "PARENT\n"


def test_a_valid_proposal_still_writes_every_file(tmp_path):
    written = apply_proposal(
        tmp_path,
        _proposal(("pipeline/train.py", _GOOD), ("pipeline/helper.py", _GOOD)),
    )

    assert len(written) == 2
    assert (tmp_path / "pipeline" / "train.py").read_text() == _GOOD
    assert (tmp_path / "pipeline" / "helper.py").read_text() == _GOOD


def test_a_docstring_mentioning_the_rule_is_not_an_import(tmp_path):
    """Reported on PR #118.

    The check scanned raw text, so a generated script whose docstring
    *documents* the constraint — "never `import labpilot` in a declaring
    script" — was rejected for obeying it. The tree is already parsed two lines
    earlier; an import is an AST fact.
    """
    content = (
        '"""Trains the model.\n'
        "\n"
        "    Declares its own dependencies, so it must never\n"
        "    import labpilot or use `from labpilot...` here.\n"
        '    """\n'
        "# /// script\n"
        '# dependencies = ["lightgbm>=4.0"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "\n"
        "\n"
        "def main():\n"
        "    return lgb\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_a_real_import_is_still_caught(tmp_path):
    content = (
        "# /// script\n"
        '# dependencies = ["lightgbm>=4.0"]\n'
        "# ///\n"
        "from labpilot.research_engine.execution.metrics import compute_metric\n"
        "\n"
        "\n"
        "def main():\n"
        "    return compute_metric\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    with pytest.raises(ApplyError, match="ephemeral environment"):
        apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))
