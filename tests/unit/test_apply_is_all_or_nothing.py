"""M19 §2 exit criterion 2: the workspace is untouched when a proposal is rejected.

`apply_proposal` validated and wrote each file in the same loop, so a proposal
refused on its third file had already written the first two — a tree that is
neither the parent nor the proposal. The next experiment's parent is then
whatever the half-apply left behind.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

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


# --- PR #118 round 3 ---------------------------------------------------------

_GUARD = '\n\nif __name__ == "__main__":\n    main()\n'


def test_a_traversal_is_rejected_not_erased(tmp_path):
    """Reported on PR #118: `lstrip("./")` strips a character set, not a prefix,
    so `../pipeline/evil.py` arrived as `pipeline/evil.py` and `_is_allowed`
    never saw the `..` it exists to reject. Nothing escaped the workspace — what
    was lost was the error telling the model its path was wrong."""
    with pytest.raises(ApplyError, match="path not allowed"):
        apply_proposal(tmp_path, _proposal(("../pipeline/evil.py", "x = 1\n")))


def test_a_leading_dot_slash_is_still_normalised(tmp_path):
    """The carve-out must not cost the behaviour it guards."""
    script = "def main():\n    return 1\n" + _GUARD
    written = apply_proposal(tmp_path, _proposal(("./pipeline/train.py", script)))

    assert written == [tmp_path / "pipeline" / "train.py"]


def test_an_undeclared_import_is_refused(tmp_path):
    """Reported on PR #118. `uv run --script` builds the environment from the
    PEP 723 block alone, so an import it does not name is a ModuleNotFoundError
    one campaign step later. PR #102 fixed this once for the template pack; M19
    §2 deleted the pack and the test with it, and every `train.py` is now
    model-written."""
    content = (
        "# /// script\n"
        '# dependencies = ["lightgbm>=4.0"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "import joblib\n"
        "\n\n"
        "def main():\n"
        "    return lgb, joblib\n"
        "\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    with pytest.raises(ApplyError, match="joblib"):
        apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_the_inline_dependency_form_is_read(tmp_path):
    """Both PEP 723 spellings are legal and a model writes each about half the
    time. Reading only the one-per-line form would report every inline-form
    script as missing every dependency it has."""
    content = (
        "# /// script\n"
        '# dependencies = ["lightgbm>=4.0", "pandas"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "import pandas as pd\n"
        "\n\n"
        "def main():\n"
        "    return lgb, pd\n"
        "\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_an_import_named_differently_from_its_distribution_passes(tmp_path):
    content = (
        "# /// script\n"
        '# dependencies = ["scikit-learn"]\n'
        "# ///\n"
        "import sklearn\n"
        "\n\n"
        "def main():\n"
        "    return sklearn\n"
        "\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_a_script_with_no_block_is_not_asked_about_dependencies(tmp_path):
    """No block means `uv run --script` is not what runs it."""
    content = "import lightgbm as lgb\n\n\ndef main():\n    return lgb\n" + _GUARD

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_a_write_failure_puts_the_earlier_files_back(tmp_path):
    """Reported on PR #118. Validating everything before writing anything covers
    a *later* file being refused; it said nothing about the write loop, where a
    failure on file N left files 1..N-1 on disk — the "neither the parent nor
    the proposal" state this module exists to prevent."""
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("original\n", encoding="utf-8")

    real_write = Path.write_text

    def _fail_on_the_second(self, data, *args, **kwargs):
        if self.name == "infer.py":
            raise OSError("disk full")
        return real_write(self, data, *args, **kwargs)

    with mock.patch.object(Path, "write_text", _fail_on_the_second):
        with pytest.raises(ApplyError, match="Nothing was applied"):
            apply_proposal(
                tmp_path,
                _proposal(
                    ("pipeline/train.py", "def main():\n    return 2\n" + _GUARD),
                    ("pipeline/infer.py", "def main():\n    return 1\n" + _GUARD),
                ),
            )

    assert (tmp_path / "pipeline" / "train.py").read_text(encoding="utf-8") == "original\n"
    assert not (tmp_path / "pipeline" / "infer.py").exists()


# --- PR #118 round 4 ---------------------------------------------------------


def test_the_rollback_names_the_file_that_failed(tmp_path):
    """Reported on PR #118: the cleanup loop reused the write loop's variable,
    so after rollback it held the last *restored* file and the error blamed a
    file that had written fine."""
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("original\n", encoding="utf-8")
    real_write = Path.write_text

    def _fail_on_infer(self, data, *args, **kwargs):
        if self.name == "infer.py":
            raise OSError("disk full")
        return real_write(self, data, *args, **kwargs)

    with mock.patch.object(Path, "write_text", _fail_on_infer):
        with pytest.raises(ApplyError, match="infer.py"):
            apply_proposal(
                tmp_path,
                _proposal(
                    ("pipeline/train.py", "def main():\n    return 2\n" + _GUARD),
                    ("pipeline/infer.py", "def main():\n    return 1\n" + _GUARD),
                ),
            )


def test_the_rollback_removes_directories_it_created(tmp_path):
    """ "Nothing was applied" is not true of a tree left holding new empty
    directories. Reported on PR #118."""
    real_write = Path.write_text

    def _fail_on_infer(self, data, *args, **kwargs):
        if self.name == "infer.py":
            raise OSError("disk full")
        return real_write(self, data, *args, **kwargs)

    with mock.patch.object(Path, "write_text", _fail_on_infer):
        with pytest.raises(ApplyError):
            apply_proposal(
                tmp_path,
                _proposal(
                    ("pipeline/train.py", "def main():\n    return 2\n" + _GUARD),
                    ("pipeline/infer.py", "def main():\n    return 1\n" + _GUARD),
                ),
            )

    assert not (tmp_path / "pipeline").exists()


def test_an_unreadable_existing_file_is_an_apply_error(tmp_path):
    """The snapshot used to run before the guard, so a `PermissionError` came
    out raw from a module whose every other rejection is an `ApplyError`.
    Reported on PR #118."""
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("original\n", encoding="utf-8")

    def _cannot_read(self, *args, **kwargs):
        raise OSError("permission denied")

    with mock.patch.object(Path, "read_bytes", _cannot_read):
        with pytest.raises(ApplyError, match="Nothing was applied"):
            apply_proposal(
                tmp_path,
                _proposal(("pipeline/train.py", "def main():\n    return 2\n" + _GUARD)),
            )

    assert (tmp_path / "pipeline" / "train.py").read_text(encoding="utf-8") == "original\n"


def test_a_dependency_line_may_be_indented(tmp_path):
    """Reported on PR #118: the comment prefix was matched at position zero, so
    one space before the `#` made the block unparseable and a script that
    declared its imports correctly was rejected for not declaring them."""
    content = (
        "# /// script\n"
        ' # dependencies = ["lightgbm>=4.0"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "\n\n"
        "def main():\n"
        "    return lgb\n" + _GUARD
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_a_separator_difference_is_not_a_missing_dependency(tmp_path):
    """PEP 503 normalisation: `scikit_learn` declared against a `sklearn`
    import is the same distribution. Reported on PR #118."""
    content = (
        "# /// script\n"
        '# dependencies = ["scikit_learn"]\n'
        "# ///\n"
        "import sklearn\n"
        "\n\n"
        "def main():\n"
        "    return sklearn\n" + _GUARD
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_a_deferred_import_must_still_be_declared(tmp_path):
    """`def main(): import xgboost` fails exactly like a top-level one, so
    scanning `tree.body` alone missed it. Reported on PR #118."""
    content = (
        "# /// script\n"
        '# dependencies = ["lightgbm"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "\n\n"
        "def main():\n"
        "    import xgboost\n"
        "    return lgb, xgboost\n" + _GUARD
    )

    with pytest.raises(ApplyError, match="xgboost"):
        apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))


def test_an_optional_import_under_try_is_not_required(tmp_path):
    """That is how an optional dependency is written, and the code handles its
    absence itself."""
    content = (
        "# /// script\n"
        '# dependencies = ["lightgbm"]\n'
        "# ///\n"
        "import lightgbm as lgb\n"
        "\n"
        "try:\n"
        "    import wandb\n"
        "except ImportError:\n"
        "    wandb = None\n"
        "\n\n"
        "def main():\n"
        "    return lgb, wandb\n" + _GUARD
    )

    assert apply_proposal(tmp_path, _proposal(("pipeline/train.py", content)))
