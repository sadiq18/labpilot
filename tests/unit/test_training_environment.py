"""Generated code declares what it needs, and runs without the operator's keys.

Measured on rogii 2026-08-07: the LLM wrote `import catboost` — a sound choice
for tabular regression — and every run died at line 19 because catboost is not a
labpilot dependency. Eight consecutive executions failed identically, twenty
campaign steps produced zero evidence cards, and nothing had told the model what
was installed.

The fix is not a maintained allowlist. The script declares its own dependencies
and runs in a throwaway environment, so codegen can reach for the right library.
"""

from __future__ import annotations

import os

import pytest

from labpilot.research_engine.execution.training.environment import (
    child_environment,
    declared_dependencies,
    declares_dependencies,
    is_secret_env,
    training_command,
)

PEP723 = '''"""Docstring."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lightgbm>=4.0",
#   "catboost",
# ]
# ///

import catboost
'''

PLAIN = '"""Docstring."""\n\nimport json\n'


@pytest.fixture
def script(tmp_path):
    def _write(body: str, name: str = "train.py"):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        return p

    return _write


# --- declaring dependencies -------------------------------------------------


def test_a_declared_block_is_detected(script):
    assert declares_dependencies(script(PEP723))


def test_a_script_without_a_block_is_not(script):
    assert not declares_dependencies(script(PLAIN))


def test_dependencies_are_read_in_order(script):
    assert declared_dependencies(script(PEP723)) == ["lightgbm>=4.0", "catboost"]


def test_no_block_means_no_dependencies(script):
    assert declared_dependencies(script(PLAIN)) == []


def test_a_comment_mentioning_the_fence_is_not_a_block(script):
    """The fence must start a line, or prose about PEP 723 becomes metadata."""
    body = '"""Docs."""\n\n# see the # /// script convention\nimport json\n'
    assert not declares_dependencies(script(body))


# --- choosing how to run ----------------------------------------------------


def test_a_declaring_script_runs_in_an_ephemeral_env(script, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    cmd = training_command(script(PEP723), python="/usr/bin/python3")
    assert cmd[:3] == ["uv", "run", "--script"]


def test_a_plain_script_keeps_the_current_interpreter(script, monkeypatch):
    """Every template predating this change must keep working: with no declared
    deps an ephemeral env would only remove access to labpilot's own."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    cmd = training_command(script(PLAIN), python="/usr/bin/python3")
    assert cmd == ["/usr/bin/python3", str(script(PLAIN))]


def test_without_uv_it_falls_back_and_warns(script, monkeypatch, caplog):
    monkeypatch.setattr("shutil.which", lambda _: None)
    path = script(PEP723)
    with caplog.at_level("WARNING"):
        cmd = training_command(path, python="/usr/bin/python3")
    assert cmd == ["/usr/bin/python3", str(path)]
    assert "uv is not on PATH" in caplog.text


# --- credentials do not reach generated code --------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "KAGGLE_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "SOME_VENDOR_API_KEY",
        "MY_DB_PASSWORD",
    ],
)
def test_secrets_are_recognised(name):
    """Prefix and marker matching, because provider keys arrive under names no
    list can enumerate ahead of time — the same open-world problem as package
    names, handled the same way."""
    assert is_secret_env(name)


@pytest.mark.parametrize("name", ["PATH", "HOME", "LANG", "PYTHONPATH", "TMPDIR"])
def test_ordinary_variables_survive(name):
    assert not is_secret_env(name)


def test_the_child_environment_drops_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("KAGGLE_KEY", "kg-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = child_environment()
    assert "OPENROUTER_API_KEY" not in env
    assert "KAGGLE_KEY" not in env
    assert env["PATH"] == "/usr/bin"


def test_the_child_environment_is_not_empty(monkeypatch):
    """Stripping everything would break the run rather than protect it."""
    monkeypatch.setenv("PATH", "/usr/bin")
    assert child_environment()


def test_an_explicit_base_is_filtered_too():
    assert child_environment({"PATH": "/bin", "GROQ_API_KEY": "x"}) == {"PATH": "/bin"}


def test_os_environ_is_not_mutated(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-x")
    child_environment()
    assert os.environ["GROQ_API_KEY"] == "sk-x", "filtering must copy, not strip in place"


# --- a stale metrics file must not pass for a fresh result ------------------


def test_a_stale_metrics_file_is_not_this_run(tmp_path):
    """The guard asked "is there a metrics file?" when the question is "did this
    run write one?".

    `metrics.json` from an earlier successful run sits at the workspace root and
    survives a failure. Measured on rogii 2026-08-07: eight consecutive
    executions died at `import catboost`, every one recorded `run_experiment
    completed`, and the campaign looked healthy for 20 steps.
    """
    import time
    from types import SimpleNamespace

    from labpilot.research_engine.tools.handlers.specialists import _metrics_written_since

    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"cv_rmse": 194.8}', encoding="utf-8")
    ref = SimpleNamespace(kind="metrics", path=str(metrics))

    started_later = time.time() + 60  # the run began after the file was written
    assert not _metrics_written_since(ref, started_later)


def test_a_fresh_metrics_file_counts(tmp_path):
    import time
    from types import SimpleNamespace

    from labpilot.research_engine.tools.handlers.specialists import _metrics_written_since

    started = time.time()
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"cv_rmse": 190.9}', encoding="utf-8")
    assert _metrics_written_since(SimpleNamespace(kind="metrics", path=str(metrics)), started)


def test_a_missing_metrics_ref_is_not_a_result():
    from labpilot.research_engine.tools.handlers.specialists import _metrics_written_since

    assert not _metrics_written_since(None, 0.0)


def test_a_ref_pointing_at_nothing_is_not_a_result(tmp_path):
    from types import SimpleNamespace

    from labpilot.research_engine.tools.handlers.specialists import _metrics_written_since

    ref = SimpleNamespace(kind="metrics", path=str(tmp_path / "gone.json"))
    assert not _metrics_written_since(ref, 0.0)


# --- the invariant that PR #102 review caught the hard way -------------------


def _template_scripts():
    from pathlib import Path

    root = (
        Path(__file__).resolve().parents[2]
        / "src/labpilot/research_engine/execution/capabilities/code_engineering/templates"
    )
    return sorted(root.rglob("train.py.j2"))


def test_templates_exist():
    assert _template_scripts(), "no templates found — this guard would pass vacuously"


@pytest.mark.parametrize("path", _template_scripts(), ids=lambda p: p.parent.name)
def test_a_declaring_template_must_not_import_labpilot(path):
    """`uv run --script` cannot see labpilot, so the two cannot coexist.

    Caught in review: PEP 723 blocks were added to `tabular_regression` and
    `tabular_classification`, both of which do
    `from labpilot.research_engine.execution.metrics import compute_metric`.
    With uv on PATH — the common case — those templates would have run in an
    ephemeral env and died with ModuleNotFoundError. The templates *changed* by
    that PR were the ones it would have broken.

    A template either declares its dependencies and stands alone, or uses
    labpilot's environment. Never both.
    """
    body = path.read_text(encoding="utf-8")
    if "# /// script" not in body:
        pytest.skip(f"{path.parent.name} does not declare dependencies")
    assert "labpilot" not in body, (
        f"{path.parent.name} declares PEP 723 dependencies but imports labpilot, "
        "which the ephemeral environment cannot see"
    )


@pytest.mark.parametrize("path", _template_scripts(), ids=lambda p: p.parent.name)
def test_a_declaring_template_declares_every_third_party_import(path):
    """`joblib` was undeclared and worked only as a sklearn transitive — the
    kind of accident that breaks when a resolver picks a different tree."""
    import re

    body = path.read_text(encoding="utf-8")
    if "# /// script" not in body:
        pytest.skip(f"{path.parent.name} does not declare dependencies")

    declared = " ".join(declared_dependencies(body))
    stdlib_ok = {
        "json",
        "os",
        "sys",
        "pathlib",
        "typing",
        "dataclasses",
        "math",
        "itertools",
        "collections",
        "warnings",
        "time",
        "re",
        "logging",
    }
    alias = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
    imported = {m or n for m, n in re.findall(r"^import (\w+)|^from (\w+)", body, re.M)}
    for mod in sorted(imported - stdlib_ok):
        assert alias.get(mod, mod) in declared, (
            f"{path.parent.name} imports {mod!r} without declaring it"
        )


def test_the_runner_uses_the_ephemeral_command_and_scrubbed_env(tmp_path, monkeypatch):
    """Locks the integration this change exists to ship.

    The unit tests cover `training_command` and `child_environment` in
    isolation; nothing asserted `TrainingRunner.run` actually calls them.
    """
    import subprocess

    from labpilot.research_engine.execution.training.runner import TrainingRunner

    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "train.py").write_text(PEP723, encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-must-not-reach-generated-code")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")

    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    TrainingRunner(tmp_path).run(timeout=5)

    assert seen["cmd"][:3] == ["uv", "run", "--script"]
    assert seen["env"] is not None, "the child must not inherit the parent environment"
    assert "OPENROUTER_API_KEY" not in seen["env"]


def test_a_mismatched_quote_is_not_a_dependency():
    """`["']...["']` accepted `"catboost>=1.2'` — a closing quote that does not
    match its opening one. That is not valid TOML, so `uv` rejects the whole
    block at install time, turning one typo into a failed run rather than one
    skipped dependency. The backreference makes a malformed entry simply not a
    dependency, leaving the rest of the block usable.
    """
    from labpilot.research_engine.execution.training.environment import (
        declared_dependencies,
    )

    script = (
        '# /// script\n# dependencies = [\n#   "catboost>=1.2\',\n#   "lightgbm",\n# ]\n# ///\n'
    )
    assert declared_dependencies(script) == ["lightgbm"]


def test_both_quote_styles_still_parse():
    """PEP 723 metadata is TOML, single quotes are valid, and models emit them."""
    from labpilot.research_engine.execution.training.environment import (
        declared_dependencies,
    )

    script = (
        "# /// script\n# dependencies = [\n#   \"catboost>=1.2\",\n#   'lightgbm',\n# ]\n# ///\n"
    )
    assert declared_dependencies(script) == ["catboost>=1.2", "lightgbm"]


def test_a_block_with_no_closing_fence_still_parses():
    """`text.find` returns -1 when the closing marker is absent; line 104
    already substitutes `len(text)`. Pinned because a review reported this as
    unhandled."""
    from labpilot.research_engine.execution.training.environment import (
        declared_dependencies,
    )

    assert declared_dependencies('# /// script\n# dependencies = [\n#   "catboost",\n') == [
        "catboost"
    ]
