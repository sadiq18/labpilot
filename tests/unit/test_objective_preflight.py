"""A campaign whose objective cannot be justified does not start.

Checked at launch rather than mid-run, and the placement is the whole design:
refusing to *start* an unattended job costs nothing and reaches the operator
while they are still at the keyboard. Halting at 2am reaches nobody until
morning and throws the night away.

Both live workspaces on this machine fail it, for different reasons:

* `rogii/competition.json` has `evaluation_metric: None` — two weeks of
  campaigns against no stated objective at all.
* `playground-series-s6e7` states `balanced_accuracy_score`, which nothing here
  can compute, so cross-validation optimised plain accuracy and never said so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from labpilot.cli.conduct import _preflight_objective, _stated_objective


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root


def _workspace(tmp_path: Path, metric: dict | None, **extra) -> _Workspace:
    spec = {"slug": "demo", "problem_type": "tabular_regression", **extra}
    spec["evaluation_metric"] = metric
    (tmp_path / "competition.json").write_text(json.dumps(spec), encoding="utf-8")
    return _Workspace(tmp_path)


def _blocks(ws) -> bool:
    try:
        _preflight_objective(ws, "demo", assume_yes=True)
    except typer.Exit as exit_:
        assert exit_.exit_code == 2, "a refusal must be distinguishable from success"
        return True
    return False


# --- the two live workspaces ------------------------------------------------


def test_a_competition_with_no_stated_metric_never_starts(tmp_path: Path) -> None:
    """rogii, exactly."""
    assert _blocks(_workspace(tmp_path, None))


def test_a_metric_nothing_can_compute_never_starts(tmp_path: Path) -> None:
    """playground-series-s6e7, exactly. The campaign would have optimised a
    proxy and recorded it under the stated metric's name."""
    ws = _workspace(
        tmp_path,
        {"name": "balanced_accuracy_score", "direction": "maximize"},
        problem_type="tabular_classification",
    )
    assert _blocks(ws)


def test_a_declaration_that_contradicts_the_evaluator_never_starts(tmp_path: Path) -> None:
    """The failure that cost rogii all fifteen of its evidence cards."""
    assert _blocks(_workspace(tmp_path, {"name": "rmse", "direction": "maximize"}))


def test_a_resolved_objective_starts(tmp_path: Path) -> None:
    """The gate must not cost the behaviour it guards."""
    assert not _blocks(_workspace(tmp_path, {"name": "rmse", "direction": "minimize"}))


def test_a_workspace_with_no_contract_at_all_never_starts(tmp_path: Path) -> None:
    """No `competition.json` is not permission to proceed."""
    assert _blocks(_Workspace(tmp_path))


# --- the safety property: --yes must never answer a question ----------------


def test_yes_never_prompts_even_on_a_terminal(tmp_path: Path, monkeypatch) -> None:
    """The single most consequential line in this design.

    Auto-answering "which fact is true?" is how a wrong objective gets frozen
    into a workspace — and everything downstream reuses it. `--yes` may
    auto-approve *actions the operator asked for*; it must never decide a fact.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    asked: list[str] = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: asked.append(a[0] if a else ""))

    with pytest.raises(typer.Exit):
        _preflight_objective(_workspace(tmp_path, None), "demo", assume_yes=True)

    assert asked == [], "--yes reached the prompt"


def test_a_non_interactive_run_never_prompts(tmp_path: Path, monkeypatch) -> None:
    """A cron job or a CI run has nobody to answer, so blocking on input would
    hang instead of failing. Same rule `verify_artifact` follows."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    asked: list[str] = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: asked.append("asked"))

    with pytest.raises(typer.Exit):
        _preflight_objective(_workspace(tmp_path, None), "demo", assume_yes=False)

    assert asked == []


def test_an_operator_at_a_terminal_is_offered_the_choice(tmp_path: Path, monkeypatch) -> None:
    """Interactive is where a human can weigh it, so the refusal is a question
    rather than a wall — and declining still stops the run."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)

    _preflight_objective(_workspace(tmp_path, None), "demo", assume_yes=False)  # proceeds

    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit):
        _preflight_objective(_workspace(tmp_path, None), "demo", assume_yes=False)


# --- reading the contract ---------------------------------------------------


def test_a_malformed_contract_is_not_read_as_an_objective(tmp_path: Path) -> None:
    """Unparseable is unknown, and unknown blocks. It must not read as 'no
    constraint stated'."""
    (tmp_path / "competition.json").write_text("{ not json", encoding="utf-8")

    assert _stated_objective(_Workspace(tmp_path), "demo") == (None, None, None, None)
    assert _blocks(_Workspace(tmp_path))


def test_a_bogus_direction_is_discarded_rather_than_trusted(tmp_path: Path) -> None:
    """Only the two real values are accepted; anything else falls through to the
    probe rather than being carried as a declaration."""
    ws = _workspace(tmp_path, {"name": "rmse", "direction": "sideways"})

    _raw, declared, _task, _target = _stated_objective(ws, "demo")
    assert declared is None
    assert not _blocks(ws), "the probe should still resolve rmse on its own"


def test_the_legacy_metric_key_is_read_too(tmp_path: Path) -> None:
    """Hand-written contracts use `metric`; the parser writes `evaluation_metric`.
    `direction.py` reads both for the same reason."""
    (tmp_path / "competition.json").write_text(
        json.dumps({"slug": "demo", "metric": {"name": "rmse", "direction": "minimize"}}),
        encoding="utf-8",
    )

    raw, declared, _task, _target = _stated_objective(_Workspace(tmp_path), "demo")
    assert raw == "rmse"
    assert declared == "minimize"


# --- review findings --------------------------------------------------------


def test_a_resolved_objective_is_recorded_on_the_session(tmp_path: Path) -> None:
    """What the preflight returns is stamped into the session metadata, so a
    campaign's objective is legible long after the console line is gone."""
    meta = _preflight_objective(
        _workspace(tmp_path, {"name": "rmse", "direction": "minimize"}),
        "demo",
        assume_yes=True,
    )

    assert meta["objective_metric"] == "rmse"
    assert meta["objective_direction"] == "minimize"
    assert meta["objective_confidence"] > 0


def test_an_override_is_recorded_not_just_printed(tmp_path: Path, monkeypatch) -> None:
    """Review finding. Confirming "run anyway" was announced on the console and
    stored nowhere, so a campaign built on an unknown direction was afterwards
    indistinguishable from a resolved one — the same class of failure as rogii's
    fifteen cards, with a console line as its only trace."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)

    meta = _preflight_objective(_workspace(tmp_path, None), "demo", assume_yes=False)

    assert meta["objective_override"] is True
    assert "no evaluation metric" in meta["objective_blocked_reason"]


def test_the_candidates_reach_the_operator(tmp_path: Path, capsys) -> None:
    """A question with no options is the wall the alternatives field exists to
    prevent."""
    ws = _workspace(tmp_path, {"name": "rmse", "direction": "maximize"})
    with pytest.raises(typer.Exit):
        _preflight_objective(ws, "demo", assume_yes=True)

    assert "candidates:" in capsys.readouterr().out


def test_the_target_column_reaches_the_objective(tmp_path: Path) -> None:
    """Review finding. `ObjectiveSpec.target` was declared, documented and never
    populated — every shipped spec carried None while reading as though the
    field were live. `competition.json` has no target; `profile.json` does."""
    ws = _workspace(tmp_path, {"name": "rmse", "direction": "minimize"})
    (tmp_path / "profile.json").write_text(json.dumps({"target_column": "TVT"}), encoding="utf-8")

    assert _stated_objective(ws, "demo")[3] == "TVT"
    assert _preflight_objective(ws, "demo", assume_yes=True)["objective_target"] == "TVT"


def test_a_missing_or_broken_profile_is_not_a_target(tmp_path: Path) -> None:
    """Absent, unparseable and target-less must all read as 'not stated', and
    none of them may stop a resolvable objective from launching."""
    ws = _workspace(tmp_path, {"name": "rmse", "direction": "minimize"})
    assert _stated_objective(ws, "demo")[3] is None

    (tmp_path / "profile.json").write_text("{ not json", encoding="utf-8")
    assert _stated_objective(ws, "demo")[3] is None

    (tmp_path / "profile.json").write_text(json.dumps({"target_column": None}), encoding="utf-8")
    assert _stated_objective(ws, "demo")[3] is None
    assert not _blocks(ws)


def test_the_printed_confidence_names_the_source_that_capped_it(tmp_path: Path, capsys) -> None:
    """Review finding. The line read `(minimize, from measured, confidence 0.90)`
    — pairing the direction's source with a confidence the *registry* set. The
    model was fixed to report the capping source and the console line was not,
    so the one an operator actually reads still mismatched."""
    _preflight_objective(
        _workspace(tmp_path, {"name": "rmse", "direction": "minimize"}), "demo", assume_yes=True
    )

    out = capsys.readouterr().out
    assert "minimize from measured" in out
    assert "capped by registry" in out


def test_resuming_a_session_is_gated_too(tmp_path: Path) -> None:
    """Review finding. The gate covered `run` only, so a contract edited between
    sessions let `continue` and `resume` step against an objective `run` would
    refuse. Both route through `_continue_session`.
    """
    import inspect

    from labpilot.cli import conduct

    source = inspect.getsource(conduct._continue_session)
    assert "_preflight_objective" in source


def test_a_campaign_started_under_an_override_can_still_be_resumed() -> None:
    """Review finding. `_continue_session` gated with `assume_yes=True`, which
    never prompts and always exits — so a campaign the operator had deliberately
    launched under an unresolved objective could never be continued. The launch
    accepted the answer; every resume afterwards refused it.

    Asserted on the source because the alternative is standing up a store, a
    workspace and a registry to observe one branch.
    """
    import inspect

    from labpilot.cli import conduct

    source = inspect.getsource(conduct._continue_session)
    gate = source.index("_preflight_objective")
    guard = source.index('meta.get("objective_override")')

    assert guard < gate, "the gate runs before the override is read"
    assert source.index("meta = dict(session.metadata)") < guard, (
        "the override is read before the session is loaded"
    )
