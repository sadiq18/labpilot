"""M20 exit criterion 5: a broken artifact fails at the gate that owns it.

The other four criteria ask things of gates one at a time — that each can reject,
that none rebuilds a command production owns, that the corpus is real, that a
derived file says so. This one asks the question those cannot: run the whole
campaign against a deliberately broken artifact and see **where it stops**.

`15-gates-must-fail.md` calls it the check that cannot be satisfied by accident,
because passing it needs the gate to be both *present* and *correct*. A missing
gate shows up as a failure three steps downstream. A gate that fires on
everything shows up as the wrong owner. Only a gate that is there and right
produces a campaign that stops exactly where the defect is.

Measured on rogii 2026-08-08, the truncated `train.py` in the corpus passed
`research_review` **and** `run_smoke_test` and failed at `run_training` — task 10
of 16, seven steps from the codegen that produced it, with the error naming uv
rather than the file. Today it stops at `write_code`, task 3, and the thirteen
tasks after it never run.

The artifact enters the way it really did: as the proposal a codegen agent
returns. Planting it directly in the workspace would test the gates against a
file nothing in the system claims to have written, which is a different and
easier question.

Each case asserts four things. Two of them do the detecting:

* the campaign stopped at the **owning task** — a missing gate downstream of the
  defect, or a gate firing on something it does not own, both show up here;
* the recorded error is that gate's **own reason** — `write_code` has two gates
  this file exercises, and without this they collapse into one.

The other two say the campaign behaved like a campaign: every task before the
owner passed, and every task after it never ran. Both are **entailed** by the
first assertion given today's plan shape — sixteen tasks in a strict chain, all
`abort_on_failure`, and the engine returning on the first error — so neither
currently detects anything the first does not. Measured, on review: deleting them
and re-running six gate mutations killed every one regardless. They are kept
because they state the criterion's own wording ("not three steps downstream") and
would start carrying weight the moment the plan gains a branch or a task that
does not abort; they are documented here rather than removed so nobody reads them
as the part that catches things.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from helpers.campaign import GUARD, HEALTHY, SMOKE_SHORT_PATH, run_baseline_campaign
from helpers.real_failures import real_failure

from labpilot.research_engine.planner.schemas.task_types import TaskStatus, TaskType


@dataclass(frozen=True)
class Broken:
    """One defective `train.py`, the task that must stop it, and why.

    `says` is not decoration. Asserting only the task type cannot tell *which*
    of a task's gates fired, and `write_code` has two that this file exercises:
    the corpus artifact trips the PEP 723 check **and** the entry-point check,
    so with the type alone the two cases collapse into one. Verified by
    neutering the dependency-block check — every test here stayed green while
    case 1 quietly fell through to the other gate.
    """

    label: str
    train_py: str
    owner: TaskType
    says: str
    was: str


_CASES = (
    Broken(
        "truncated codegen output, from the corpus",
        real_failure("truncated_train_py.txt"),
        TaskType.WRITE_CODE,
        "unterminated PEP 723 block",
        "rogii 2026-08-08: passed review and smoke, failed at run_training",
    ),
    Broken(
        "parses, but defines no entry point",
        '"""A plan, not a program."""\n# TODO: write the pipeline\n',
        TaskType.WRITE_CODE,
        'no `if __name__ == "__main__":` guard',
        "the shape the corpus artifact reduces to once its PEP 723 block is closed",
    ),
    Broken(
        "runs, and raises immediately",
        'def main():\n    raise SystemExit("boom")\n' + GUARD,
        TaskType.RUN_SMOKE_TEST,
        "boom",
        "the gate whose whole purpose is to run it once before training does",
    ),
    Broken(
        "trains, and writes no metrics",
        "def main():\n    pass\n" + GUARD,
        TaskType.RUN_TRAINING,
        "did not write metrics.json",
        "exit 0 is not a result; the run has to leave something behind",
    ),
    Broken(
        "produces metrics, and no predictions",
        SMOKE_SHORT_PATH + '    with open("metrics.json", "w") as handle:\n'
        '        json.dump({"cv_accuracy": 0.5}, handle)\n' + GUARD,
        TaskType.RUN_INFERENCE,
        "no predictions.csv and no submission.csv",
        "inference used to fabricate `id,prediction\\n0,0` and pass; fixed in M20 §1",
    ),
)


@pytest.mark.slow
def test_a_healthy_pipeline_completes_every_task(tmp_path, monkeypatch):
    """The control, and it is doing real work: without it, a case that stopped
    at task 3 would be indistinguishable from a harness that cannot reach task 4.
    """
    execution, tasks = run_baseline_campaign(tmp_path, monkeypatch, HEALTHY)

    assert execution.status == "succeeded", execution.error
    assert len(tasks) == 16, "the baseline plan's shape changed; the cases below assume it"
    assert [task.status for task in tasks] == [TaskStatus.DONE] * len(tasks)


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_a_broken_artifact_stops_at_the_gate_that_owns_it(tmp_path, monkeypatch, case):
    """The criterion. See the module docstring for which of these four
    assertions detect, and which are entailed by the plan's shape."""
    execution, tasks = run_baseline_campaign(tmp_path, monkeypatch, case.train_py)

    assert execution.status == "failed", f"nothing stopped it ({case.was})"

    # Exactly one task can be FAILED by construction — the engine resets any
    # prior failure to PENDING and returns on the first error — so asserting
    # that would be asserting the engine, not the campaign.
    failed = next(task for task in tasks if task.status == TaskStatus.FAILED)
    stopped_at = tasks.index(failed)
    error = str(failed.metadata.get("error") or "")

    assert failed.type == case.owner, (
        f"stopped at {failed.type.value}, not {case.owner.value} — {case.was}. Error: {error}"
    )
    assert case.says in error, (
        f"{case.owner.value} stopped it, but for a different reason than this case is "
        f"about. Expected {case.says!r}, got: {error}"
    )
    # Entailed by the assertions above given a strict-chain plan — see the module
    # docstring. Kept as the criterion's own wording, not as detection.
    assert all(task.status == TaskStatus.DONE for task in tasks[:stopped_at]), (
        "a gate ahead of the owner fired on something it does not own: "
        f"{[t.type.value for t in tasks[:stopped_at] if t.status != TaskStatus.DONE]}"
    )
    assert all(task.status == TaskStatus.PENDING for task in tasks[stopped_at + 1 :]), (
        "the campaign carried on past the failure: "
        f"{[t.type.value for t in tasks[stopped_at + 1 :] if t.status != TaskStatus.PENDING]}"
    )


#: The same defect as the corpus artifact — a stdlib name inside the PEP 723
#: block — in a script that declares nothing else and does the job. The corpus
#: file cannot be driven through a campaign here: after `glob` is stripped it
#: still declares pandas, numpy, scikit-learn, lightgbm and pyarrow, so the smoke
#: gate routes to `uv run --script` and resolves five packages against PyPI. That
#: made this the one non-hermetic test in the file — a PyPI blip turned criterion
#: 5's evidence red for a reason unrelated to any gate, and a cold run left
#: ~600 MB in `~/.cache/uv`. Reported reviewing this branch.
_STDLIB_IN_BLOCK = (
    '# /// script\n# requires-python = ">=3.11"\n# dependencies = [\n#   "glob",\n# ]\n# ///\n'
) + HEALTHY


def test_the_corpus_artifact_loses_its_stdlib_entry_and_keeps_the_rest():
    """The repair itself, against the corpus file verbatim.

    `stdlib_dependency_block.txt` declares `glob` among five real packages. uv
    resolves the block as a unit, so that one line made it refuse **all six** and
    the run never started — defect 11, a failure at the very front of the
    campaign whose error named none of them.

    Asserted at the apply layer rather than through a campaign, because driving
    the real file would resolve those five packages over the network.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
        strip_stdlib_dependencies,
    )

    kept, dropped = strip_stdlib_dependencies(real_failure("stdlib_dependency_block.txt"))

    assert dropped == ["glob"]
    assert '"glob"' not in kept, "the stdlib entry survived"
    for real in ("pandas", "numpy", "scikit-learn", "lightgbm>=4.0", "pyarrow"):
        assert f'"{real}"' in kept, f"the strip took {real} with it"


@pytest.mark.slow
def test_a_repaired_defect_stops_nothing(tmp_path, monkeypatch):
    """The other half of ownership, and the reason this file is not just a list
    of failures.

    A stdlib name in the PEP 723 block used to end a campaign before it started.
    `strip_stdlib_dependencies` removes it during apply, so a script carrying
    that defect and nothing else now completes all sixteen tasks — the defect
    gates nothing, which is the correct outcome for one that is prevented rather
    than caught.

    Prevention and rejection are both right answers. What would be wrong is a
    campaign that still stops somewhere for a reason nobody can trace back here.
    """
    execution, tasks = run_baseline_campaign(tmp_path, monkeypatch, _STDLIB_IN_BLOCK)

    assert execution.status == "succeeded", execution.error
    assert all(task.status == TaskStatus.DONE for task in tasks)
    applied = (tmp_path / "competitions" / "demo" / "pipeline" / "train.py").read_text(
        encoding="utf-8"
    )
    assert '"glob"' not in applied, "the stdlib entry survived apply"
