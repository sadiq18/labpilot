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
rather than the file. Today it stops at `write_code`, task 3, and the sixteen
tasks after it never run.

The artifact enters the way it really did: as the proposal a codegen agent
returns. Planting it directly in the workspace would test the gates against a
file nothing in the system claims to have written, which is a different and
easier question.

Each case asserts three things, and the second and third are what make it a
statement about *ownership* rather than about failure:

* the campaign stopped at the owning task;
* every task before it **passed** — so the defect was not caught early by luck,
  and the gates ahead of the owner did not fire on something they do not own;
* every task after it never ran — so nothing downstream is doing the owner's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from helpers.real_failures import real_failure

from labpilot.config import KaggleConfig, ProfilerConfig
from labpilot.research_engine.execution.engineer import (
    ResearchEngineer,
    default_capability_registry,
)
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner import compile_baseline_plan
from labpilot.research_engine.planner.schemas.task_types import TaskStatus, TaskType

_GUARD = '\n\nif __name__ == "__main__":\n    main()\n'

#: A script that does everything the plan asks: trains, records a metric, and
#: writes predictions. The control — without it, "the campaign stopped at task N"
#: could mean the harness cannot get past task N at all.
_HEALTHY = (
    "import json\n"
    "\n"
    "\n"
    "def main():\n"
    '    with open("metrics.json", "w") as handle:\n'
    '        json.dump({"cv_accuracy": 0.5}, handle)\n'
    '    with open("predictions.csv", "w") as handle:\n'
    '        handle.write("id,y\\n0,0\\n1,0\\n")\n' + _GUARD
)


@dataclass(frozen=True)
class Broken:
    """One defective `train.py` and the task that must be the one to stop it."""

    label: str
    train_py: str
    owner: TaskType | None
    was: str


_CASES = (
    Broken(
        "truncated codegen output, from the corpus",
        real_failure("truncated_train_py.txt"),
        TaskType.WRITE_CODE,
        "rogii 2026-08-08: passed review and smoke, failed at run_training",
    ),
    Broken(
        "parses, but defines no entry point",
        '"""A plan, not a program."""\n# TODO: write the pipeline\n',
        TaskType.WRITE_CODE,
        "the shape the corpus artifact reduces to once its PEP 723 block is closed",
    ),
    Broken(
        "runs, and raises immediately",
        'def main():\n    raise SystemExit("boom")\n' + _GUARD,
        TaskType.RUN_SMOKE_TEST,
        "the gate whose whole purpose is to run it once before training does",
    ),
    Broken(
        "trains, and writes no metrics",
        "def main():\n    pass\n" + _GUARD,
        TaskType.RUN_TRAINING,
        "exit 0 is not a result; the run has to leave something behind",
    ),
    Broken(
        "produces metrics, and no predictions",
        "import json\n\n\ndef main():\n"
        '    with open("metrics.json", "w") as handle:\n'
        '        json.dump({"cv_accuracy": 0.5}, handle)\n' + _GUARD,
        TaskType.RUN_INFERENCE,
        "inference used to fabricate `id,prediction\\n0,0` and pass; fixed in M20 §1",
    ),
)


class _FakeKaggle:
    """Enough of a competition to profile: two features, a binary target."""

    def download_competition(self, slug: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]}).to_csv(
            dest / "train.csv", index=False
        )
        pd.DataFrame({"id": [0, 1], "x": [1.5, 2.5]}).to_csv(dest / "test.csv", index=False)
        pd.DataFrame({"id": [0, 1], "y": [0, 0]}).to_csv(
            dest / "sample_submission.csv", index=False
        )


def _codegen_returning(content: str):
    """An agent whose proposal is the artifact under test.

    `last_used_llm` because without it `origin` becomes `last_resort` and the
    step fails *before* apply — which would make every case here stop at
    `write_code` for a reason that has nothing to do with the artifact.
    """

    class _Agent:
        last_used_llm = True

        def run(self, ctx):
            return CodeProposal(
                summary="artifact under test",
                files=[CodeFileSpec(path="pipeline/train.py", content=content, action="write")],
            )

    return _Agent()


def _run_campaign(tmp_path: Path, monkeypatch, train_py: str):
    """The real baseline plan, sixteen tasks, through the real registry.

    Not a dry run: `dry_run` makes the smoke gate syntax-only and stubs training,
    so a script that crashes when executed completes all sixteen tasks. Verified
    while writing this — under `{"dry_run": True}` every case below passes.
    """
    monkeypatch.setattr(
        "labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt",
        lambda *args, **kwargs: "",
    )

    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, "demo").ensure()
    paths.report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "competition": "demo",
                "techniques": {"items": []},
                "retrieval": {"queries": []},
            }
        ),
        encoding="utf-8",
    )
    compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)

    registry = default_capability_registry(install_packages=False)
    registry.require(TaskType.WRITE_CODE)._agent = _codegen_returning(train_py)

    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=registry,
        constraints={
            "kaggle": KaggleConfig(cache_dir=tmp_path / "cache"),
            "kaggle_client": _FakeKaggle(),
            "profiler": ProfilerConfig(),
            "skip_download": False,
            "dry_run": False,
            "allow_upload": False,
        },
    )
    try:
        execution = engineer.run_plan("P-001")
        plan = engineer._plan_store.get_plan("P-001")
        assert plan is not None
        return execution, sorted(plan.tasks, key=lambda task: task.order)
    finally:
        engineer.close()


@pytest.mark.slow
def test_a_healthy_pipeline_completes_every_task(tmp_path, monkeypatch):
    """The control, and it is doing real work: without it, a case that stopped
    at task 3 would be indistinguishable from a harness that cannot reach task 4.
    """
    execution, tasks = _run_campaign(tmp_path, monkeypatch, _HEALTHY)

    assert execution.status == "succeeded", execution.error
    assert len(tasks) == 16, "the baseline plan's shape changed; the cases below assume it"
    assert [task.status for task in tasks] == [TaskStatus.DONE] * len(tasks)


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_a_broken_artifact_stops_at_the_gate_that_owns_it(tmp_path, monkeypatch, case):
    """The criterion. See the module docstring for why the two ordering
    assertions carry more weight than the first."""
    execution, tasks = _run_campaign(tmp_path, monkeypatch, case.train_py)

    assert execution.status == "failed", f"nothing stopped it ({case.was})"

    failed = [task for task in tasks if task.status == TaskStatus.FAILED]
    assert len(failed) == 1, f"more than one gate fired: {[t.type.value for t in failed]}"

    stopped_at = tasks.index(failed[0])
    assert failed[0].type == case.owner, (
        f"stopped at {failed[0].type.value}, not {case.owner.value} — {case.was}. "
        f"Error: {failed[0].metadata.get('error')}"
    )
    assert all(task.status == TaskStatus.DONE for task in tasks[:stopped_at]), (
        "a gate ahead of the owner fired on something it does not own: "
        f"{[t.type.value for t in tasks[:stopped_at] if t.status != TaskStatus.DONE]}"
    )
    assert all(task.status == TaskStatus.PENDING for task in tasks[stopped_at + 1 :]), (
        "the campaign carried on past the failure: "
        f"{[t.type.value for t in tasks[stopped_at + 1 :] if t.status != TaskStatus.PENDING]}"
    )


@pytest.mark.slow
def test_a_repaired_defect_stops_nothing_and_the_real_one_still_does(tmp_path, monkeypatch):
    """The other half of ownership, and the reason this file is not just a list
    of failures.

    The corpus's `stdlib_dependency_block.txt` declares `glob` in its PEP 723
    block. uv refused all **six** dependencies over that one line and the run
    never started — defect 11, a failure at the very front of the campaign whose
    error named none of the six. `strip_stdlib_dependencies` removes it during
    apply, so that defect gates nothing now.

    The file is otherwise a runnable script that writes no metrics, so the
    campaign still stops — at `run_training`, ten tasks later, for the thing that
    is actually wrong with it. That is the point: prevention moved the stopping
    point from a dependency resolver that could not name the problem to the gate
    that owns the problem the artifact still has.

    Asserting `succeeded` here would have been wrong, and was: the first version
    of this test claimed the artifact completes a campaign, which is only true
    under `train_stub`, where nothing runs it.
    """
    execution, tasks = _run_campaign(
        tmp_path, monkeypatch, real_failure("stdlib_dependency_block.txt")
    )
    by_type = {task.type: task for task in tasks}

    # The repaired defect: apply strips the stdlib entry, so every step that the
    # dependency block used to kill now passes.
    assert by_type[TaskType.WRITE_CODE].status == TaskStatus.DONE
    assert by_type[TaskType.INSTALL_PACKAGE].status == TaskStatus.DONE
    assert by_type[TaskType.RUN_SMOKE_TEST].status == TaskStatus.DONE
    applied = tmp_path / "competitions" / "demo" / "pipeline" / "train.py"
    block = applied.read_text(encoding="utf-8")
    assert '"glob"' not in block, "the stdlib entry survived apply"
    assert '"pandas"' in block, "apply dropped the real dependencies with it"

    # What is still wrong with it, at the gate that owns that.
    assert execution.status == "failed"
    failed = [task for task in tasks if task.status == TaskStatus.FAILED]
    assert [task.type for task in failed] == [TaskType.RUN_TRAINING]
