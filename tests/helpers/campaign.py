"""Run the real baseline plan end to end, with a chosen `pipeline/train.py`.

Shared by M20 criteria 4 and 5. Why the constraints are what they are is in
`docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
from labpilot.research_engine.planner.schemas.task_types import TaskType

COMPETITION = "demo"

GUARD = '\n\nif __name__ == "__main__":\n    main()\n'

#: Fixtures return early under the smoke gate's flag. Without it they write
#: their artifacts during smoke, and the training and inference gates then find
#: evidence they did not produce.
SMOKE_SHORT_PATH = (
    "import json\n"
    "import os\n"
    "\n"
    "\n"
    "def main():\n"
    '    if os.environ.get("LABPILOT_SMOKE"):\n'
    "        return\n"
)

#: Trains, records a metric, writes predictions — a campaign that completes.
HEALTHY = (
    SMOKE_SHORT_PATH + '    with open("metrics.json", "w") as handle:\n'
    '        json.dump({"cv_accuracy": 0.5}, handle)\n'
    '    with open("predictions.csv", "w") as handle:\n'
    '        handle.write("id,y\\n0,0\\n1,0\\n")\n' + GUARD
)


class FakeKaggle:
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


def codegen_returning(content: str):
    """An agent whose proposal is `content`.

    `last_used_llm` keeps `origin` off `last_resort`, which would fail the step
    before apply and hide whatever the content was meant to exercise.
    """

    class _Agent:
        last_used_llm = True

        def run(self, ctx):
            return CodeProposal(
                summary="artifact under test",
                files=[CodeFileSpec(path="pipeline/train.py", content=content, action="write")],
            )

    return _Agent()


def run_baseline_campaign(tmp_path: Path, monkeypatch, train_py: str):
    """Execute the sixteen-task baseline plan; return `(execution, tasks)`.

    Not a dry run: `dry_run` makes the smoke gate syntax-only and stubs training
    and inference, so a script that raises on its first line completes every task.
    """
    monkeypatch.setattr(
        "labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt",
        lambda *args, **kwargs: "",
    )
    # `child_environment()` forwards the whole environment to the training
    # subprocess, so an exported LABPILOT_SMOKE reaches the fixtures there too.
    monkeypatch.delenv("LABPILOT_SMOKE", raising=False)

    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, COMPETITION).ensure()
    paths.report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "competition": COMPETITION,
                "techniques": {"items": []},
                "retrieval": {"queries": []},
            }
        ),
        encoding="utf-8",
    )
    compile_baseline_plan(COMPETITION, knowledge_dir=knowledge, llm_client=None)

    registry = default_capability_registry(install_packages=False)
    registry.require(TaskType.WRITE_CODE)._agent = codegen_returning(train_py)

    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition=COMPETITION,
        registry=registry,
        constraints={
            "kaggle": KaggleConfig(cache_dir=tmp_path / "cache"),
            "kaggle_client": FakeKaggle(),
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
