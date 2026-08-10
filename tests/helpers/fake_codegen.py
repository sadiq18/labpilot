"""An LLM double that returns a usable `CodeProposal`.

Tests that assert "write_code produces a runnable pipeline" used to get one for
free from the Jinja template fallback, with `llm_client=None`. M19 §2 deleted
the pack, so codegen with no model now produces nothing and the step fails —
which is the point of the deletion, and not what those tests are about.

They are about applying, overriding and smoke-checking generated code. This
supplies the code so each keeps testing its own subject.

`FakeCodegenLLM` also varies `pipeline/train.py`'s content by the prompt's
`Technique:` line (see `code_engineer_user.j2:7`), so M15's contract test for
`implement` — different technique in, different `train.py` out — can run
without a live LLM call. Existing callers that never set a technique still get
the original fixed `TRAIN` unchanged, so this is additive, not a behaviour
change for them.
"""

from __future__ import annotations

import re

TRAIN = (
    '"""Generated train."""\n'
    "\n"
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    '    Path("metrics.json").write_text(json.dumps({"cv_accuracy": 0.5}))\n'
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)

INFER = (
    '"""Generated infer."""\n'
    "\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def predict() -> None:\n"
    '    Path("submission.csv").write_text("id,prediction\\n0,0\\n")\n'
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    predict()\n"
)


#: Matches the "Technique: <value>" line `code_engineer_user.j2` always
#: renders (empty as "—" when the prompt data carries none).
_TECHNIQUE_LINE = re.compile(r"^Technique: (.*)$", re.MULTILINE)
#: The "Goal:" block, which carries the caller's `description` on the
#: standalone `implement` tool path — where `Technique:` renders as "—"
#: because the technique never reaches the plan metadata the prompt reads
#: (M15 re-audit, 2026-08-11).
_GOAL_BLOCK = re.compile(r"^Goal:\n(.*)$", re.MULTILINE)


def _technique_from_prompt(user: str) -> str:
    """Pull the distinguishing intent back out of the rendered user prompt.

    Parsing the rendered text rather than threading a separate parameter
    through `complete(system, user)` — the double has to match the real
    `LLMClient.complete` signature it stands in for, which only ever sees the
    two rendered strings, the same as the real codegen call does.

    Falls back to the `Goal:` line when `Technique:` is empty. Both are real
    inputs a real model would condition on, and on the standalone `implement`
    tool path the goal is the *only* one that carries the request — see
    `_GOAL_BLOCK`.
    """
    match = _TECHNIQUE_LINE.search(user)
    if match:
        value = match.group(1).strip()
        if value not in {"", "—"}:
            return value
    goal = _GOAL_BLOCK.search(user)
    if goal:
        return goal.group(1).strip()
    return ""


def _train_for(technique: str) -> str:
    if not technique:
        return TRAIN
    return (
        '"""Generated train."""\n'
        "\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"TECHNIQUE = {technique!r}\n"
        "\n"
        "\n"
        "def main() -> None:\n"
        '    Path("metrics.json").write_text(\n'
        '        json.dumps({"cv_accuracy": 0.5, "technique": TECHNIQUE})\n'
        "    )\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


class FakeCodegenLLM:
    """Answers every codegen call with a two-file proposal.

    `pipeline/train.py`'s content varies by the prompt's declared technique;
    `pipeline/infer.py` never varies — infer doesn't depend on which training
    technique was used, so a real codegen call wouldn't vary it either.
    """

    last_served = "fake"

    def complete(self, system: str, user: str) -> str:
        from labpilot.research_engine.execution.schemas.code_proposal import (
            CodeFileSpec,
            CodeProposal,
        )

        train = _train_for(_technique_from_prompt(user))
        return CodeProposal(
            summary="generated baseline",
            rationale="test double",
            files=[
                CodeFileSpec(path="pipeline/train.py", content=train),
                CodeFileSpec(path="pipeline/infer.py", content=INFER),
            ],
        ).model_dump_json()
