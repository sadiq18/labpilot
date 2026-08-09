"""An LLM double that returns a usable `CodeProposal`.

Tests that assert "write_code produces a runnable pipeline" used to get one for
free from the Jinja template fallback, with `llm_client=None`. M19 §2 deleted
the pack, so codegen with no model now produces nothing and the step fails —
which is the point of the deletion, and not what those tests are about.

They are about applying, overriding and smoke-checking generated code. This
supplies the code so each keeps testing its own subject.
"""

from __future__ import annotations

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


class FakeCodegenLLM:
    """Answers every codegen call with the same two-file proposal."""

    last_served = "fake"

    def complete(self, system: str, user: str) -> str:
        from labpilot.research_engine.execution.schemas.code_proposal import (
            CodeFileSpec,
            CodeProposal,
        )

        return CodeProposal(
            summary="generated baseline",
            rationale="test double",
            files=[
                CodeFileSpec(path="pipeline/train.py", content=TRAIN),
                CodeFileSpec(path="pipeline/infer.py", content=INFER),
            ],
        ).model_dump_json()
