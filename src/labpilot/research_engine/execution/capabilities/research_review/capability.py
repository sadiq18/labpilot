"""Research Review — research-correctness gate (LLM optional; deterministic policy)."""

from __future__ import annotations

import ast

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class ResearchReviewCapability(BaseCapability):
    """Gate after code/config changes.

    Without an LLM: deterministic checks (train.py exists + parses). Critical
    findings block. With ``force_block`` in task metadata (tests), always fail.
    """

    name = "research_review"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.RESEARCH_REVIEW})

    def execute(self, context: TaskContext) -> TaskEvidence:
        if context.task.metadata.get("force_block"):
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary="blocked by force_block metadata",
                checks=["force_block"],
                error="critical research finding (forced)",
            )

        train = context.workspace_root / "pipeline" / "train.py"
        findings: list[str] = []
        if not train.is_file():
            findings.append("critical: missing pipeline/train.py")
        else:
            source = train.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                findings.append(f"critical: train.py syntax error: {exc}")
            else:
                if not _has_standard_main_guard(tree):
                    findings.append(
                        "critical: train.py missing standard "
                        "``if __name__ == '__main__':`` entrypoint "
                        "(script would exit without running training)"
                    )

        # Optional LLM judgement slice (soft): never invent metrics; may add notes.
        llm_notes: list[str] = []
        if self._llm is not None and not is_dry_run(context):
            llm_notes.append("llm_client present; deterministic checks still authoritative")

        critical = [f for f in findings if f.startswith("critical:")]
        passed = not critical
        return evidence(
            context,
            capability=self.name,
            passed=passed,
            summary="review passed" if passed else "review blocked",
            checks=["train_exists", "syntax"] + (["llm_note"] if llm_notes else []),
            paths=[str(train)] if train.is_file() else [],
            error="; ".join(critical) if critical else None,
            metadata={
                "findings": findings,
                "llm_notes": llm_notes,
                "decision": "allow" if passed else "block",
            },
        )


def _has_standard_main_guard(tree: ast.AST) -> bool:
    """True when the module has ``if __name__ == "__main__":`` (ASCII exact)."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
            continue
        left, right = test.left, test.comparators[0]
        # ``__name__ == "__main__"`` or ``"__main__" == __name__``
        names = {left, right}
        name_nodes = [n for n in names if isinstance(n, ast.Name) and n.id == "__name__"]
        str_nodes = [
            n
            for n in names
            if isinstance(n, ast.Constant) and n.value == "__main__"
        ]
        if name_nodes and str_nodes:
            return True
    return False
