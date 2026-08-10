"""CapabilityExecutor protocol — tools the Research Engineer dispatches to."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


@runtime_checkable
class CapabilityExecutor(Protocol):
    """Stable skill surface; does not own the global queue."""

    name: str

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        ...

    def prepare(self, context: TaskContext) -> None:
        ...

    def execute(self, context: TaskContext) -> TaskEvidence:
        ...

    def verify(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        ...

    def rollback(self, context: TaskContext) -> None:
        ...

    def collect_evidence(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        ...


class BaseCapability:
    """Convenience base with no-op prepare/rollback and pass-through verify."""

    name: str = "base"

    #: Does this capability's `passed` mean *"I checked, and it is sound"*?
    #:
    #: M20's second option, made declarable. A capability whose verdict cannot
    #: be false is a gate that cannot fail — and the fix is either to give it a
    #: failing path, or to stop claiming it verified anything. This is the
    #: second, said out loud: `verifies = False` means the verdict reports that
    #: the step *ran*, not that its result was checked, and the evidence card
    #: says so rather than looking like every other pass.
    #:
    #: Not a way out. `test_every_gate_rejects_something.py` accepts it in place
    #: of a rejection test, so setting it is a claim a reviewer can see.
    #:
    #: It is a *class* declaration, not a field on `TaskEvidence` — the earlier
    #: wording said it appeared "beside `passed` on the card", which nothing
    #: wrote. Reported on PR #120. A capability that declines to verify says so
    #: on its evidence through `checks`, the way `StubCapability` stamps
    #: `stub_no_verification`.
    verifies: bool = True

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset()

    def prepare(self, context: TaskContext) -> None:
        return None

    def verify(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        return evidence

    def rollback(self, context: TaskContext) -> None:
        return None

    def collect_evidence(
        self, context: TaskContext, evidence: TaskEvidence
    ) -> TaskEvidence:
        return evidence

    def execute(self, context: TaskContext) -> TaskEvidence:  # pragma: no cover - abstract
        raise NotImplementedError
