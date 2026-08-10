"""Runtime capability — select / record local (and dry-run remote) environments."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class RuntimeCapability(BaseCapability):
    """Deterministic runtime selection. Never uses an LLM."""

    name = "runtime"

    def __init__(self, *, default_runtime_id: str = "local-default") -> None:
        self._default = default_runtime_id

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.SELECT_RUNTIME})

    def execute(self, context: TaskContext) -> TaskEvidence:
        # Idempotent resume: reuse prior job/runtime if recorded.
        prior = (context.prior_evidence.metadata if context.prior_evidence else {}) or {}
        existing_job = prior.get("job_id") or context.task.metadata.get("job_id")
        if existing_job:
            return evidence(
                context,
                capability=self.name,
                passed=True,
                summary=f"runtime job already active: {existing_job}",
                checks=["idempotent_skip_redispatch"],
                metadata={
                    "job_id": existing_job,
                    "runtime_id": prior.get("runtime_id", self._default),
                    "redispatched": False,
                },
            )

        requested = context.constraints.get("runtime_id") or context.execution.runtime_target
        runtime_id = requested or self._default
        provider = "local"
        # A runtime that was *asked for* and could not be resolved is the one
        # thing this step can get wrong, and it used to be the one thing it
        # could not report: the lookup failure fell through to the local default
        # and the step said "selected runtime local". A campaign that asked for
        # a GPU then trained somewhere else, with a passing card. M20 finding,
        # 2026-08-09.
        unresolved = ""
        try:
            from labpilot.research_engine.execution.runtimes.registry import get_runtime

            runtime = get_runtime(str(runtime_id))
            if runtime is not None:
                provider = getattr(runtime, "provider", "local")
                runtime_id = runtime.id
            elif requested:
                unresolved = f"runtime {requested!r} is not registered"
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            runtime_id = self._default
            if requested:
                unresolved = f"runtime {requested!r} could not be resolved: {exc}"

        if unresolved:
            return evidence(
                context,
                capability=self.name,
                passed=False,
                summary=f"requested runtime unavailable; refusing to substitute {self._default}",
                checks=["select_runtime"],
                error=(
                    f"{unresolved}. Falling back to {self._default!r} would run the "
                    "experiment somewhere other than the one it asked for, and "
                    "report success for it."
                ),
                metadata={"requested_runtime": str(requested), "fallback": self._default},
            )

        if provider != "local" and (
            is_dry_run(context) or context.constraints.get("remote_dry_run", True)
        ):
            # Remote path: dry-run / mock dispatch for CI.
            job_id = f"dry-{runtime_id}-{context.execution.id}"
            record = {
                "runtime_id": runtime_id,
                "provider": provider,
                "job_id": job_id,
                "status": "dry_run_dispatched",
                "dispatched_at": datetime.now(UTC).isoformat(),
            }
        else:
            job_id = f"local-{context.execution.id}"
            record = {
                "runtime_id": runtime_id,
                "provider": "local",
                "job_id": job_id,
                "status": "ready",
                "dispatched_at": datetime.now(UTC).isoformat(),
            }

        out = context.workspace_root / "artifacts" / "runtime.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        return evidence(
            context,
            capability=self.name,
            passed=True,
            summary=f"selected runtime {runtime_id}",
            checks=["select_runtime"],
            paths=[str(out)],
            metadata={**record, "redispatched": True},
        )
