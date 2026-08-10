"""Runtime capability — select / record local (and dry-run remote) environments."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.capabilities._helpers import evidence, is_dry_run
from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType

logger = logging.getLogger(__name__)


def _registry_is_configured(directories: tuple[Path, ...]) -> bool:
    """Has this workspace registered runtimes of its own?

    Asked of the **files**, not of a count. `len(list_runtimes(...)) > 1` looked
    equivalent and was not: the shipped `configs/runtimes/` holds exactly one
    file, `local-default.yaml`, which *overrides the builtin of the same id* — so
    the merged registry still has one entry and the refusal branch was
    unreachable for every workspace using the shipped directory. A count cannot
    tell "nothing registered" from "one thing registered that replaced a
    builtin". Reported on PR #120.
    """
    return any(d.is_dir() and any(d.glob("*.yaml")) for d in directories)


def _runtimes_dirs(context: TaskContext) -> tuple[Path, ...]:
    """Where this workspace keeps its runtime definitions, if anywhere.

    The registry takes directories and this call site passed none, so only the
    builtin ever resolved. Read from the workspace config, the way
    `resolve_runtimes_dir` does for the CLI.
    """
    from labpilot.research_engine.execution.codegen_strategy import workspace_config_path

    explicit = context.constraints.get("runtimes_dir")
    if explicit:
        return (Path(explicit),)
    config_path = workspace_config_path(context.workspace_root)
    if config_path is None or not config_path.is_file():
        return ()
    try:
        from labpilot.config import load_config, resolve_runtimes_dir

        return (resolve_runtimes_dir(load_config(config_path)),)
    except Exception as exc:  # noqa: BLE001 — reported by the caller's verdict
        logger.debug("runtimes directory unreadable: %s", exc)
        return ()


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
                checks=["idempotent_skip_redispatch", "no_verification"],
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

            # With the workspace's runtimes directory, the way `research
            # runtime show` resolves them. Called with none, this saw the single
            # builtin and every real name — `kaggle-gpu`, `local`, `a100` —
            # failed to resolve. Reported on PR #120: the one-line fix every
            # other call site already had.
            directories = _runtimes_dirs(context)
            runtime = get_runtime(str(runtime_id), *directories)
            if runtime is not None:
                provider = getattr(runtime, "provider", "local")
                runtime_id = runtime.id
            elif requested and _registry_is_configured(directories):
                # Only when resolution is configured here. `get_runtime` is
                # called with no directories, so it sees the single builtin and
                # nothing else — meaning *every* name fails to resolve on a
                # workspace that has not registered any runtimes, including
                # `local` and the shipped `kaggle-gpu` example. The first
                # version refused all of them. Reported on PR #120, against the
                # shipped config rather than the fabricated name the test used.
                #
                # With runtimes registered, an unknown name is a real mistake and
                # substituting the default would run the experiment elsewhere.
                # Without any, this step cannot answer the question and says so
                # rather than inventing a verdict.
                unresolved = f"runtime {requested!r} is not registered"
            elif requested:
                logger.warning(
                    "no runtimes are registered, so %r could not be checked; continuing on %s",
                    requested,
                    self._default,
                )
                # The *name* goes back to the default too. Keeping the
                # unresolved one made the card read `selected runtime a100` and
                # wrote `runtime_id: a100, provider: local` into
                # `artifacts/runtime.json` — the original M20 defect with the
                # card now positively asserting the runtime it did not get,
                # which is worse than the version that at least said "local".
                # Reported on PR #120.
                runtime_id = self._default
        except ImportError as exc:
            # Narrowed from `except Exception`, which caught a `NameError` from
            # this module's own missing `logger` and reported it as "the runtime
            # is not registered" — a bug in the handler, dressed up as a finding
            # about the user's config. The registry failing to import is the
            # only fault this can honestly diagnose. Reported on PR #120.
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
