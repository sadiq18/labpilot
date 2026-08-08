"""`AiderAgent` — M19 §3/§4: buy the edit machinery, build the research parts.

aider already does anchored edits, tuned for years against published edit-format
benchmarks. A spike on rogii's real 331-line `train.py` produced a +24/−8 delta
that left `_driver_columns`, `_add_partition_features`, `_known_rows` and
`partition_suffix_holdout` untouched — §2's core requirement, met without
labpilot writing a line of edit-format code. So this is an *adapter*, not an
edit format.

Three properties are load-bearing and each is enforced here rather than assumed:

**It runs in a copy.** The workspace is never edited by a proposal that has not
been validated. aider edits files in place, so pointing it at the workspace
would mean discovering damage afterwards instead of rejecting it beforehand.

**It goes through the proxy.** Passing `--model` to a provider transfers only
the *selection*; the budget ledger, rate limiting, runtime failover and response
cache all live in `LLMGateway` and would be bypassed. That is §4's "regression
dressed as a feature", so a gateway is required to construct this agent — there
is deliberately no path that runs aider outside M10.

**It emits whole files.** `CodeFileSpec` is path + content, and `apply_proposal`
writes `spec.content` after `ast.parse`. The diff is computed as *evidence* — it
feeds the consistency checks — and is never the thing applied. The delta is how
the model thinks; whole files are how the system applies. Those are allowed to
differ, and merging them would reinvent the apply path this design exists to
delete.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)

logger = logging.getLogger(__name__)

#: Copied into the scratch tree. Editable source only.
#:
#: `data/` is excluded deliberately: it is the bulk of a competition tree, and
#: generated code reads it by path at *run* time, not at edit time. Copying it
#: would make every experiment pay a multi-GB copy for nothing.
EDITABLE_ROOTS: tuple[str, ...] = ("pipeline",)

#: Read-only context aider is allowed to see but not edit.
CONTEXT_FILES: tuple[str, ...] = ("config.yaml", "profile.json")

#: The role the proxy resolves. `labpilot/<role>` is the only model name the
#: proxy accepts — naming a provider model would bypass role selection.
CODEGEN_ROLE = "codegen"

_DEFAULT_TIMEOUT_S = 900


class AiderError(RuntimeError):
    """An aider run that produced no usable proposal.

    ``kind`` is the classification that reaches `agent_invocations`, so the
    failure *rate* is measurable per cause rather than as one bucket. That is
    what step 2 decides on; without it "delta is worse" has no mechanism
    attached.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _default_runner(cmd: Sequence[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run aider via `uvx`, which needs no install step."""
    env = dict(os.environ)
    # The proxy has no auth — it is loopback-only and its single client is this
    # subprocess. litellm still refuses to send without *some* key, so supply a
    # placeholder rather than leaking a real one into a child process.
    env["OPENAI_API_KEY"] = "labpilot-proxy"
    return subprocess.run(  # noqa: S603 - argv is built here, never shell-parsed
        list(cmd),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class AiderAgent:
    """Produce a `CodeProposal` by editing a copy of the parent tree."""

    name = "aider"

    def __init__(
        self,
        gateway,
        *,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
        role: str = CODEGEN_ROLE,
        timeout: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not callable(getattr(gateway, "for_role", None)):
            # Checked by callable, not by attribute presence: a stub carrying
            # `for_role` as a plain value would otherwise crash later, inside
            # the proxy, where the cause is much harder to see.
            raise ValueError(
                "AiderAgent requires an LLMGateway. Without one aider calls the "
                "provider directly, which bypasses the budget ledger, rate "
                "limiting and failover — M19 §4 names that a regression."
            )
        self._gateway = gateway
        self._runner = runner or _default_runner
        self._role = role
        self._timeout = timeout

    # -- the protocol ------------------------------------------------------

    def propose(self, ctx: StructuredContext, parent: Path | None) -> CodeProposal:
        """Edit a copy of `parent` and return the resulting files.

        A baseline has no parent, so there is nothing to diff against and
        nothing to preserve. That is `WholeFileAgent`'s job, and raising here
        rather than silently regenerating keeps the routing decision visible.
        """
        if parent is None:
            raise AiderError(
                "no parent tree: a baseline is the whole-file agent's job",
                kind="no_parent",
            )
        parent = Path(parent)
        if not parent.is_dir():
            raise AiderError(f"parent tree not found: {parent}", kind="no_parent")

        instruction = _instruction(ctx)
        with tempfile.TemporaryDirectory(prefix="labpilot-aider-") as scratch_str:
            scratch = Path(scratch_str)
            edit_targets = _copy_tree(parent, scratch)
            if not edit_targets:
                raise AiderError(
                    f"nothing editable under {parent}: expected {EDITABLE_ROOTS}",
                    kind="no_source",
                )
            before = {rel: (scratch / rel).read_bytes() for rel in edit_targets}

            self._run_aider(scratch, edit_targets, instruction)

            changed = [rel for rel in edit_targets if (scratch / rel).read_bytes() != before[rel]]
            if not changed:
                # §9.6: a no-op run is a failure, not a silent success — the
                # same lesson as the stale `metrics.json` guard, which asked
                # "is there a file?" instead of "did this run write one?".
                raise AiderError("aider made no edit", kind="aider_no_edit")

            files = [
                CodeFileSpec(path=rel, content=(scratch / rel).read_text(encoding="utf-8"))
                for rel in changed
            ]

        return CodeProposal(
            summary=f"aider edited {len(files)} file(s)",
            rationale=instruction,
            files=files,
        )

    # -- internals ---------------------------------------------------------

    def _run_aider(self, scratch: Path, edit_targets: list[str], instruction: str) -> None:
        """Run aider against a proxy scoped to this call.

        The design scopes the proxy to the campaign. Scoping it to the call is
        strictly tighter — it cannot outlive the request it accounts for — and
        costs one loopback socket per proposal. Revisit if per-call startup ever
        shows up in a campaign's timings.
        """
        from fitroute.server import ProxyServer

        with ProxyServer(self._gateway) as proxy:
            cmd = _aider_command(proxy.base_url, self._role, edit_targets, instruction)
            try:
                result = self._runner(cmd, scratch, self._timeout)
            except subprocess.TimeoutExpired as exc:
                raise AiderError(
                    f"aider timed out after {self._timeout}s", kind="aider_timeout"
                ) from exc
            except FileNotFoundError as exc:
                raise AiderError(
                    "aider is not runnable (uvx missing?); the whole-file agent "
                    "is the supported path in a workspace without it",
                    kind="aider_missing",
                ) from exc

        if getattr(result, "returncode", 0) != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            raise AiderError(
                f"aider exited {result.returncode}: {stderr[-500:]}",
                kind="aider_failed",
            )


def _instruction(ctx: StructuredContext) -> str:
    """What to ask aider for, drawn from the hypothesis the plan already holds."""
    data = getattr(ctx, "data", None) or {}
    parts = [
        str(data.get("plan_goal") or "").strip(),
        str(data.get("prediction") or "").strip(),
        str(data.get("reason") or "").strip(),
    ]
    body = "\n\n".join(p for p in parts if p)
    return body or "Improve the training script for this competition."


def _copy_tree(parent: Path, scratch: Path) -> list[str]:
    """Copy editable roots plus read-only context; return editable relpaths."""
    editable: list[str] = []
    for root in EDITABLE_ROOTS:
        src = parent / root
        if not src.is_dir():
            continue
        shutil.copytree(src, scratch / root, dirs_exist_ok=True)
        editable.extend(
            str(path.relative_to(scratch))
            for path in sorted((scratch / root).rglob("*.py"))
            if path.is_file()
        )
    for name in CONTEXT_FILES:
        src = parent / name
        if src.is_file():
            shutil.copy2(src, scratch / name)
    return editable


def _aider_command(
    base_url: str,
    role: str,
    edit_targets: Sequence[str],
    instruction: str,
) -> list[str]:
    """The argv, kept in one place so the contract with the proxy is readable.

    `openai/labpilot/<role>` is what litellm sends as model `labpilot/<role>`,
    which is the only form the proxy accepts. Streaming is off because an
    OpenAI-compatible stream omits `usage` unless the provider honours
    `stream_options`, and an unmetered call defeats the ledger the proxy exists
    to feed.
    """
    return [
        "uvx",
        "--from",
        "aider-chat",
        "aider",
        "--model",
        f"openai/labpilot/{role}",
        "--openai-api-base",
        base_url,
        "--no-stream",
        "--yes-always",
        "--no-git",
        "--no-auto-commits",
        "--no-check-update",
        "--no-analytics",
        "--message",
        instruction,
        *edit_targets,
    ]
