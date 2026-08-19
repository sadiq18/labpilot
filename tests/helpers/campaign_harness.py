"""A campaign you can run in milliseconds, and make fail on purpose.

Every defect this exists to catch was found by running a real campaign against
a real competition: twenty to sixty minutes per attempt, one bug per attempt,
and nine of the twelve found that way were deterministic logic that never
needed a model or a network at all.

`tests/unit/test_campaigns.py` already scaffolds a real `Workspace`, a real
`ConductorStore` and drives the real `run_until_stop`. Two things stopped it
from finding any of them:

* every tool is an echo that writes a file and returns success, so a harness
  built on it cannot express the one shape that keeps recurring — *a tool that
  reports success having changed nothing*;
* every test passes ``prefer_offline=True``, so the policy path — schema
  validation, the gated-tool retry, the offline fallback — never runs.

So this module adds the two things that were missing, plus the seeded domain
state the gating logic actually reads. What it deliberately does not attempt:
anything that needs a real model or real network. A paper analyzer taking
sixteen minutes, aider declining an edit because the change was already there,
a model emitting a blank enum — those still need a real campaign. Real
campaigns become acceptance runs; this is the debugger.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.models import DecisionRecord
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace

# -- what a tool does on one call ---------------------------------------------


@dataclass
class Outcome:
    """One scripted tool call.

    `data` and `writes` are separate on purpose. What a tool *reports* and what
    it *changed* are different claims, and the whole bug class lives in the gap
    between them — so the harness has to be able to state them independently,
    including the combination that is a lie.
    """

    data: dict[str, Any] = field(default_factory=dict)
    writes: dict[str, str] = field(default_factory=dict)
    raises: BaseException | None = None
    refs: bool = True


def ok(**data: Any) -> Outcome:
    """Succeeds and reports `data`. Writes nothing."""
    return Outcome(data=data)


def writes(path: str, content: str = "changed", **data: Any) -> Outcome:
    """Succeeds and actually changes a file — the honest success."""
    return Outcome(data=data, writes={path: content})


def fails(exc: BaseException | str) -> Outcome:
    """Raises, the way `ExperimentProducedNoMetricsError` does."""
    return Outcome(raises=RuntimeError(exc) if isinstance(exc, str) else exc)


def silent_success(**data: Any) -> Outcome:
    """Reports success, returns no artifact refs, changes nothing.

    The exact shape of the rogii 2026-08-09 failure: `implement` recorded
    `completed` five times while `pipeline/train.py` went untouched, so
    `consecutive_failures` stayed at 0 for a campaign that produced nothing.
    """
    return Outcome(data=data, refs=False)


class _ScriptedTool:
    """Walks a sequence of outcomes; the last one repeats forever.

    Repeating rather than exhausting is deliberate: a campaign chooses its own
    tools, so a test cannot know how many times one will be called, and
    "ran out of script" would be a harness failure disguised as a campaign
    failure.
    """

    def __init__(self, name: str, outcomes: Sequence[Outcome]) -> None:
        self.name = name
        self._outcomes = list(outcomes) or [ok()]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, workspace: Workspace, **kwargs: Any) -> ToolResult:
        index = min(len(self.calls), len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        self.calls.append(dict(kwargs))
        if outcome.raises is not None:
            raise outcome.raises
        refs = []
        for rel, content in outcome.writes.items():
            path = workspace.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            refs.append(
                ArtifactRef(
                    kind="code",
                    id=f"{self.name}:{rel}",
                    schema_id="labpilot.artifact.code/v1",
                    path=str(path),
                    competition=workspace.competition,
                )
            )
        if outcome.refs and not refs:
            path = workspace.artifacts_dir / f"{self.name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.name, encoding="utf-8")
            refs.append(
                ArtifactRef(
                    kind="echo",
                    id=f"{self.name}:{len(self.calls)}",
                    schema_id="labpilot.artifact.echo/v1",
                    path=str(path),
                    competition=workspace.competition,
                )
            )
        return ToolResult(refs=refs, data=dict(outcome.data))


# -- the policy ---------------------------------------------------------------


class ScriptedPolicy:
    """An LLM client that answers with tool choices you queued.

    Exposes `complete(system, user)` because that is the first branch
    `_invoke_llm_next_action` tries. Running through it rather than
    `prefer_offline=True` is the point: it exercises JSON parsing,
    `validate_next_action`, and the gated-tool retry — the path where the
    policy once chose `generate_plan` six times in a single run.
    """

    def __init__(self, choices: Iterable[str | None]) -> None:
        self._choices = list(choices)
        self.prompts: list[str] = []
        self.last_served = "scripted"

    def complete(self, system: str, user: str) -> str:
        import json

        self.prompts.append(user)
        index = len(self.prompts) - 1
        if index >= len(self._choices):
            return json.dumps({"tool": None, "stop": True, "rationale": "script exhausted"})
        tool = self._choices[index]
        if tool is None:
            return json.dumps({"tool": None, "stop": True, "rationale": "scripted stop"})
        return json.dumps({"tool": tool, "args": {}, "rationale": f"scripted {tool}"})

    def offered(self, index: int = -1) -> list[str]:
        """The allowlist the policy was shown on a given step."""
        import json

        return list(json.loads(self.prompts[index]).get("allowlist") or [])


# -- the trace ----------------------------------------------------------------


@dataclass
class Trace:
    decisions: list[DecisionRecord]
    store: ConductorStore
    session_id: str
    tools: dict[str, _ScriptedTool]

    @property
    def chosen(self) -> list[str]:
        return [d.tool_name for d in self.decisions if d.tool_name]

    @property
    def stop_reason(self) -> str:
        stops = [d.rationale or "" for d in self.decisions if d.stop]
        return stops[-1] if stops else ""

    def calls(self, tool: str) -> int:
        return len(self.tools[tool].calls) if tool in self.tools else 0

    def task_statuses(self, tool: str) -> list[str]:
        return [t.status for t in self.store.list_tasks(self.session_id) if t.tool_name == tool]


# -- the harness --------------------------------------------------------------


class CampaignHarness:
    """A workspace, a store, scripted tools, and a scripted policy."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        tools: dict[str, Sequence[Outcome]] | None = None,
        slug: str = "harness",
    ) -> None:
        client = scaffold_workspace(tmp_path / slug, slug)
        self.workspace = Workspace.from_client(client).ensure_roots()
        self.store = ConductorStore(self.workspace.knowledge_dir, self.workspace.competition)
        self.tools: dict[str, _ScriptedTool] = {}
        self.registry = ToolRegistry()
        for name, outcomes in (tools or {}).items():
            self.register(name, outcomes)

    def register(self, name: str, outcomes: Sequence[Outcome] | Outcome) -> None:
        seq = [outcomes] if isinstance(outcomes, Outcome) else list(outcomes)
        tool = _ScriptedTool(name, seq)
        self.tools[name] = tool
        self.registry.register(ToolDescriptor(name=name, handler=tool, capability_status="fixed"))

    # -- seeding ------------------------------------------------------------

    def seed_plan(
        self,
        plan_id: str = "P-001",
        *,
        status: PlanStatus | str = PlanStatus.READY,
        hypothesis_id: str = "",
        goal: str = "seeded",
    ) -> str:
        """Write a plan the way the planner does, not as a fixture file.

        The gating defects — `selectable_plan_ids` skipping plans whose
        hypothesis was rejected, `has_runnable_plan`, `implement` following
        `run_plan` — are all SQL against this table. A fake would test the fake.
        """
        now = datetime.now(UTC)
        store = PlanStore(self.workspace.knowledge_dir, self.workspace.competition)
        try:
            store.upsert_plan(
                ResearchPlan(
                    id=plan_id,
                    competition=self.workspace.competition,
                    hypothesis_id=hypothesis_id,
                    goal=goal,
                    status=PlanStatus(str(status)),
                    created_at=now,
                    updated_at=now,
                )
            )
        finally:
            store.close()
        return plan_id

    def seed_hypothesis(
        self,
        *,
        status: HypothesisStatus | str = HypothesisStatus.PROPOSED,
        confidence: float = 0.6,
        technique: str | None = None,
        observation: str = "seeded observation",
    ) -> str:
        store = HypothesisStore(self.workspace.knowledge_dir, self.workspace.competition)
        hypothesis = store.create(
            observation=observation,
            reason="seeded",
            prediction="seeded",
            confidence=confidence,
            technique=technique,
        )
        if HypothesisStatus(str(status)) is not HypothesisStatus.PROPOSED:
            store.update_status(hypothesis.id, HypothesisStatus(str(status)))
        return hypothesis.id

    def seed_artifact(self, artifact_id: str = "kaggle-kernel:demo", **fields: Any) -> str:
        """Write a research artifact the way a fetch does.

        Needed because a scripted tool returning an `ArtifactRef` does *not*
        populate `research_artifacts` — the evidence-age logic reads that table
        directly, so a fake `analyze_competition` leaves it empty and the
        re-sweep floor never engages. Discovering that here rather than in a
        campaign is the harness earning its keep.
        """
        from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
        from labpilot.research_engine.intelligence.models import (
            ResearchArtifact,
            ResearchArtifactType,
        )

        artifact = ResearchArtifact(
            id=artifact_id,
            # What `KaggleFetchService` records a kernel as — a notebook is a
            # repository of code, and the harness should seed what production
            # writes rather than a shape only tests produce.
            type=fields.pop("type", ResearchArtifactType.REPOSITORY),
            source=fields.pop("source", "kaggle"),
            title=fields.pop("title", "seeded artifact"),
            **fields,
        )
        with KnowledgeStore(self.workspace.knowledge_dir, self.workspace.competition) as store:
            store.upsert_artifact(artifact)
        return artifact_id

    def age_artifacts(self, hours: float) -> None:
        """Backdate every research artifact.

        Moving the data rather than patching the clock: `hours_since_last_
        artifact` reads `MAX(created_at)` straight from SQL, and a test that
        patched `datetime.now` would pass against a version of the query that
        never ran. The re-sweep floor and the 24h staleness clause are both
        reachable this way, and both cost real wall-clock time to hit for real —
        we idled twenty minutes on the floor on 2026-08-09.
        """
        from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

        stamp = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        with KnowledgeStore(self.workspace.knowledge_dir, self.workspace.competition) as store:
            store._conn.execute(  # noqa: SLF001 — test-only backdating
                "UPDATE research_artifacts SET created_at = ?", (stamp,)
            )
            store._conn.commit()  # noqa: SLF001

    # -- running ------------------------------------------------------------

    def seed_profile(self, profile: Any) -> Path:
        """Put a `profile.json` where the campaign will look for one.

        Written through `write_profile` rather than by hand, so the stamp and
        the derived markdown view are the ones production writes — a hand-rolled
        profile is a different file that happens to have the same name.
        """
        from labpilot.accessor.profiler.report import write_profile

        json_path, _md = write_profile(self.workspace.root, profile)
        return json_path

    def run(
        self,
        *,
        policy: ScriptedPolicy | Iterable[str | None] | None = None,
        max_steps: int = 8,
        autonomy: int = 1,
        goal: str = "harness goal",
        budgets: dict[str, Any] | None = None,
        session_metadata: dict[str, Any] | None = None,
        schema_prompt: Any | None = None,
    ) -> Trace:
        if policy is not None and not isinstance(policy, ScriptedPolicy):
            policy = ScriptedPolicy(policy)
        metadata: dict[str, Any] = {"autonomy": autonomy, **(session_metadata or {})}
        if budgets:
            metadata["budgets"] = budgets
        session = self.store.create_session(goal, metadata=metadata)
        decisions = run_until_stop(
            self.store,
            self.workspace,
            session.id,
            self.registry,
            llm_client=policy,
            max_steps=max_steps,
            auto_approve=True,
            prefer_offline=policy is None,
            autonomy=autonomy,
            # None by default, which is what an unattended run has: no channel
            # to ask on, so an open schema question stops the campaign.
            schema_prompt=schema_prompt,
        )
        return Trace(
            decisions=decisions,
            store=self.store,
            session_id=session.id,
            tools=self.tools,
        )

    def available_tools(self) -> set[str]:
        """What the policy would be offered right now, against seeded state."""
        from labpilot.research_engine.conductor.policy import available_tools

        return available_tools(self.workspace, set(self.registry.names()))

    def close(self) -> None:
        self.store.close()


def harness(
    tmp_path: Path,
    tools: dict[str, Sequence[Outcome]] | None = None,
    **kwargs: Any,
) -> CampaignHarness:
    return CampaignHarness(tmp_path, tools=tools, **kwargs)


__all__ = [
    "CampaignHarness",
    "Outcome",
    "ScriptedPolicy",
    "Trace",
    "fails",
    "harness",
    "ok",
    "silent_success",
    "writes",
]
