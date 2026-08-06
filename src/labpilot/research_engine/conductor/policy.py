"""Constrained Conductor policy — NextAction from allowlisted tools only."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from labpilot.research_engine.conductor.approvals import (
    OfflineFallbackPrompt,
    resolve_offline_fallback,
)
from labpilot.research_engine.conductor.models import NextAction
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

# Offline / null-LLM fallback order (deterministic).
_DEFAULT_ORDER = (
    "analyze_competition",
    "search_papers",
    "query_memory",
    "generate_plan",
    "run_plan",
    "reflect",
    "submit",
)


def build_observe_bundle(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    *,
    include_context: bool = True,
    max_context_items: int = 16,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    """Gather durable state for policy input.

    When ``include_context`` is true (online path), attach a best-effort
    Context Engine summary and ranked refs. Failures never raise — observe
    always remains usable for offline / LLM policy.
    """
    session = store.get_session(session_id)
    tasks = store.list_tasks(session_id)
    feedback = store.list_feedback(session_id, limit=10)
    decisions = store.list_decisions(session_id)
    observe: dict[str, Any] = {
        "competition": workspace.competition,
        "goal": session.goal if session else "",
        "session_status": session.status if session else None,
        "layout": workspace.layout,
        "task_summary": [
            {
                "id": t.id,
                "tool": t.tool_name,
                "status": t.status,
                "error": t.error,
            }
            for t in tasks
        ],
        "completed_tools": [t.tool_name for t in tasks if t.status == "completed"],
        "operator_feedback": [
            {
                "gated_tool": f.gated_tool,
                "decision": f.decision,
                "comment": f.comment,
            }
            for f in feedback
        ],
        "recent_rationales": [
            {"tool": d.tool_name, "rationale": d.rationale, "stop": d.stop}
            for d in decisions[-5:]
        ],
    }
    # Backlog is the campaign's core scheduling signal: test what is queued,
    # gather more evidence only when the queue runs dry.
    observe["untested_hypotheses"] = untested_hypothesis_count(workspace)
    observe["hours_since_last_artifact"] = hours_since_last_artifact(workspace)
    _attach_evidence_refresh(observe, workspace)
    if include_context:
        _attach_context(
            observe,
            workspace,
            session_id,
            max_items=max_context_items,
            max_chars=max_context_chars,
        )
    return observe


def _attach_evidence_refresh(observe: dict[str, Any], workspace: Workspace) -> None:
    """Surface bus-written evidence refresh notes for policy (best-effort)."""
    note = workspace.root / "artifacts" / f"evidence_refresh_{workspace.competition}.json"
    if not note.is_file():
        return
    try:
        data = json.loads(note.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict):
        observe["evidence_refresh"] = data


def _attach_context(
    observe: dict[str, Any],
    workspace: Workspace,
    session_id: str,
    *,
    max_items: int = 16,
    max_chars: int = 4000,
) -> None:
    """Best-effort Context Engine attach; never raises. Mutates ``observe``."""
    try:
        from labpilot.research_engine.context import ContextRequest, build_context

        goal = str(observe.get("goal") or "")
        request = ContextRequest(
            competition=workspace.competition,
            goal=goal,
            query=goal,
            session_id=session_id,
            knowledge_dir=workspace.knowledge_dir,
            max_items=max_items,
            max_chars=max_chars,
        )
        bundle = build_context(request)
        observe["context_summary"] = bundle.summary(max_chars=2000)
        observe["context_refs"] = [
            {
                "id": item.id,
                "source": item.source,
                "kind": item.kind,
                "score": item.score,
                "reason": item.reason,
            }
            for item in bundle.items
        ]
        if bundle.provider_errors:
            observe["context_provider_errors"] = list(bundle.provider_errors)
    except Exception as exc:  # noqa: BLE001 — observe must stay usable
        logger.warning("Context Engine unavailable for observe: %s", exc)
        observe.setdefault("context_summary", "")
        observe.setdefault("context_refs", [])
        observe["context_provider_errors"] = [f"build_context: {exc}"]


def offline_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
) -> NextAction:
    """Deterministic next tool: first catalog tool not yet completed."""
    done = set(observe.get("completed_tools") or [])
    # Honours reject-submit feedback: skip submit if last feedback rejected it.
    feedback = observe.get("operator_feedback") or []
    rejected = {
        f["gated_tool"]
        for f in feedback
        if f.get("decision") == "reject"
    }
    for name in _DEFAULT_ORDER:
        if name not in allowlist:
            continue
        if name in done:
            continue
        if name in rejected:
            continue
        args: dict[str, Any] = {}
        if name == "generate_plan":
            args = {"baseline": True}
        if name == "search_papers":
            args = {"offline": True}
        if name == "run_plan":
            # Caller/loop may inject plan_id; offline stub uses placeholder.
            args = {"plan_id": "P-001", "dry_run": True}
        return NextAction(
            tool=name,
            args=args,
            rationale=f"offline policy: next unfinished tool is {name}",
            stop=False,
        )
    return NextAction(tool=None, rationale="offline policy: catalog exhausted", stop=True)


def validate_next_action(action: NextAction, allowlist: set[str]) -> NextAction:
    """Reject invented tools; force stop if invalid."""
    if action.stop or not action.tool:
        return NextAction(
            tool=None,
            args={},
            rationale=action.rationale or "stop",
            stop=True,
        )
    if action.tool not in allowlist:
        return NextAction(
            tool=None,
            args={},
            rationale=f"rejected non-catalog tool: {action.tool}",
            stop=True,
        )
    return action


class _GatedToolError(RuntimeError):
    """The policy picked a real tool that is gated right now.

    Distinct from a transport failure so the retry can name it back to the
    model instead of repeating an identical prompt.
    """

    def __init__(self, tool: str, available: list[str]) -> None:
        self.tool = tool
        super().__init__(f"policy chose unavailable tool {tool!r}; available now: {available}")


def _invoke_llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any,
    rejected: list[str] | None = None,
) -> NextAction:
    """Ask for one tool choice, telling the model what it already got wrong.

    ``rejected`` carries tools this step already tried and that are gated right
    now. Without it a retry re-sends an identical prompt, so a model that
    wanted a gated tool asks for it again — measured 2026-08-07, where
    `generate_plan` was chosen six times in one run and every retry was
    indistinguishable from the first.
    """
    catalog = sorted(allowlist)
    system = (
        "You are the LabPilot Research Conductor. Choose the single next tool "
        "from the allowlist, or stop. Never invent tools. Prefer operator_feedback "
        "comments when deciding. Use context_summary and context_refs as ranked "
        "evidence (higher score is stronger). Respond with JSON only: "
        '{"tool": "<name>|null", "args": {}, "rationale": "...", "stop": false}'
    )
    payload: dict[str, Any] = {"allowlist": catalog, "observe": observe}
    if rejected:
        payload["already_rejected"] = {
            "tools": sorted(set(rejected)),
            "why": (
                "Not available right now — a precondition is unmet (for example "
                "generate_plan is gated while a plan is still unrun). Choose a "
                "different tool from allowlist, or stop."
            ),
        }
    user = json.dumps(payload, indent=2, default=str)
    if hasattr(llm_client, "complete"):
        text = llm_client.complete(system, user)
    elif hasattr(llm_client, "generate"):
        text = str(llm_client.generate(task="planning", prompt=user))
    else:
        raise TypeError("llm_client has no complete/generate method")
    data = _parse_json(text)
    action = NextAction.model_validate(data)
    validated = validate_next_action(action, allowlist)
    # Choosing a tool that is currently gated is a *recoverable* policy mistake,
    # not a reason to end the campaign. Measured 2026-08-07: `generate_plan` is
    # removed from the allowlist while a plan is unrun; the policy chose it
    # anyway and the run stopped at step 4 with "rejected non-catalog tool".
    #
    # Raising routes it into the retry/offline-fallback loop below, which either
    # gets a better answer or falls back to the deterministic order — both of
    # which make progress. A genuine `stop` from the model is still honoured.
    if validated.stop and action.tool and not action.stop:
        raise _GatedToolError(action.tool, sorted(allowlist))
    return validated


def llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any | None,
    *,
    prefer_offline: bool = False,
    auto_offline_fallback: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
    max_llm_retries: int = 5,
) -> NextAction:
    """Ask the LLM for a structured NextAction.

    On LLM failure (or missing client in online mode), ask the operator before
    using the deterministic offline order: allow, deny, or retry.
    Intentional ``prefer_offline`` skips the prompt.
    """
    if prefer_offline:
        return offline_next_action(observe, allowlist)

    retries = 0
    # Tools this step already asked for that are currently gated. Fed back into
    # the next attempt so a retry is a genuinely different question.
    rejected: list[str] = []
    gated_retries = 0
    while True:
        if llm_client is None:
            reason = "No LLM client available"
        else:
            try:
                return _invoke_llm_next_action(observe, allowlist, llm_client, rejected)
            except _GatedToolError as exc:
                # Retry *directly* rather than falling through to the offline
                # prompt below: under `--yes` that prompt auto-allows, so the
                # run would drop to the deterministic order without the model
                # ever being told what was wrong. Bounded by the same budget.
                rejected.append(exc.tool)
                gated_retries += 1
                logger.info(
                    "Policy asked for gated tool %r; retrying with it ruled out", exc.tool
                )
                if gated_retries <= max_llm_retries:
                    continue
                reason = f"LLM policy kept choosing gated tools: {sorted(set(rejected))}"
                logger.warning("%s", reason)
            except Exception as exc:
                reason = f"LLM policy failed: {exc}"
                logger.warning("Conductor policy LLM failed (%s)", exc)

        decision = resolve_offline_fallback(
            reason,
            auto=auto_offline_fallback,
            prompt=offline_fallback_prompt,
        )
        if decision == "allow":
            logger.info("Operator allowed offline policy fallback (%s)", reason)
            return offline_next_action(observe, allowlist)
        if decision == "deny":
            logger.info("Operator denied offline policy fallback (%s)", reason)
            return NextAction(
                tool=None,
                rationale=f"operator denied offline policy fallback ({reason})",
                stop=True,
            )
        # retry
        retries += 1
        if retries > max_llm_retries:
            logger.warning(
                "Exceeded max LLM retries (%s); treating as deny", max_llm_retries
            )
            return NextAction(
                tool=None,
                rationale=(
                    f"operator retry exhausted after {max_llm_retries} attempts "
                    f"({reason})"
                ),
                stop=True,
            )
        logger.info("Operator requested LLM policy retry (%d/%d)", retries, max_llm_retries)


# A campaign only needs a handful of untested ideas in front of it. Below this
# it is worth spending minutes gathering more evidence; at or above it, that
# time is better spent testing what is already queued.
_HYPOTHESIS_BACKLOG_TARGET = int(os.environ.get("LABPILOT_HYPOTHESIS_BACKLOG_TARGET", "3"))
# Re-sweeping the same kernels and papers minutes apart mostly re-ingests the
# same sources under new artifact ids, bloating the store without adding
# information. Evidence has to be allowed to go stale before refetching.
_EVIDENCE_COOLDOWN_HOURS = float(os.environ.get("LABPILOT_EVIDENCE_COOLDOWN_HOURS", "6.0"))


def available_tools(workspace: Workspace, allowlist: set[str]) -> set[str]:
    """Drop tools whose preconditions the workspace does not yet satisfy.

    Offering the whole catalog regardless of state lets a campaign burn steps
    on impossible work — reflecting before anything has run, or submitting
    before a model exists. Filtering first turns "the model picked badly" into
    "that option was never on the table".
    """
    from labpilot.research_engine.conductor.loop import (
        _latest_execution_id,
        _latest_plan_id,
    )

    has_plan = _latest_plan_id(workspace) is not None
    has_execution = _latest_execution_id(workspace) is not None

    # Evidence gathering is expensive (kernels, discussions, papers, repos —
    # minutes of network and LLM work). Once there is a backlog of untested
    # hypotheses, the useful move is to *test* one, not to re-derive the same
    # techniques and beliefs again. Gathering reopens when the backlog runs dry.
    gather_ok, gather_reason = should_gather_evidence(workspace)
    if not gather_ok:
        logger.info("Skipping evidence gathering: %s", gather_reason)

    requires: dict[str, bool] = {
        # Nothing to reflect on until an experiment has produced evidence.
        "reflect": has_execution,
        # Cannot run, or submit the result of, a plan that does not exist.
        "run_plan": has_plan,
        "run_experiment": has_plan,
        "submit": has_execution,
        "submit_learn": has_execution,
        # Re-analysing with work already queued is the single most expensive
        # way for a campaign to make no progress.
        "analyze_competition": gather_ok,
        "search_papers": gather_ok,
        # Same brake one level down: queuing another plan while one is still
        # unrun adds no information and starves the thing that does.
        "generate_plan": not has_unrun_plan(workspace),
    }
    return {name for name in allowlist if requires.get(name, True)}


def hours_since_last_artifact(workspace: Workspace) -> float | None:
    """Age of the newest research artifact in hours, or None when there are none."""
    from datetime import UTC, datetime

    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

    try:
        with KnowledgeStore(workspace.knowledge_dir, workspace.competition) as store:
            row = store._conn.execute(  # noqa: SLF001 — read-only freshness probe
                "SELECT MAX(created_at) AS newest FROM research_artifacts"
            ).fetchone()
    except Exception:  # noqa: BLE001 — no store means no artifacts
        return None
    newest = row["newest"] if row else None
    if not newest:
        return None
    try:
        stamp = datetime.fromisoformat(str(newest))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - stamp).total_seconds() / 3600.0


def should_gather_evidence(workspace: Workspace) -> tuple[bool, str]:
    """Decide whether pulling more artifacts is worth it right now.

    Two independent brakes, because either one alone lets the store bloat:
    a queue of untested ideas means the bottleneck is *testing*, not evidence;
    and a recent sweep means another one would mostly re-ingest the same
    kernels and papers under new artifact ids.
    """
    backlog = untested_hypothesis_count(workspace)
    if backlog >= _HYPOTHESIS_BACKLOG_TARGET:
        return False, f"{backlog} untested hypotheses already queued"

    age_hours = hours_since_last_artifact(workspace)
    if age_hours is not None and age_hours < _EVIDENCE_COOLDOWN_HOURS:
        return False, f"evidence gathered {age_hours:.1f}h ago (cooldown)"

    if age_hours is None:
        return True, "no evidence gathered yet"
    return True, f"backlog {backlog} is thin and evidence is {age_hours:.1f}h old"


def has_unrun_plan(workspace: Workspace) -> bool:
    """True when a compiled plan is still waiting to be executed."""
    from labpilot.research_engine.artifacts.plan import PlanArtifacts

    artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        plans = artifacts.list()
    except Exception:  # noqa: BLE001
        return False
    finally:
        artifacts.close()
    return any(str(p.status) in {"ready", "draft"} for p in plans)


def untested_hypothesis_count(workspace: Workspace) -> int:
    """How many proposed hypotheses are waiting to be tested."""
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    try:
        store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
        return len(store.list(status=HypothesisStatus.PROPOSED))
    except Exception:  # noqa: BLE001 — absent store means nothing queued
        return 0


def decide_next(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
    prefer_offline: bool = False,
    auto_offline_fallback: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
) -> tuple[NextAction, dict[str, Any]]:
    """Observe + think; return validated NextAction and observe bundle.

    Online path attaches Context Engine evidence to observe. ``prefer_offline``
    skips retrieve entirely (no forced Context Engine success).
    """
    allowlist = available_tools(workspace, set(registry.names()))
    observe = build_observe_bundle(
        store,
        workspace,
        session_id,
        include_context=not prefer_offline,
    )
    action = llm_next_action(
        observe,
        allowlist,
        llm_client,
        prefer_offline=prefer_offline,
        auto_offline_fallback=auto_offline_fallback,
        offline_fallback_prompt=offline_fallback_prompt,
    )
    return action, observe


def _parse_json(text: str) -> dict[str, Any]:
    """Parse a policy decision.

    Delegates to the shared extractor: this used to be a second, naive copy
    (first ``{`` to last ``}``), so hardening one parser left the Conductor's
    own decisions just as brittle as before.
    """
    from labpilot.llm.json_utils import parse_json_object

    return parse_json_object(text)
