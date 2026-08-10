"""Constrained Conductor policy — NextAction from allowlisted tools only."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from labpilot.accessor.common.provenance import record_invocation
from labpilot.research_engine.conductor.actions import _default_args
from labpilot.research_engine.conductor.approvals import (
    OfflineFallbackPrompt,
    resolve_offline_fallback,
)
from labpilot.research_engine.conductor.models import NextAction
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.intelligence.hypothesis.viability import (
    pool_counts,
    viable_hypothesis_count,
)
from labpilot.research_engine.planner.schemas.task_types import (
    is_unrun_plan_status,
)
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

# Offline / null-LLM fallback order (deterministic).
#: Tools worth doing again once the first pass is complete — each produces new
#: information on a repeat. `analyze_competition` and `search_papers` do not.
_REPEATABLE = ("generate_plan", "run_plan", "reflect")

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
            {"tool": d.tool_name, "rationale": d.rationale, "stop": d.stop} for d in decisions[-5:]
        ],
    }
    # Backlog is the campaign's core scheduling signal. The policy is shown the
    # *viable* count — the same number `should_gather_evidence` decides on — so
    # its reasoning and the allowlist cannot disagree about the same workspace.
    # On rogii that difference was 46 versus a handful, and a policy told "46
    # queued" will never conclude it needs better ideas.
    #
    # The raw count is kept beside it rather than dropped: "46 proposed, 3
    # viable" is a more useful observation than either number alone, and the
    # gap between them is itself a signal that the pool has gone stale.
    #
    # Both under names that say what they hold. `untested_hypotheses` used to
    # mean the raw proposed count and was quietly repointed at the viable one —
    # same label, different number, for every consumer including the policy
    # prompt itself. A number the model reads deserves a name it can trust, so
    # the new meaning gets a new key and the old name keeps the old meaning.
    #
    # One store read for both. Asked separately they globbed every `H-*.json`,
    # parsed each and mirrored the whole pool into SQLite — twice, back to back,
    # on every policy step.
    viable, proposed_total = pool_counts(workspace.knowledge_dir, workspace.competition)
    observe["viable_hypotheses"] = viable
    observe["untested_hypotheses"] = proposed_total
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
    # `.get` on both keys. The line above already used it for `decision`, so a
    # row lacking `gated_tool` survived the filter and then raised `KeyError`
    # from the subscript — a malformed feedback row taking the offline policy
    # down with it. `build_observe_bundle` always writes both, but this reads
    # whatever is in the store, including rows written before it did.
    rejected = {f.get("gated_tool") for f in feedback if f.get("decision") == "reject"}
    rejected.discard(None)
    for name in _DEFAULT_ORDER:
        if name not in allowlist:
            continue
        if name in done:
            continue
        if name in rejected:
            continue
        return NextAction(
            tool=name,
            args=_default_args(name),
            rationale=f"offline policy: next unfinished tool is {name}",
            stop=False,
        )

    # Everything has run once. A research loop is not a one-pass checklist:
    # after analyze -> plan -> run -> reflect, the useful move is to plan
    # against a new hypothesis, not to stop. Returning `stop=True` here ended
    # two healthy campaigns — S-019 at step 8 and S-020 at step 27 — with the
    # LLM working fine by then.
    #
    # Only the tools that produce new information on a repeat are eligible;
    # re-analysing the competition does not. The allowlist still gates them, so
    # this cannot spin: `generate_plan` is removed while a plan is unrun, and
    # `run_plan` while none is runnable.
    available = [t for t in _REPEATABLE if t in allowlist and t not in rejected]
    if not available:
        return NextAction(
            tool=None,
            rationale="offline policy: no repeatable tool is currently available",
            stop=True,
        )

    # Least-recently-used, not fixed order. Taking `_REPEATABLE` in order meant
    # `generate_plan` won whenever it was available, giving plan -> run -> plan
    # -> run and never reflecting again after the first pass.
    # `completed_tools` is an ordered list with repeats, so "how long since this
    # last ran" is answerable.
    ordered = list(observe.get("completed_tools") or [])
    positions = {tool: i for i, tool in enumerate(ordered)}  # last index wins
    available.sort(key=lambda t: positions.get(t, -1))
    choice = available[0]

    # Repeating the thing we just did, with nothing else on offer, is a spin.
    # A DRAFT plan makes `has_unrun_plan` true and `has_runnable_plan` false, so
    # the allowlist can narrow to `{reflect}` alone — bounded by max_steps, but
    # it burns a whole degraded campaign re-reflecting on the same state.
    if len(available) == 1 and ordered and ordered[-1] == choice:
        return NextAction(
            tool=None,
            rationale=(
                f"offline policy: {choice} is the only available tool and it "
                "just ran; nothing further to try"
            ),
            stop=True,
        )
    return NextAction(
        tool=choice,
        args=_default_args(choice),
        rationale=f"offline policy: cycling back to {choice}",
        stop=False,
    )


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
    """The policy picked a tool it cannot have right now.

    Distinct from a transport failure so the retry can name it back to the
    model instead of repeating an identical prompt.

    ``known`` separates two cases the retry must describe differently: a real
    tool whose precondition is unmet (``generate_plan`` while a plan is unrun),
    versus a name that does not exist at all. Telling a model that an invented
    tool "failed a precondition" invites it to wait for that precondition.
    """

    def __init__(self, tool: str, available: list[str], *, known: bool) -> None:
        self.tool = tool
        self.known = known
        kind = "gated" if known else "invented"
        super().__init__(f"policy chose {kind} tool {tool!r}; available now: {available}")


def _invoke_llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any,
    rejected: list[tuple[str, bool]] | None = None,
    *,
    all_tools: set[str] | None = None,
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
        gated = sorted({t for t, known in rejected if known})
        invented = sorted({t for t, known in rejected if not known})
        already: dict[str, Any] = {}
        if gated:
            already["gated"] = {
                "tools": gated,
                "why": (
                    "These tools exist but a precondition is unmet right now (for "
                    "example generate_plan is gated while a plan is still unrun). "
                    "They may become available later; do not wait for them here."
                ),
            }
        if invented:
            already["not_real"] = {
                "tools": invented,
                "why": "No such tool exists. Choose only from allowlist.",
            }
        already["instruction"] = "Choose a different tool from allowlist, or stop."
        payload["already_rejected"] = already
    user = json.dumps(payload, indent=2, default=str)
    # Recorded here rather than in `decide_next`'s handler so success and
    # failure are stamped at the same place the call happens. The Conductor
    # policy is the highest-frequency LLM caller in the system and is *not* a
    # micro agent, so none of this reached the provenance store — measured
    # 2026-08-07, an 8-step campaign recorded one invocation while the policy
    # fell back to the offline order three times. That missing number is
    # exactly what M14 2b needs to decide whether fatal failure is safe.
    try:
        if hasattr(llm_client, "complete"):
            text = llm_client.complete(system, user)
        elif hasattr(llm_client, "generate"):
            text = str(llm_client.generate(task="planning", prompt=user))
        else:
            raise TypeError("llm_client has no complete/generate method")
        data = _parse_json(text)
        action = NextAction.model_validate(data)
    except Exception as exc:
        record_invocation(
            agent="ConductorPolicy",
            generated_by="rule_engine",
            llm_role="default",
            failure_reason=str(exc),
        )
        raise
    record_invocation(
        agent="ConductorPolicy",
        generated_by="llm",
        llm_role="default",
        served=getattr(llm_client, "last_served", None),
    )
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
        # `catalog` is every tool that exists; `allowlist` is those available
        # now. Present in the first but not the second means gated.
        # Absent `all_tools` we cannot distinguish the two, so report the
        # weaker claim ("not available") rather than assert either.
        catalog = all_tools if all_tools is not None else allowlist
        raise _GatedToolError(action.tool, sorted(allowlist), known=action.tool in catalog)
    return validated


def llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any | None,
    *,
    all_tools: set[str] | None = None,
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
    rejected: list[tuple[str, bool]] = []
    gated_retries = 0
    while True:
        if llm_client is None:
            reason = "No LLM client available"
        else:
            try:
                return _invoke_llm_next_action(
                    observe, allowlist, llm_client, rejected, all_tools=all_tools
                )
            except _GatedToolError as exc:
                # Retry *directly* rather than falling through to the offline
                # prompt below: under `--yes` that prompt auto-allows, so the
                # run would drop to the deterministic order without the model
                # ever being told what was wrong. Bounded by the same budget.
                rejected.append((exc.tool, exc.known))
                gated_retries += 1
                logger.info("Policy asked for gated tool %r; retrying with it ruled out", exc.tool)
                if gated_retries <= max_llm_retries:
                    continue
                reason = (
                    "LLM policy kept choosing unavailable tools: "
                    f"{sorted({t for t, _ in rejected})}"
                )
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
            logger.warning("Exceeded max LLM retries (%s); treating as deny", max_llm_retries)
            return NextAction(
                tool=None,
                rationale=(f"operator retry exhausted after {max_llm_retries} attempts ({reason})"),
                stop=True,
            )
        logger.info("Operator requested LLM policy retry (%d/%d)", retries, max_llm_retries)


# A campaign only needs a handful of *viable* ideas in front of it. Below this,
# minutes spent gathering evidence are worth more than minutes spent testing.
#
# Counted by `viable_hypothesis_count`, not by row count: the previous version
# counted every `proposed` row, so 46 stale entries — most never selected, some
# off-domain for the competition — held the gate shut as firmly as 46 good ones.
_VIABLE_TARGET = int(os.environ.get("LABPILOT_VIABLE_HYPOTHESIS_TARGET", "5"))
# Re-sweeping the same kernels and papers minutes apart mostly re-ingests the
# same sources under new artifact ids. A day is long enough that a competition's
# kernels and discussions have plausibly moved, and short enough that a
# long-running campaign refreshes what it knows rather than compounding one
# morning's snapshot.
#
# Measured from the newest `research_artifacts` row — the last *fetch* — not
# from hypothesis age, so testing activity never masks stale evidence.
_EVIDENCE_COOLDOWN_HOURS = float(os.environ.get("LABPILOT_EVIDENCE_COOLDOWN_HOURS", "24.0"))
# Hard floor between sweeps, whatever else is true. Without it, a pool that
# stays thin — a sweep that found nothing, or one whose candidates were all
# deduped away — would gather on every single step.
_MIN_RESWEEP_HOURS = float(os.environ.get("LABPILOT_MIN_RESWEEP_HOURS", "0.5"))


def available_tools(workspace: Workspace, allowlist: set[str]) -> set[str]:
    """Drop tools whose preconditions the workspace does not yet satisfy.

    Offering the whole catalog regardless of state lets a campaign burn steps
    on impossible work — reflecting before anything has run, or submitting
    before a model exists. Filtering first turns "the model picked badly" into
    "that option was never on the table".
    """
    from labpilot.research_engine.conductor.loop import (
        _latest_execution_id,
    )

    # Not `_latest_plan_id(...) is not None`: that falls back to the newest
    # plan of *any* status, so a workspace whose plans are all done still
    # reported one. `run_plan` was then offered, the Engineer refused with
    # "status=done; need ready or in_progress", and the campaign lost a step.
    has_runnable = has_runnable_plan(workspace)
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
        # Cannot run a plan the Engineer would refuse.
        "run_plan": has_runnable,
        "run_experiment": has_runnable,
        "submit": has_execution,
        "submit_learn": has_execution,
        # Re-analysing with work already queued is the single most expensive
        # way for a campaign to make no progress.
        "analyze_competition": gather_ok,
        # Never offered. Every conductor path forces `offline=True` — the
        # template, `_default_args`, and the CLI's own registry wrapper all do —
        # and the policy only ever chooses tool *names*, so a campaign's
        # `search_papers` writes `count: 0` and returns. It cannot gather
        # evidence, only spend a step. Literature is reached through
        # `analyze_competition`, where it is a config choice rather than a
        # separate step; on rogii 2026-08-09 this took step 1 of 8 and produced
        # an empty hit list.
        "search_papers": False,
        # Same brake one level down: queuing another plan while one is still
        # unrun adds no information and starves the thing that does.
        "generate_plan": not has_unrun_plan(workspace),
        # `implement` writes code *for* something. With no runnable plan there
        # is no hypothesis to implement, and the tool was ungated entirely —
        # so when `run_plan`, `run_experiment` and `generate_plan` all closed,
        # the policy reached for the one door left open and spent every step
        # there. Measured on rogii 2026-08-09: 16 dispatches, 5 recorded
        # `completed`, `train.py` untouched throughout.
        #
        # Gating on the same condition as `run_plan` is deliberate: both act on
        # a plan, so both should disappear together rather than leaving one as
        # an escape hatch from the other's absence.
        "implement": has_runnable,
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
    """Gather when the pool is thin **or** the evidence is stale.

    This was two brakes in series — gather only if the backlog was thin *and*
    the last sweep was old — which made either one a veto. The backlog clause
    was checked first, so on rogii 2026-08-09 with **46 proposed hypotheses**
    the staleness clause was never evaluated at all, and
    `analyze_competition` / `search_papers` left the allowlist permanently.

    That is a ratchet: the pool blocks the only thing that could refresh it, and
    the pool grows. It also inverted the intent — a queue of stale ideas is the
    strongest reason to go and find better ones, not a reason to stop looking.

    So the conditions are now independent. Either is sufficient:

    * **thin** — fewer than `_VIABLE_TARGET` hypotheses the campaign might
      actually pick, counted by `viable_hypothesis_count` so that rows the
      selector has passed over for two campaigns stop voting;
    * **stale** — no artifact newer than `_EVIDENCE_COOLDOWN_HOURS`, which
      guarantees recovery no matter how large the pool grows.

    The cost of gathering when it was not needed is a sweep that mostly
    re-ingests known kernels. The cost of *not* gathering was four campaigns
    that could not improve, so the asymmetry is deliberate.
    """
    age_hours = hours_since_last_artifact(workspace)

    # A floor under both clauses, not a third gate. Making the conditions
    # independent introduces a failure the AND version could not have: a
    # campaign whose pool stays thin — because a sweep found nothing new, or
    # because dedupe dropped it all — would sweep again every step. That is the
    # old ratchet inverted, and just as expensive.
    #
    # Minutes, not hours: long enough that no campaign re-sweeps inside a single
    # loop, short enough that it never becomes the reason evidence goes stale.
    if age_hours is not None and age_hours < _MIN_RESWEEP_HOURS:
        return False, f"evidence gathered {age_hours * 60:.0f} minutes ago"

    viable = viable_hypothesis_count(workspace.knowledge_dir, workspace.competition)
    if viable < _VIABLE_TARGET:
        return True, f"only {viable} viable hypotheses queued"

    if age_hours is None:
        return True, "no evidence gathered yet"
    if age_hours >= _EVIDENCE_COOLDOWN_HOURS:
        return True, f"evidence is {age_hours:.1f}h old"

    return (
        False,
        f"{viable} viable hypotheses queued and evidence gathered {age_hours:.1f}h ago",
    )


def _plan_statuses(workspace: Workspace) -> list[str]:
    from labpilot.research_engine.artifacts.plan import PlanArtifacts

    artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        return [str(p.status) for p in artifacts.list()]
    except Exception:  # noqa: BLE001 — absent store means "no plans"
        return []
    finally:
        artifacts.close()


def has_unrun_plan(workspace: Workspace) -> bool:
    """True when a plan still represents outstanding work (may not be runnable)."""
    return any(is_unrun_plan_status(s) for s in _plan_statuses(workspace))


def has_runnable_plan(workspace: Workspace) -> bool:
    """True when a plan can actually be dispatched to the Engineer right now.

    Excludes plans whose hypothesis is retired — otherwise `run_plan` and
    `run_experiment` stay in the allowlist targeting work already settled, and
    the campaign keeps choosing them. That is what happened on rogii after
    `H-051` was correctly rejected.

    One indexed query. The obvious implementation — read the rejected
    hypotheses, list the plans, filter in Python — costs a file read per
    rejected hypothesis plus `list_plans()`, which hydrates every plan, its
    tasks and each task's dependency edges. That is N+2 queries and a full
    object graph to answer a boolean, on every policy step.
    """
    from labpilot.research_engine.intelligence.paths import store_is_absent
    from labpilot.research_engine.planner.store import PlanStore

    if store_is_absent(workspace.knowledge_dir, workspace.competition):
        return False
    store = None
    try:
        # Inside the guard: opening the store is where a corrupt database fails.
        store = PlanStore(workspace.knowledge_dir, workspace.competition)
        return bool(store.selectable_plan_ids())
    except Exception:
        # This answer decides whether the conductor runs anything at all, and a
        # fault used to give the same one as an empty store: *nothing runnable*.
        # A locked database stopped a campaign and looked like a finished one.
        # M20, 2026-08-09.
        logger.exception(
            "cannot read plans for %s; treating as nothing runnable", workspace.competition
        )
        return False
    finally:
        if store is not None:
            store.close()


def untested_hypothesis_count(workspace: Workspace) -> int:
    """How many proposed hypotheses are waiting to be tested."""
    from labpilot.research_engine.intelligence.paths import hypotheses_are_absent
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    if hypotheses_are_absent(workspace.knowledge_dir, workspace.competition):
        return 0
    try:
        store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
        return len(store.list(status=HypothesisStatus.PROPOSED))
    except Exception:
        # Zero is how the conductor learns there is nothing left to test, and a
        # fault said zero. See `has_runnable_plan`.
        logger.exception(
            "cannot count hypotheses for %s; treating as none queued", workspace.competition
        )
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
    all_tools = set(registry.names())
    allowlist = available_tools(workspace, all_tools)
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
        all_tools=all_tools,
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
