"""Campaign budgets and automatic stop evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

StopReason = Literal[
    "none",
    "submission_budget",
    "wall_time",
    "cost_budget",
    "metric_target",
    "plateau",
    "operator_pause",
    "policy_stop",
    "max_steps",
]


class BudgetConfig(BaseModel):
    """Resource and objective limits for a campaign session."""

    max_submissions: int | None = None
    max_wall_s: float | None = None
    max_cost_usd: float | None = None
    target_metric: str | None = None
    target_value: float | None = None
    maximize: bool = True
    plateau_window: int = 3
    plateau_epsilon: float = 1e-6


class BudgetState(BaseModel):
    """Live counters persisted in session metadata / metrics table."""

    submissions: int = 0
    llm_cost_usd: float = 0.0
    wall_started_at: str | None = None
    metric_history: list[float] = Field(default_factory=list)
    last_metric: float | None = None

    def ensure_wall_start(self) -> None:
        if not self.wall_started_at:
            self.wall_started_at = datetime.now(UTC).isoformat()

    def elapsed_s(self, *, now: datetime | None = None) -> float:
        if not self.wall_started_at:
            return 0.0
        start = datetime.fromisoformat(self.wall_started_at)
        current = now or datetime.now(UTC)
        return max(0.0, (current - start).total_seconds())


def evaluate_stops(
    config: BudgetConfig,
    state: BudgetState,
    *,
    now: datetime | None = None,
) -> StopReason:
    """Return the first matching automatic stop reason, or ``none``.

    ``max_submissions=0`` means **do not submit**, not **do not run**. The cap
    is a limit on an action the campaign may take; reaching it is only a reason
    to stop once the campaign has actually taken it. Evaluated before the first
    step, ``0 >= 0`` ended a campaign having done nothing — one `stop` decision
    and no research — so the natural way to ask for "run, but never upload"
    silently asked for nothing at all.

    `submit_tools_allowed` is what enforces the zero; this only stops on a cap
    the campaign has spent.
    """
    if (
        config.max_submissions is not None
        and config.max_submissions > 0
        and state.submissions >= config.max_submissions
    ):
        return "submission_budget"
    if config.max_wall_s is not None and state.elapsed_s(now=now) >= config.max_wall_s:
        return "wall_time"
    if config.max_cost_usd is not None and state.llm_cost_usd >= config.max_cost_usd:
        return "cost_budget"
    if config.target_metric and config.target_value is not None and state.last_metric is not None:
        if config.maximize and state.last_metric >= config.target_value:
            return "metric_target"
        if not config.maximize and state.last_metric <= config.target_value:
            return "metric_target"
    hist = state.metric_history
    n = max(1, config.plateau_window)
    if len(hist) >= n:
        window = hist[-n:]
        gain = max(window) - min(window)
        if gain <= config.plateau_epsilon:
            return "plateau"
    return "none"


def submit_tools_allowed(config: BudgetConfig) -> bool:
    """False when the campaign is configured never to submit.

    Enforced by removing the submit tools from the allowlist rather than by
    approving and then refusing them. `--yes` maps every gated tool to
    `auto_approve`, so approval is not a brake in a non-interactive run: a
    campaign told not to submit could still upload to Kaggle because nobody was
    at the terminal to say no.
    """
    return not (config.max_submissions is not None and config.max_submissions <= 0)


def budgets_from_metadata(meta: dict[str, Any]) -> tuple[BudgetConfig, BudgetState]:
    cfg = BudgetConfig.model_validate(meta.get("budgets") or {})
    state = BudgetState.model_validate(meta.get("budget_state") or {})
    return cfg, state


def budgets_to_metadata(
    meta: dict[str, Any],
    config: BudgetConfig,
    state: BudgetState,
) -> dict[str, Any]:
    out = dict(meta)
    out["budgets"] = config.model_dump()
    out["budget_state"] = state.model_dump()
    return out
