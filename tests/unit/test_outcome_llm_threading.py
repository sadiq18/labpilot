"""Learning from a successful experiment must not die for want of a client.

Measured 2026-08-07, rogii campaign run 5. An experiment *succeeded*, and the
learn-from-it step raised:

    record_successful_execution -> update_hypothesis_from_local
      -> maybe_mint_combo_from_success -> ComboPortfolioAgent
    LLMUnavailableError: requires an LLM and none is configured

`maybe_mint_combo_from_success` already accepted `llm_client`; the caller passed
three arguments where four were needed, and `update_hypothesis_from_local` had
no parameter to carry it. The client was threaded through four layers and
dropped at the fifth.

Non-fatal and silent before M14 phase 2a: the agent would have degraded to its
rule engine, so this looked like "the system doesn't mint combos" rather than a
wiring bug. Combos and stacked hypotheses feed M8's objective loop, so the loss
is exactly the learning the campaign exists to produce.
"""

from __future__ import annotations

import inspect

from labpilot.research_engine.execution import outcome


def test_the_client_has_a_path_through_every_layer():
    """Each hop must declare the parameter, or the chain breaks silently."""
    for name in (
        "record_successful_execution",
        "update_hypothesis_from_local",
        "maybe_mint_combo_from_success",
    ):
        params = inspect.signature(getattr(outcome, name)).parameters
        assert "llm_client" in params, f"{name} cannot carry an LLM client"


def test_the_client_actually_reaches_combo_minting(monkeypatch):
    """The bug was a caller omission, so assert the value arrives — a signature
    check alone would have passed throughout the broken period."""
    seen: dict[str, object] = {}

    def _spy(*, knowledge_dir, competition, summary, llm_client=None):  # noqa: ANN001
        seen["llm_client"] = llm_client
        return []

    monkeypatch.setattr(outcome, "maybe_mint_combo_from_success", _spy)
    for other in ("maybe_mint_ablation_from_combo_win", "maybe_mint_stacked_from_success"):
        monkeypatch.setattr(outcome, other, lambda **kw: [])
    monkeypatch.setattr(outcome, "maybe_mint_improvement_hypothesis", lambda **kw: None)
    monkeypatch.setattr(outcome, "record_combo_avoid_on_loss", lambda **kw: None)
    monkeypatch.setattr(outcome, "promote_outcome_claims", lambda **kw: None)
    monkeypatch.setattr(outcome, "notify_proposed_hypotheses", lambda **kw: None)

    sentinel = object()
    summary = outcome.ExecutionOutcomeSummary(
        competition="demo", execution_id="E-1", plan_id="P-1", learning_gain=0.5
    )

    class _Store:
        def list(self):
            return []

        def get(self, *_a, **_k):
            return None

    monkeypatch.setattr(outcome, "HypothesisStore", lambda *a, **k: _Store())

    outcome.update_hypothesis_from_local(
        knowledge_dir=__import__("pathlib").Path("."),
        competition="demo",
        summary=summary,
        llm_client=sentinel,
    )

    assert seen.get("llm_client") is sentinel, (
        "combo minting was called without the client — the measured bug"
    )
