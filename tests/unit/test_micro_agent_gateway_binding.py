"""Binding an agent to its LLM role, and what counts as a gateway.

`BaseMicroAgent.__init__` accepts a gateway wherever a client is accepted, so
existing `Agent(llm_client=...)` call sites keep working and plain test stubs
stay valid. The detection has to tolerate a stub that happens to carry a
`for_role` attribute without meaning to be a gateway.
"""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent


class _Agent(BaseMicroAgent):
    name = "probe"
    llm_role = "codegen"


def test_a_non_callable_for_role_attribute_does_not_break_construction():
    """`hasattr` would call a plain attribute and raise TypeError during
    `__init__` — before the agent has done anything — for something that merely
    means "this is not a gateway"."""

    class _NotAGateway:
        for_role = "reasoning"  # an attribute, not a method

    stub = _NotAGateway()
    assert _Agent(llm_client=stub).llm_client is stub


def test_a_real_gateway_is_still_bound_to_the_role():
    """The carve-out must not cost the behaviour it guards."""
    bound = object()

    class _Gateway:
        def __init__(self) -> None:
            self.asked: str | None = None

        def for_role(self, role: str) -> object:
            self.asked = role
            return bound

    gateway = _Gateway()
    agent = _Agent(llm_client=gateway)
    assert gateway.asked == "codegen"
    assert agent.llm_client is bound


def test_a_plain_client_is_left_alone():
    class _Client:
        pass

    client = _Client()
    assert _Agent(llm_client=client).llm_client is client


def test_no_client_stays_none():
    assert _Agent().llm_client is None
