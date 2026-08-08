"""The claim is written before the edit, which is what makes it a check.

M19 §5 requires a claim *independent* of the code — one read off the diff cannot
test the diff. Whole-file codegen declares its own; aider returns a diff and no
structured claim, so every aider delta landed `delta_unchecked` and three of the
four checks went dark exactly when deltas arrived.

`DeltaBriefAgent` closes that by producing intent first and execution second.
This is the third mechanism proposed for the job — after `technique` metadata and
a technique→symbol map — and the first that keeps the ordering §5 depends on.
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.execution.schemas.delta_brief import DeltaBrief


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, []),
        ({"kept": None}, []),
        ({"kept": []}, []),
        ({"kept": "lgb"}, ["lgb"]),
        ({"kept": " lgb "}, ["lgb"]),
        ({"kept": ["lgb", "  ", "cb"]}, ["lgb", "cb"]),
        ({"kept": ["  lgb", "cb  "]}, ["lgb", "cb"]),
    ],
)
def test_a_claim_parses_however_the_model_spells_it(payload, expected):
    """Same leniency as `CodeProposal`, for the same reason: a ValidationError
    is read by the retry path as a malformed response and re-asked, costing a
    step over optional metadata. Names are stripped because the checks match
    them against AST symbols, which carry no surrounding whitespace."""
    assert DeltaBrief.model_validate_json(json.dumps(payload)).kept == expected


def test_an_empty_brief_claims_nothing():
    """An empty list is a correct answer — a hyperparameter change adds no
    symbol — and must not read as a claim that was made."""
    assert DeltaBrief().claims_anything() is False


@pytest.mark.parametrize(
    "brief",
    [
        DeltaBrief(kept=["lgb"]),
        DeltaBrief(added=["cb"]),
        DeltaBrief(combined=["lgb", "cb"]),
    ],
)
def test_any_populated_list_is_a_claim(brief):
    assert brief.claims_anything() is True


def test_the_instruction_survives_intact():
    """It is prose for the editor, not a list of symbols, and must not be
    normalised the way the claim lists are."""
    text = "Ensemble LightGBM with CatBoost; average their predictions."
    assert DeltaBrief(instruction=text).instruction == text


def test_the_agent_asks_the_reasoning_role_not_codegen():
    """This writes no code. Pinning it to `codegen` would make every delta pay
    codegen prices for a short structured answer."""
    from labpilot.research_engine.execution.micro_agents.delta_brief import DeltaBriefAgent

    assert DeltaBriefAgent.llm_role == "reasoning"
    assert DeltaBriefAgent.output_model is DeltaBrief


def test_the_prompt_forbids_technique_names_and_record_references():
    """The failure this mechanism replaces: `technique` held `hyp:H-010`,
    `feature_engineering`, and once the bare word `the`."""
    from labpilot.research_engine.execution.micro_agents.delta_brief.agent import (
        DeltaBriefAgent,
    )

    system = DeltaBriefAgent(llm_client=None).system_prompt()

    assert "hyp:H-010" in system
    assert "SWA" in system  # named as the canonical non-symbol
    assert "not an importable symbol" in system


def test_the_prompt_warns_against_padding_the_claim():
    """A false name makes a correct experiment look inconsistent, which
    discredits the mechanism — worse than checking less."""
    from labpilot.research_engine.execution.micro_agents.delta_brief.agent import (
        DeltaBriefAgent,
    )

    system = DeltaBriefAgent(llm_client=None).system_prompt().lower()

    assert "do not pad" in system
    assert "empty list is a correct answer" in system


def test_the_user_prompt_carries_the_parent_and_the_retry_reason():
    """Symbols must be nameable from code that exists, and a retry that repeats
    the last failure is the loop this codebase has paid for repeatedly."""
    from labpilot.accessor.common.micro_agents import StructuredContext
    from labpilot.research_engine.execution.micro_agents.delta_brief.agent import (
        DeltaBriefAgent,
    )

    prompt = DeltaBriefAgent(llm_client=None).user_prompt(
        StructuredContext(
            competition="rogii",
            data={
                "plan_goal": "ensemble two models",
                "prior_train_py": "import lightgbm as lgb\n",
                "retry_reason": "Fields with bad pandas dtypes: Geology: object",
                "technique": "SWA",
            },
        )
    )

    assert "import lightgbm as lgb" in prompt
    assert "Geology: object" in prompt
    assert "SWA" in prompt
