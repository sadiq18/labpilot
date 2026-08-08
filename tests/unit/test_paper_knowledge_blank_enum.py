"""A blank enum must not cost a paper its extracted knowledge.

Measured on rogii 2026-08-09, during the first campaign where `search_papers`
was the only door left open. `PaperAnalyzerAgent` returned every field
populated except `grounded_in`, which came back `""`. Validation failed, the
micro-agent retried twice more, and the paper was skipped — thirteen useful
fields discarded over one the model had simply declined to fill.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from labpilot.research_engine.intelligence.literature.models import PaperKnowledge


def test_blank_grounded_in_is_the_same_as_omitting_it():
    """The exact payload that killed the extraction."""
    knowledge = PaperKnowledge.model_validate(
        {"paper_id": "P1", "title": "A paper", "grounded_in": ""}
    )

    assert knowledge.grounded_in == "abstract"


def test_the_rest_of_the_extraction_survives():
    """The point of the coercion: knowledge is kept, not just validation passed."""
    knowledge = PaperKnowledge.model_validate(
        {
            "paper_id": "P1",
            "techniques": ["gradient boosting"],
            "ideas_worth_testing": ["stack the two heads"],
            "grounded_in": "   ",
        }
    )

    assert knowledge.techniques == ["gradient boosting"]
    assert knowledge.ideas_worth_testing == ["stack the two heads"]


def test_omitting_it_still_works():
    assert PaperKnowledge().grounded_in == "abstract"


@pytest.mark.parametrize("value", ["abstract", "pdf_excerpt", "metadata"])
def test_the_real_values_pass_through(value):
    assert PaperKnowledge.model_validate({"grounded_in": value}).grounded_in == value


def test_an_unrecognized_value_still_fails():
    """Blank is 'I did not say'. 'full_text' is a claim about provenance we
    cannot honour — recording it as `abstract` would misstate what the
    extraction was grounded in, so this must stay loud."""
    with pytest.raises(ValidationError):
        PaperKnowledge.model_validate({"grounded_in": "full_text"})
