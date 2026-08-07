r"""A recipe miner must not invent technique identities.

The fallback took the first word before "feature" —
``\b([A-Za-z][A-Za-z0-9_]{2,40})\s+feature`` — which matches any English word of
three or more letters. Measured on rogii 2026-08-07: beliefs existed for `the`,
`add`, `built`, `computed`, `average`, `context`, `model`, `neighbour`, `tangent`
and `booster`, and a campaign asked the code engineer to implement `the`.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.intelligence.feature_recipes import _name_from_sentence

#: Every one of these produced a belief row on rogii.
_JUNK_SOURCES = [
    "We added the features to the model",
    "add features for context",
    "built new features here",
    "the computed features were dropped",
    "average features across the window",
    "context features help the model",
    "model features were rebuilt",
    "neighbour features by distance",
    "tangent features from the curve",
    "booster features tuned later",
]


@pytest.mark.parametrize("sentence", _JUNK_SOURCES)
def test_english_prose_does_not_mint_a_technique(sentence):
    """An unrecognised recipe stays generic rather than naming itself.

    A name we cannot stand behind is worse than no name: it becomes an identity
    in the ledger, accrues beliefs, and competes with techniques that were
    actually measured.
    """
    assert _name_from_sentence(sentence) == "feature_engineering"


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Compute rolling features over the window", "rolling_features"),
        ("use lag features", "lag_features"),
        ("apply target encoding to the categoricals", "target_encoding"),
        ("one-hot encode the categories", "one_hot_encoding"),
        ("TF-IDF over the text column", "tfidf"),
        ("polynomial expansion of the numerics", "polynomial_features"),
        ("aggregations grouped by well", "aggregation_features"),
        ("log1p the skewed columns", "log1p_transform"),
    ],
)
def test_recognised_techniques_still_resolve(sentence, expected):
    """Removing the scrape must not cost the names that were real."""
    assert _name_from_sentence(sentence) == expected


def test_the_specific_string_that_reached_production():
    """`technique candidate: 'the' has no deterministic recipe; codegen
    implements it from the hypothesis description` — from a live campaign log."""
    assert _name_from_sentence("We added the features to the model") != "the"
