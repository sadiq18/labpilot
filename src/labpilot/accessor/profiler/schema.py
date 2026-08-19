"""Value-plane types the profiler and its sources both name.

M22 step 3. These live apart from `tabular.py` because `source.py` declares some
of them too — a source states the metric it was told about — and a shared type
imported by both is the only arrangement that does not create a cycle.

Every member of every enum here has a producer. A `SplitRelationship` the
profiler cannot yet conclude, or an `ExclusionReason` nothing assigns, would be
a declaration nothing reaches, and the milestone exists to remove those: the
values that arrive with their detectors are listed in the docstrings below.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = [
    "ExclusionReason",
    "MetricRef",
    "SplitRelationship",
]

#: How the scored units relate to the training units.
#:
#: * ``partition_suffix`` — the scored rows are a contiguous tail of each test
#:   partition, so validation has to hold out each partition's tail.
#: * ``disjoint_units`` — a scoring input exists and nothing suggests otherwise.
#:   The residual conclusion, and capped accordingly: "IID" is what is left when
#:   nothing else fired, and an independence assumption that is wrong turns a CV
#:   score into fiction.
#: * ``no_test_provided`` — there is no scoring input at all. A fact, not a
#:   guess, and the ordinary case for a dataset that is not a competition.
#: * ``unknown`` — not concluded.
#:
#: ``temporal_split``, ``same_entities_new_period`` and ``environment`` arrive
#: with the detectors that can conclude them.
SplitRelationship = Literal[
    "partition_suffix",
    "disjoint_units",
    "no_test_provided",
    "unknown",
]

#: Why a column is not a feature. Each is a *measurement* — two people with the
#: same bytes would agree — so exclusions carry a reason rather than a
#: confidence.
#:
#: * ``is_target`` / ``is_id`` — resolved elsewhere in the schema.
#: * ``unavailable_at_scoring`` — in train and not in the scoring input, so a
#:   model that uses it cannot run.
#: * ``equals_target`` — equal to the target wherever both are present. rogii's
#:   ``TVT_input``: the strongest predictor in the dataset and a leak as a plain
#:   feature, because a model learns "copy" and then meets NaN on every scored
#:   row.
#: * ``constant`` — one distinct value, so it carries nothing.
#:
#: ``post_outcome`` and ``operator_excluded`` arrive with the timestamp
#: comparison and the answer file that produce them.
ExclusionReason = Literal[
    "is_target",
    "is_id",
    "unavailable_at_scoring",
    "equals_target",
    "constant",
]


class MetricRef(BaseModel):
    """What the dataset is scored by, and how that was reached.

    ``source`` is the part that generalises. On Kaggle the metric is declared;
    off it, the metric comes from the goal a person stated, and the difference
    between "declared" and "guessed from the target's shape" is exactly what a
    confidence has to be able to express.

    Resolution to a canonical ``key`` belongs to the caller, not here:
    ``accessor`` may not import ``research_engine``, where the metric registry
    lives. A source states what it was told; the profiler records it and says
    where it came from.
    """

    name: str
    key: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    source: Literal["declared", "operator"] = "declared"
