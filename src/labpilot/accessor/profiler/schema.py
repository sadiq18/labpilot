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
    "ModalityPresence",
    "PredictionUnit",
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
#: ``temporal_split`` and ``same_entities_new_period`` arrive with the detectors
#: that can conclude them; ``environment`` arrived with its own in step 5.
SplitRelationship = Literal[
    "partition_suffix",
    "disjoint_units",
    "no_test_provided",
    #: The scoring input is an environment to act in, not units to predict.
    #: Arrives here with its detector: a dataset with no tables at all.
    "environment",
    "unknown",
]

#: What one prediction is about.
#:
#: * ``row`` — one row of the scoring input, the ordinary case.
#: * ``partition_row`` — one row of one partition, where the scored rows are a
#:   tail of each: predictions are not exchangeable across partitions.
#: * ``episode`` — an environment run rather than a row.
#: * ``unknown`` — not concluded.
#:
#: The question that makes `unit_count`, the split and M23's floor mean the same
#: thing across modalities: a row, an image, a clip and an episode are all "one
#: unit" and only this says which.
PredictionUnit = Literal["row", "partition_row", "episode", "unknown"]

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


class ModalityPresence(BaseModel):
    """One modality the dataset contains, and whether it is the main one.

    A **list**, because a dataset is often more than one thing: rogii is 1,546
    per-well tables *and* a directory of PNG previews, and the detector used to
    count the images, prefer tabular, and throw the images away — so nothing
    downstream could know they existed. Preferring tabular is still right;
    discarding the rest was not.
    """

    #: ``audio`` has no detector and is never *inferred*; it is here because
    #: profiles on disk carry it — birdclef's says `modality: "audio"` — and
    #: adopting a legacy string into this list is a real producer. Detecting it
    #: is a separate piece of work with its own fixture.
    modality: Literal["tabular", "text", "image", "audio", "environment"]
    role: Literal["primary", "auxiliary"]
    #: What was seen, in the terms an operator would use.
    detail: str = ""
    image_dir: str | None = None
    image_column: str | None = None
    text_column: str | None = None


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
    #: What the truth must look like, from the metric registry's `target_kind`.
    #: Carried rather than re-derived: `accessor` may not import the registry,
    #: and a second list of metric keys here would be the third overlapping
    #: vocabulary — the defect #145 removed and a test now forbids.
    target_kind: Literal["continuous", "discrete", "any"] | None = None
    source: Literal["declared", "operator"] = "declared"
