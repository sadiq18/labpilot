"""Why the profile says what it says.

M22 step 2. The value plane — ``target_column``, ``id_column`` — is unchanged
and keeps every name it had. Beside it now sits an evidence plane:
``schema.inferences[field]``, holding the signals that fired and a confidence
that is a **function** of them rather than a number someone chose.

Three rules hold the whole thing up:

* **An inference carries no value.** The fact lives in ``profile.target_column``
  and nowhere else, so the two planes cannot disagree — there is only one copy.
* **No call site writes a float.** :meth:`Inference.of` computes it, and a
  validator re-checks on every load, so a hand-set confidence cannot survive a
  round trip through the model.
* **A signal names what was observed**, never where it was read from. ``.csv``,
  a warehouse table and an environment fire the same catalogue.

Confidence is **not a probability**. It is coverage over a fixed checklist: how
much of the evidence that would settle this question actually fired. Two runs
over the same bytes produce the same number, which is what makes it testable.

Design: ``docs/research-os/autonomy-roadmap/design/17-dataset-understanding.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CATALOGUE",
    "Alternative",
    "Band",
    "Inference",
    "Note",
    "RejectedClaim",
    "Signal",
    "SignalSpec",
    "band_of",
    "combine",
]

Band = Literal["asserted", "probable", "uncertain"]

#: Act on it.
ASSERTED_AT = 0.85
#: Act on it, and say so wherever the answer is passed on.
PROBABLE_AT = 0.60


def band_of(confidence: float) -> Band:
    """Which band a confidence falls in. Below `probable`, the answer is asked about."""
    if confidence >= ASSERTED_AT:
        return "asserted"
    if confidence >= PROBABLE_AT:
        return "probable"
    return "uncertain"


@dataclass(frozen=True)
class SignalSpec:
    """One catalogue entry: what the evidence is worth, and what it can never buy."""

    id: str
    weight: float
    #: Which family the weight comes from (§7.4 of the design). Recorded so a
    #: weight can be argued with as a member of a class rather than one at a
    #: time — a per-dataset weight is how a profiler becomes a Kaggle profiler.
    family: Literal["stated", "structural", "distributional", "positional"]
    #: Binds only when *every* signal that fired is capped. A weak rule stops
    #: being the ceiling as soon as real evidence arrives beside it: the
    #: objection to a positional rule is that it decides alone, not that it is
    #: worthless as corroboration.
    cap: float | None = None
    #: What the signal means, in the terms an operator would use.
    means: str = ""


def _spec(
    id: str,
    weight: float,
    family: str,
    *,
    cap: float | None = None,
    means: str = "",
) -> SignalSpec:
    return SignalSpec(id=id, weight=weight, family=family, cap=cap, means=means)  # type: ignore[arg-type]


#: Every signal the profiler can fire today, and nothing else. A catalogue entry
#: with no code that produces it is a declaration nothing reaches — the defect
#: class this milestone removes — so the naming family and `dtype_matches_metric`
#: arrive with the code that observes them, not before.
#:
#: **Being a candidate is the entry condition, not evidence.** There is no
#: "withheld at scoring" signal: when eight columns are withheld, knowing that
#: this one is says nothing about which is the label, and giving every candidate
#: the same points lifts all of them over the ask threshold together — which is
#: how a tie stops looking like one. `sole_withheld_column` fires only where
#: withholding *identifies* the column, and identification is the whole of its
#: value.
#:
#: **A tie broken by sorting is not a signal either.** Where today's code takes
#: `sorted(candidates)[-1]`, every candidate keeps the confidence its own
#: evidence earns and a note records that position chose between them. Paying
#: the winner for being alphabetically last would put it above its twin by
#: exactly the amount of nothing.
CATALOGUE: dict[str, SignalSpec] = {
    entry.id: entry
    for entry in (
        # --- target ---------------------------------------------------------
        _spec(
            "named_in_prediction_template",
            0.80,
            "structural",
            means="the prediction template names this column",
        ),
        _spec(
            "sole_withheld_column",
            0.70,
            "structural",
            means="it is the only column train has and the scoring input does not",
        ),
        _spec(
            "present_across_train_units",
            0.40,
            "distributional",
            means="carried by the modal count of the primary kind's tables",
        ),
        _spec(
            "non_null_in_train",
            0.20,
            "distributional",
            means="observed wherever a label is supposed to be",
        ),
        _spec("is_numeric", 0.15, "distributional", means="numeric, which a label often is"),
        _spec(
            "positional_template_overlap",
            0.10,
            "positional",
            cap=0.50,
            means="second column of the template's overlap with train — position, not evidence",
        ),
        _spec(
            "operator_answer",
            1.00,
            "stated",
            means="a person answered the question — the only evidence that settles one",
        ),
        _spec(
            "declared_by_source",
            0.90,
            "stated",
            means="the environment states it — a competition's metric, a config, a goal",
        ),
        # --- objective ------------------------------------------------------
        _spec(
            "direction_declared",
            0.30,
            "distributional",
            means="the declaration says which way is better, so it is not half a metric",
        ),
        # --- split ----------------------------------------------------------
        _spec(
            "no_scoring_input",
            0.90,
            "structural",
            means="there is no scoring input at all — a fact about the dataset, not a guess",
        ),
        _spec(
            "scored_rows_are_partition_tail",
            0.80,
            "structural",
            means="the scored rows are a contiguous tail of each partition",
        ),
        _spec(
            "scoring_input_present",
            0.50,
            "structural",
            cap=0.75,
            means="a scoring input exists and nothing contradicts disjoint units — the residual",
        ),
        # --- modality -------------------------------------------------------
        _spec(
            "single_modality_present",
            0.70,
            "structural",
            means="only one modality is present, so there is nothing to weigh it against",
        ),
        _spec(
            "no_tabular_data",
            0.90,
            "structural",
            means="no tables at all — an environment to act in, not units to predict",
        ),
        _spec(
            "csv_majority",
            0.40,
            "distributional",
            means="tables outnumber the other modality's files, so they carry the signal",
        ),
        _spec(
            "llm_modality_tiebreak",
            0.30,
            "stated",
            cap=0.60,
            means="a model broke the tie — never on its own worth acting on as settled",
        ),
        # --- prediction unit --------------------------------------------------
        _spec(
            "submission_row_per_scoring_row",
            0.80,
            "structural",
            means="the template has one row per row of the scoring input",
        ),
        _spec(
            "scored_rows_are_a_partition_tail",
            0.60,
            "structural",
            means="the scored rows are a tail of each partition, so rows are not exchangeable",
        ),
        # --- identity -------------------------------------------------------
        _spec(
            "present_in_train_and_scoring",
            0.50,
            "structural",
            means="present on both sides, as a key must be",
        ),
        _spec(
            "unique_per_unit",
            0.40,
            "distributional",
            means="one distinct value per row in the sample",
        ),
        _spec(
            "first_template_column",
            0.10,
            "positional",
            cap=0.50,
            means="first column of the prediction template — position, not evidence",
        ),
    )
}


class Signal(BaseModel):
    """One piece of evidence that fired, named from the catalogue.

    No weight: it lives in :data:`CATALOGUE`, keyed by ``id``, so a stored
    profile cannot disagree with the rule that produced it — the same reason an
    :class:`Inference` carries no value. It also makes :func:`combine` a pure
    function of the signal *ids*, which is what the self-consistency check tests.
    """

    id: str
    #: What was actually seen, for a human reading `research schema show`.
    detail: str = ""

    @model_validator(mode="after")
    def _known_to_the_catalogue(self) -> Signal:
        if self.id not in CATALOGUE:
            raise ValueError(f"unknown signal {self.id!r}; add it to CATALOGUE with a weight")
        return self


def combine(signals: list[Signal]) -> float:
    """Coverage over the checklist, as a number in [0, 1].

    Noisy-OR over the weights the catalogue gives each fired signal: each one
    removes a share of the remaining doubt, so ten weak signals never add up to
    a strong one. Caps bind only when nothing uncapped fired.

    The naming family — a column called ``label``, ``y``, ``target`` — has a
    0.20 ceiling *across all of its members*, and arrives with the first naming
    signal in step 3. Implementing the ceiling before anything can reach it
    would be a guard nothing exercises.
    """
    specs = [CATALOGUE[signal.id] for signal in signals]
    if not specs:
        return 0.0
    raw = 1 - prod(1 - spec.weight for spec in specs)
    caps = [spec.cap for spec in specs if spec.cap is not None]
    bounded = min(raw, *caps) if len(caps) == len(specs) else raw
    return round(bounded, 4)


class Alternative(BaseModel):
    """A candidate that did not win, with the evidence it did fire.

    Recorded rather than dropped: "the answer is `Zone_Depth`" and "the answer is
    `Zone_Depth`, and `Depth` fired identical evidence" are different claims, and
    only the second one lets a reader see a coin flip for what it is.
    """

    candidate: str
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[Signal] = Field(default_factory=list)

    @classmethod
    def of(cls, candidate: str, signals: list[Signal]) -> Alternative:
        return cls(candidate=candidate, confidence=combine(signals), signals=signals)

    @model_validator(mode="after")
    def _confidence_is_derived(self) -> Alternative:
        expected = combine(self.signals)
        if abs(self.confidence - expected) > 1e-9:
            raise ValueError(
                f"confidence {self.confidence} for {self.candidate!r} is not "
                f"combine(signals)={expected}"
            )
        return self


class RejectedClaim(BaseModel):
    """A claim the data contradicted, kept rather than discarded.

    A declaration or a model proposal that turned out to be wrong is evidence
    about the *source*, and dropping it silently is how a workspace ends up
    unable to say why it disbelieved something.
    """

    claim: str
    source: Literal["declared", "llm", "operator"] = "declared"
    refuted_by: str = ""


class Inference(BaseModel):
    """Why the value plane says what it says. Carries no value.

    Build with :meth:`of`. The validator makes a hand-set confidence fail on
    construction *and* on every load, so the self-consistency property holds for
    profiles written by older code as well as by this one.
    """

    signals: list[Signal] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    band: Band
    alternatives: list[Alternative] = Field(default_factory=list)
    rejected: list[RejectedClaim] = Field(default_factory=list)

    @classmethod
    def of(
        cls,
        signals: list[Signal],
        *,
        alternatives: list[Alternative] | None = None,
        rejected: list[RejectedClaim] | None = None,
    ) -> Inference:
        confidence = combine(signals)
        return cls(
            signals=signals,
            confidence=confidence,
            band=band_of(confidence),
            alternatives=alternatives or [],
            rejected=rejected or [],
        )

    @model_validator(mode="after")
    def _confidence_and_band_are_derived(self) -> Inference:
        expected = combine(self.signals)
        if abs(self.confidence - expected) > 1e-9:
            raise ValueError(f"confidence {self.confidence} is not combine(signals)={expected}")
        if self.band != band_of(self.confidence):
            raise ValueError(f"band {self.band!r} is not band_of({self.confidence})")
        return self


class Note(BaseModel):
    """Something the profile has to say, in a form a decision can read.

    ``warnings`` was a list of prose that four things render and nothing parses.
    It stays, as a computed view over these, so every existing reader keeps
    working while a consumer that wants to *act* — M23's gate, M25's findings —
    reads ``code`` instead of matching substrings.
    """

    code: str
    text: str
    field: str | None = None
    severity: Literal["info", "caution", "blocking"] = "info"
