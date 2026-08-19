"""What is being optimised, how it is measured, and which way is better.

`ObjectiveSpec` is the answer to *"what does better mean here?"* — and it carries
where that answer came from, how much of the evidence that would settle it
actually fired, and the evidence itself. A campaign that cannot state its
objective with justification should not launch experiments against it.

**The resolution order, and why there are two of them.**

Identifying the metric and orienting it are different questions with different
best sources:

    metric identity:  explicit spec > registry > evaluation code > rules > LLM > ask
    direction:        probe > explicit spec > registry > rules > LLM > ask

Direction leads with the probe because, once an evaluator is executable,
direction is *observable* (`direction_probe`) rather than declarable. A registry
entry is a claim; a probe is a measurement of the scorer that will actually be
used. Where the two disagree the objective is not resolved — it is contradictory,
which is a reason to stop rather than to pick the more convenient one.

**The LLM's place is level 5, and it is not implemented here.** The design is
that it returns a structured candidate which deterministic code then verifies —
it discovers semantics, it does not configure experiments. Levels 1–4 resolve
every Kaggle-shaped competition on their own, and baseline reliability must not
depend on a model being reachable.

Nothing here defaults. An unresolved objective says so, and
`ObjectiveSpec.blocks_launch` is what a preflight consults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.competition.direction_probe import (
    Direction,
    probe_metric_direction,
)
from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    _slug,
    direction_of,
    is_scorable,
    normalize_metric_key,
    requires_probabilities,
)

#: Where a field's value came from, strongest first. Ordering is meaningful:
#: `_rank` uses it to decide whether a later source may overwrite an earlier one.
ObjectiveSource = Literal[
    "operator",       # a human answered; nothing outranks this
    "measured",       # probed against the executable evaluator
    "explicit",       # the competition's own evaluation specification
    "registry",       # a catalogued metric
    "code",           # an evaluator found in the workspace
    "rules",          # morphology of the metric's name
    "llm",            # proposed by a model and structurally verified
    "unknown",
]

_RANK: dict[str, int] = {
    "operator": 0,
    "measured": 1,
    "explicit": 2,
    "registry": 3,
    "code": 4,
    "rules": 5,
    "llm": 6,
    "unknown": 7,
}

#: Confidence earned by each source. `rules` sits at the bottom of "probable" so
#: it acts but carries its alternatives; `llm` sits below the ask threshold by
#: construction, so a model-proposed objective always asks before it is used.
_CONFIDENCE: dict[str, float] = {
    "operator": 1.00,
    "measured": 0.99,
    "explicit": 0.95,
    "registry": 0.90,
    "code": 0.85,
    "rules": 0.60,
    "llm": 0.55,
    "unknown": 0.0,
}

#: Below this an objective is not actionable: ask when there is someone to ask,
#: refuse to launch when there is not. Matches M22's band for a schema field —
#: one threshold for "we do not know this well enough to spend money on it".
ACTIONABLE_CONFIDENCE = 0.60

#: Name fragments that orient a metric nobody catalogued. Deliberately small and
#: morphological: this is level 4, a last deterministic resort before asking, not
#: an attempt to enumerate the metric space.
_MINIMIZE_HINTS: tuple[str, ...] = (
    "error", "loss", "deviation", "distance", "misfit", "residual", "regret",
    "cost", "penalty", "divergence", "perplexity",
)
_MAXIMIZE_HINTS: tuple[str, ...] = (
    "accuracy", "score", "gain", "precision", "recall", "auc", "auroc", "auprc",
    "f1", "iou", "dice", "ndcg", "map", "correlation", "agreement", "reward",
    "win", "coverage", "lift",
)


class ObjectiveSpec(BaseModel):
    """What is optimised, how it is scored, and which way is better."""

    task: str | None = None
    target: str | None = None

    #: Canonical key when the metric is catalogued, else the slug of whatever was
    #: stated. A slug is a perfectly good stable identity for comparison — only
    #: *aliasing* needs a catalogue.
    metric_name: str | None = None
    #: What was actually said, before normalisation, so the trail is auditable.
    metric_raw: str = ""
    #: Whether `compute_metric` can produce this number. Naming a metric and
    #: being able to score it are different capabilities.
    scorable: bool = False

    direction: Direction | None = None
    direction_source: ObjectiveSource = "unknown"

    #: Where the metric's *identity* came from, separately from its direction.
    identity_source: ObjectiveSource = "unknown"
    #: The weaker of the two, so it always explains `confidence`. Reporting the
    #: stronger one made `source='measured', confidence=0.90` unreadable — the
    #: 0.90 came from the registry and nothing said so.
    source: ObjectiveSource = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    #: The answers a question could offer, when one can be asked. Empty means
    #: there is nothing to choose between — not that the choice is unimportant.
    alternatives: list[str] = Field(default_factory=list)
    #: Field names that could not be resolved. Non-empty means "ask", and names
    #: *what* to ask about — an objective that says "uncertain" without saying
    #: which part is not actionable.
    unresolved: list[str] = Field(default_factory=list)
    #: Set when two sources disagree. A contradiction is not low confidence; it
    #: is a reason to stop, because one of the sources is wrong.
    contradiction: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Whether experiments may be launched against this objective."""
        return (
            self.contradiction is None
            and not self.unresolved
            and self.direction is not None
            and self.metric_name is not None
            and self.confidence >= ACTIONABLE_CONFIDENCE
        )

    @property
    def blocks_launch(self) -> bool:
        return not self.is_actionable

    def why_blocked(self) -> str:
        """One line an operator can act on, or empty when nothing blocks."""
        if self.contradiction:
            return f"objective is contradictory: {self.contradiction}"
        if self.metric_name is None:
            return "no evaluation metric could be resolved for this competition"
        if "local_scoring" in self.unresolved:
            return (
                f"{self.metric_name!r} is the stated objective and nothing here can "
                "compute it, so cross-validation would optimise a proxy without "
                "saying so — name the proxy deliberately, or add an implementation"
            )
        if self.direction is not None and self.unresolved:
            return f"objective under-specified: {', '.join(self.unresolved)}"
        if self.direction is None:
            return (
                f"cannot tell whether {self.metric_name!r} should be maximised or "
                "minimised, and guessing inverts every conclusion drawn from it"
            )
        if self.confidence < ACTIONABLE_CONFIDENCE:
            return (
                f"objective resolved only from {self.source!r} "
                f"(confidence {self.confidence:.2f} < {ACTIONABLE_CONFIDENCE:.2f})"
            )
        return ""


def infer_direction_from_name(metric_name: str) -> tuple[Direction | None, str | None]:
    """Level 4: orient a metric from the morphology of its name.

    Returns `(direction, matched_hint)`. This is the last deterministic resort
    before asking, and it is deliberately weak — an `error` is minimised and a
    `score` is maximised often enough to be worth trying, and not often enough to
    be trusted at `asserted`.

    A name matching both families resolves to None rather than to whichever list
    was checked first. `mean_absolute_error_score` is not a question this should
    answer confidently.
    """
    slug = metric_name.strip().lower().replace("-", "_").replace(" ", "_")
    lo = next((h for h in _MINIMIZE_HINTS if h in slug), None)
    hi = next((h for h in _MAXIMIZE_HINTS if h in slug), None)
    if lo and hi:
        return None, None
    if lo:
        return "minimize", lo
    if hi:
        return "maximize", hi
    return None, None


def _rank(source: ObjectiveSource) -> int:
    return _RANK.get(source, _RANK["unknown"])


def resolve_direction(
    metric_key: str | None,
    *,
    declared: Direction | None = None,
    declared_source: ObjectiveSource = "explicit",
    num_classes: int | None = None,
    probe: bool = True,
) -> tuple[Direction | None, ObjectiveSource, list[str], str | None]:
    """Direction, its source, the evidence, and any contradiction.

    Probe first when the metric is executable: a measurement of the scorer that
    will be used outranks anything anyone declared about its name. When the probe
    and a declaration disagree, neither wins — the objective is contradictory and
    the caller must stop.
    """
    evidence: list[str] = []

    measured: Direction | None = None
    if probe and metric_key and is_scorable(metric_key):
        reading = probe_metric_direction(
            metric_key,
            needs_probabilities=requires_probabilities(metric_key),
            num_classes=num_classes,
        )
        if reading.direction is not None:
            measured = reading.direction
            evidence.append(
                f"probed the evaluator: {reading.direction} "
                f"({len(reading.evidence)} pairs agreed)"
            )
        elif reading.error:
            evidence.append(f"probe inconclusive: {reading.error}")

    if measured and declared and measured != declared:
        return (
            None,
            "unknown",
            evidence,
            f"the evaluator measures as {measured} but {declared_source} says {declared}",
        )
    if measured:
        return measured, "measured", evidence, None

    if declared:
        evidence.append(f"{declared_source} states {declared}")
        return declared, declared_source, evidence, None

    catalogued = direction_of(metric_key) if metric_key else None
    if catalogued:
        evidence.append(f"registry: {metric_key} is {catalogued}")
        return catalogued, "registry", evidence, None

    if metric_key:
        inferred, hint = infer_direction_from_name(metric_key)
        if inferred:
            evidence.append(f"name contains {hint!r}, which reads as {inferred}")
            return inferred, "rules", evidence, None

    return None, "unknown", evidence, None


def resolve_objective(
    *,
    metric_raw: str | None,
    declared_direction: Direction | None = None,
    task: str | None = None,
    target: str | None = None,
    probe: bool = True,
) -> ObjectiveSpec:
    """Resolve an objective from what the competition stated.

    Levels 1–4 only. Level 5 (LLM proposal, structurally verified) and level 6
    (human clarification) are the caller's to add — the second is
    `blocks_launch`.
    """
    evidence: list[str] = []
    unresolved: list[str] = []

    if not (metric_raw or "").strip():
        return ObjectiveSpec(
            task=task,
            target=target,
            unresolved=["metric"],
            evidence=["no evaluation metric stated"],
        )

    key = normalize_metric_key(metric_raw)
    if key:
        evidence.append(f"{metric_raw!r} resolves to {key!r}")
        source: ObjectiveSource = "explicit" if declared_direction else "registry"
    else:
        # Unknown metric: the slug is still a stable identity, which is enough to
        # compare two readings of it. Only aliasing needs a catalogue.
        key = _slug(metric_raw)
        evidence.append(f"{metric_raw!r} is not catalogued; using {key!r} as its identity")
        source = "explicit" if declared_direction else "rules"

    # The resolver knows the task from the competition; the probe does not and
    # must not. All it needs is the shape facts `compute_metric` asks for.
    num_classes = 2 if (task or "").endswith("classification") else None
    direction, direction_source, direction_evidence, contradiction = resolve_direction(
        key,
        declared=declared_direction,
        num_classes=num_classes,
        probe=probe,
    )
    evidence.extend(direction_evidence)

    if direction is None and contradiction is None:
        unresolved.append("direction")

    # Knowing the objective and being able to *measure* it are different. When
    # `compute_metric` has no implementation, the selector falls back to a
    # supported metric and CV silently optimises a proxy — which is the
    # metric-mismatch class this whole layer exists to remove, arriving one
    # level up. Measured on disk: playground-series-s6e7 states
    # `balanced_accuracy_score` and every campaign scored plain accuracy.
    if not is_scorable(key):
        unresolved.append("local_scoring")

    confidence = 0.0 if contradiction else min(
        _CONFIDENCE[source], _CONFIDENCE[direction_source]
    )

    # What an operator could actually be asked. A blocked objective whose only
    # gap is orientation has exactly two answers, and offering them is the
    # difference between a question and a wall.
    alternatives: list[str] = []
    if contradiction or "direction" in unresolved:
        alternatives = ["maximize", "minimize"]

    return ObjectiveSpec(
        task=task,
        target=target,
        metric_name=key,
        metric_raw=metric_raw,
        scorable=is_scorable(key),
        direction=direction,
        direction_source=direction_source,
        identity_source=source,
        source=source if _rank(source) >= _rank(direction_source) else direction_source,
        confidence=confidence,
        evidence=evidence,
        unresolved=unresolved,
        alternatives=alternatives,
        contradiction=contradiction,
    )
