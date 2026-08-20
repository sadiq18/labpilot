"""What a hypothesis scored, and everything needed to compare it.

This is the seam M12 is about, and it was already here — unnamed. Every verdict
in the system passes through `build_evidence_card`, which takes exactly this
triple as loose arguments:

    build_evidence_card(treatment_metrics=..., control_metrics=..., maximize=...)

Score, control, direction. What made that Kaggle-shaped was never the shape; it
was *where each part came from* — the score from `cv_`-prefixed keys a training
template wrote, the direction from a `competition.json` that never saw the
number it is being asked to orient, and a second opinion from a public
leaderboard.

Naming the triple is the whole change. A validator that runs a benchmark
harness, reproduces a paper, or executes a test suite produces the same three
facts; it just has nowhere to put them today, and no way to say "I already know
which way is better" without a competition file to write it in.

The protocol is three lines because the interesting content is the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from labpilot.research_engine.intelligence.competition.metric_vocabulary import MetricDirection


@dataclass(frozen=True)
class ValidationResult:
    """One measurement of one hypothesis, with the context to compare it.

    Three fields carry the load, and each answers a failure this repo recorded
    rather than a shape somebody liked:

    * **`metric` travels with the score.** `build_evidence_card` already refuses
      to subtract two runs scored on different keys — measured on rogii, six
      cards recorded a "gain" of -194.30 by comparing a stub's `cv_accuracy` of
      0.5 against a real `cv_rmse` of 194.80. It recovers that key by re-reading
      the blob; carrying it removes the re-derivation.

    * **`direction` is nullable, and is not a `bool`.** `maximize: bool = True`
      is what recorded rogii's one genuine improvement (194.80 -> 190.97) as
      `rejected`. `None` is how a validator says *"I could not tell"* without
      inventing a sign, and the caller must refuse rather than guess.

    * **`secondary` is a role, not a leaderboard.** A public leaderboard, a
      held-out set and a replication attempt are structurally one thing: a
      second measurement the first could not see. Naming it `lb_gain` is what
      made a leaderboard look mandatory.

    `raw` is the untouched blob the score came from. It stays because the
    evidence card reads more than the primary metric out of it — `cv_std`,
    `train_time_s`, `peak_memory_mb` — and because a result that discards its
    own source cannot be re-derived when the extraction improves.
    """

    score: float | None
    metric: str
    direction: MetricDirection | None
    source: str
    provenance: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    secondary: float | None = None

    @property
    def maximize(self) -> bool | None:
        """The legacy boolean, for callers that still speak it.

        `None` where direction is unknown, so a caller cannot silently read
        "unknown" as "maximize" — the exact narrowing that inverted every verdict
        on an MSE competition.
        """
        return None if self.direction is None else self.direction == "maximize"


class HypothesisValidator(Protocol):
    """Turns a hypothesis into a comparable result."""

    def validate(
        self, hypothesis_id: str | None, workspace: Any, context: Any
    ) -> ValidationResult:
        """Run whatever this domain means by "try it", and report what it scored.

        `hypothesis_id`, not `hypothesis`: the plan calls this parameter
        `hypothesis`, and the only production call site passes
        `context.plan.hypothesis_id` — a `str | None`. The Kaggle validator
        ignores the argument, so the mismatch would have stayed invisible until
        an implementation that *needs* the hypothesis read the parameter the
        protocol told it held one, got `"H-014"` or `None`, and either raised on
        attribute access or validated the wrong thing.

        A validator needing more than the id resolves it from `HypothesisStore`
        with `context.competition` and `context.paths.base_dir`. Passing the
        resolved object here instead would put a store read on every comparison
        for a value nothing currently uses.
        """
        ...
