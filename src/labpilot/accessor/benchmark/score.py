"""Score the shipped path against a captured competition.

M24. Two rules keep the number meaningful:

* **The harness runs the shipped path**, never a reimplementation. Observed
  values are read from the *serialized* profile, because a fact that is correct
  in memory and lost on serialization is not correct.
* **Partial credit is per criterion, never fractional.** "Target right, id
  wrong" is two criteria with two answers. The table *is* the partial credit,
  and it is legible; a fractional per-fixture score just invites tuning the
  denominator. `understood` is the strict aggregate: every applicable criterion
  passes, or the competition is not understood.

Five verdicts, and three of them are neither pass nor fail:

* ``not_applicable`` — the competition has no such answer to get right.
* ``unverifiable`` — the *capture* cannot speak to it. Scoring it would measure
  the truncation, not the profiler.
* ``known_failure`` — it is wrong today, on purpose, and the fixture says so.
  The day it goes green is visible rather than silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from labpilot.accessor.benchmark.fixture import CompetitionFixture
from labpilot.accessor.profiler.schema import MetricRef
from labpilot.accessor.profiler.source import DeclaredFacts

__all__ = [
    "CRITERIA",
    "CriterionResult",
    "Scorecard",
    "disagreements",
    "profile_and_score",
    "score_fixture",
    "score_full_dataset",
]

Verdict = Literal["pass", "fail", "not_applicable", "unverifiable", "known_failure"]

#: The criteria this corpus scores today. Baseline criteria arrive with the
#: floor and the generic model; listing one here before it can be produced would
#: score every fixture `fail` on a mechanism that does not exist.
CRITERIA = (
    "target_column",
    "id_columns",
    "train_test_relationship",
    "modality",
    "feature_columns",
    "metric_name",
    "abstention",
    # M24 tier 1's "dummy baseline 100%", read honestly: not that the floor
    # scored well — a floor that scored well is a gate no model can pass — but
    # that the dumbest defensible answer produces a submission the competition
    # would accept. Scored by the harness rather than declared per fixture,
    # because it is a property of the run and not of the capture.
    "dummy_baseline",
)


class CriterionResult(BaseModel):
    criterion: str
    verdict: Verdict
    expected: str = ""
    observed: str = ""
    detail: str = ""


class Scorecard(BaseModel):
    slug: str
    results: list[CriterionResult] = Field(default_factory=list)

    @property
    def understood(self) -> bool:
        """Every applicable criterion passed, and none is red on purpose.

        A fixture with one `fail` is not 6/7 understood — it is a competition
        the system has the wrong idea about, and the aggregate the plan asks for
        (*"dataset understanding > 95%"*) is the strict one.

        `known_failure` counts against it too. A declared defect is still a
        defect: `playground-series-s6e7` optimises accuracy where the rules say
        balanced accuracy, and calling that understood because the fixture
        admits it would make the number congratulate us for knowing.
        """
        return all(
            result.verdict not in ("fail", "known_failure") for result in self.results
        ) and any(result.verdict == "pass" for result in self.results)

    def verdict_for(self, criterion: str) -> Verdict | None:
        for result in self.results:
            if result.criterion == criterion:
                return result.verdict
        return None


def _observed(profile: dict, criterion: str) -> object:
    if criterion == "metric_name":
        metric = profile.get("metric") or {}
        return metric.get("key") or metric.get("name")
    if criterion == "abstention":
        return None
    return profile.get(criterion)


def _matches(expected: object, observed: object) -> bool:
    if isinstance(expected, list) and isinstance(observed, list):
        return list(expected) == list(observed)
    return expected == observed


def score_fixture(
    fixture: CompetitionFixture,
    profile: dict,
    open_questions: list[str],
    dummy: object | None = None,
) -> Scorecard:
    """Compare a serialized profile against what the fixture says is true.

    `open_questions` is the field list M22's questions name — the input for the
    abstention criterion, where the right answer is *"it should have refused"*.
    """
    results: list[CriterionResult] = []
    expectations = fixture.expected.model_dump()

    for criterion in CRITERIA:
        if criterion == "abstention":
            results.append(_score_abstention(fixture, open_questions))
            continue
        if criterion == "dummy_baseline":
            results.append(_score_dummy(dummy))
            continue
        # `must_ask` first. A field that is both expected-to-be-asked and
        # unverifiable reported "the capture cannot say" when the truth is "the
        # system was supposed to refuse" — the abstention criterion still scored
        # it, so nothing broke, but the per-criterion table gave the wrong
        # reason, and the table is the whole point of per-criterion credit.
        if criterion in fixture.expected.must_ask:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    verdict="not_applicable",
                    detail="expected to be asked about, not answered",
                )
            )
            continue
        if criterion in fixture.unverifiable:
            results.append(
                CriterionResult(
                    criterion=criterion,
                    verdict="unverifiable",
                    detail=fixture.unverifiable[criterion],
                )
            )
            continue
        expected = expectations.get(criterion)
        if expected is None:
            results.append(CriterionResult(criterion=criterion, verdict="not_applicable"))
            continue
        observed = _observed(profile, criterion)
        passed = _matches(expected, observed)
        verdict: Verdict = "pass" if passed else "fail"
        if not passed and criterion in fixture.known_failures:
            verdict = "known_failure"
        results.append(
            CriterionResult(
                criterion=criterion,
                verdict=verdict,
                expected=json.dumps(expected),
                observed=json.dumps(observed),
                detail=fixture.known_failures.get(criterion, "") if not passed else "",
            )
        )
    return Scorecard(slug=fixture.slug, results=results)


def _score_dummy(dummy: object | None) -> CriterionResult:
    """Whether the floor emitted a submission the competition would accept.

    `None` means nobody tried, which is what a headers-only capture leaves: with
    no rows there is no constant to fit and no sample to shape, so the criterion
    is `unverifiable` rather than failed. Scoring it as a miss would be measuring
    the truncation, which is the rule the whole corpus runs on.
    """
    if dummy is None:
        return CriterionResult(
            criterion="dummy_baseline",
            verdict="unverifiable",
            detail="no rows in this capture, so no submission could be emitted",
        )
    valid = bool(getattr(dummy, "valid", False))
    reasons = list(getattr(dummy, "reasons", ()) or [])
    return CriterionResult(
        criterion="dummy_baseline",
        verdict="pass" if valid else "fail",
        expected=json.dumps("a submission the competition would accept"),
        observed=json.dumps("valid" if valid else reasons[:3]),
        detail="" if valid else "; ".join(reasons),
    )


def _score_abstention(fixture: CompetitionFixture, open_questions: list[str]) -> CriterionResult:
    """Did the system refuse exactly where it should have?

    Both directions are failures, and they are different ones: answering a
    question it could not know is a guess, and asking about something it could
    have resolved is noise that teaches an operator to dismiss the prompt.
    """
    expected = sorted(fixture.expected.must_ask)
    asked = sorted(open_questions)
    if not expected and not asked:
        return CriterionResult(criterion="abstention", verdict="not_applicable")
    return CriterionResult(
        criterion="abstention",
        verdict="pass" if expected == asked else "fail",
        expected=json.dumps(expected),
        observed=json.dumps(asked),
        detail="" if expected == asked else "asked about a different set than expected",
    )


def profile_and_score(fixture_dir: Path, workdir: Path) -> Scorecard:
    """Expand, run the shipped profiler, and score. One call per fixture.

    Deliberately the *profiler* rather than a hand-rolled reader: the corpus
    exists to measure what the system does, and anything reimplemented here
    would be measuring this file instead.
    """
    from labpilot.accessor.benchmark.expand import expand_fixture

    fixture = expand_fixture(fixture_dir, workdir)
    return _score_directory(fixture, workdir, _declared_facts(workdir))


def score_full_dataset(
    fixture: CompetitionFixture, data_dir: Path, declared_from: Path | None = None
) -> Scorecard:
    """The same scoring, against the real dataset instead of the fixture.

    Tier 3. The corpus is only allowed to stand in for real data if the two
    agree, and this is the half that reads the real thing — same profiler, same
    expectations, same scorer, so a difference in the result is a difference in
    the *capture* and nothing else.

    `declared_from` is the expanded fixture, whose `competition.json` is carried
    byte-verbatim. Reading it from there rather than from the dataset directory
    is deliberate: the spec is not what truncation destroyed, and a full run that
    scored `metric_name` as unknowable purely because Kaggle ships no
    `competition.json` would report a disagreement the capture did not cause.
    """
    return _score_directory(fixture, Path(data_dir), _declared_facts(declared_from or data_dir))


def _score_directory(
    fixture: CompetitionFixture, directory: Path, declared: DeclaredFacts
) -> Scorecard:
    from labpilot.accessor.profiler.questions import pending_schema_questions
    from labpilot.accessor.profiler.source import LocalFileSource
    from labpilot.accessor.profiler.tabular import TabularProfiler
    from labpilot.config import ProfilerConfig

    source = LocalFileSource(Path(directory), declared)
    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, fixture.slug)
    questions = [question.field for question in pending_schema_questions(profile)]
    return score_fixture(fixture, json.loads(profile.model_dump_json()), questions)


#: Verdicts a fixture is *claiming* it can score. `unverifiable` and
#: `not_applicable` are the fixture saying it cannot speak to a criterion, so
#: they are exactly the ones tier 3 must not hold it to.
_CLAIMED = ("pass", "fail", "known_failure")


def disagreements(hermetic: Scorecard, full: Scorecard) -> dict[str, tuple[str, str]]:
    """Criteria where the fixture and the real dataset reach different verdicts.

    Empty is the licence the hermetic corpus runs on. A non-empty result names
    criteria whose **capture mode is wrong** — the plan's remedy is to move them
    to `unverifiable` rather than to argue with the number, because a fixture
    that quietly answers differently from the dataset it stands for is worse
    than one that admits it cannot answer.
    """
    verdicts = {result.criterion: result.verdict for result in full.results}
    return {
        result.criterion: (result.verdict, verdicts.get(result.criterion, "missing"))
        for result in hermetic.results
        if result.verdict in _CLAIMED
        and verdicts.get(result.criterion, "missing") != result.verdict
    }


def _declared_facts(workdir: Path) -> DeclaredFacts:
    """What the captured `competition.json` states, if it captured one.

    The metric criterion is unscoreable without it, and carrying the file is
    ~1 KB — cheaper than any other way of knowing what a competition is scored
    by. Resolution stays outside `accessor`, so only what the file already holds
    is used here.
    """
    path = workdir / "competition.json"
    if not path.is_file():
        return DeclaredFacts()
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DeclaredFacts()
    metric = spec.get("evaluation_metric") or {}
    ref = None
    if metric.get("name"):
        direction = metric.get("direction")
        ref = MetricRef(
            name=metric["name"],
            key=metric.get("key"),
            direction=direction if direction in ("maximize", "minimize") else None,
        )
    return DeclaredFacts(
        title=spec.get("title", ""), description=spec.get("description", ""), metric=ref
    )
