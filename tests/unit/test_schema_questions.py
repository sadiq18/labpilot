"""Ask, or warn and block — never answer for the operator.

M22 step 4. The tie the earlier steps made *visible* now has consequences: a
campaign that cannot say which column is the label stops and says so, and the
only way past it is a person.

Goal 5 of the design, and the one test here that would be worthless without a
mutation check: `test_an_unanswerable_campaign_stops` is proved by deleting the
block and watching the campaign run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers.campaign_harness import CampaignHarness, ok
from helpers.dataset_shapes import build_partitioned_without_template, build_strong_signals

from labpilot.accessor.profiler.questions import (
    ANSWERS_FILENAME,
    BLOCKING_FIELDS,
    answers_fingerprint,
    load_answers,
    open_questions,
    pending_schema_questions,
    question_id,
    record_answer,
)
from labpilot.accessor.profiler.source import DeclaredFacts, LocalFileSource
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig


def _tie_profile(tmp_path: Path, answers: dict[str, str] | None = None) -> DatasetProfile:
    """The partitioned shape with no template: two candidates, identical evidence."""
    data_dir = build_partitioned_without_template(tmp_path)
    source = LocalFileSource(data_dir, DeclaredFacts(answers=answers or {}))
    return TabularProfiler(ProfilerConfig()).profile_dataset(source, "tie")


# --- questions are derived ---------------------------------------------------


def test_an_uncertain_target_raises_a_question(tmp_path: Path) -> None:
    """The tie from step 2, now with a consequence."""
    profile = _tie_profile(tmp_path)

    questions = pending_schema_questions(profile)

    assert [q.field for q in questions] == ["target_column"]
    question = questions[0]
    assert question.provisional == "Zone_Depth"
    assert [c.candidate for c in question.candidates] == ["Depth"]
    # Every candidate, with what it fired — the operator is answering from the
    # same evidence the profiler had.
    assert question.candidates[0].confidence == profile.inferences["target_column"].confidence


def test_a_confident_answer_raises_nothing(tmp_path: Path) -> None:
    profile = TabularProfiler(ProfilerConfig()).profile_directory(
        build_strong_signals(tmp_path), "strong"
    )

    assert pending_schema_questions(profile) == []


def test_a_question_id_follows_its_candidates_not_its_asking() -> None:
    """Asked twice about the same set is one question; a changed set is a new one."""
    first = question_id("comp", "target_column", ["Depth", "Zone_Depth"])
    again = question_id("comp", "target_column", ["Zone_Depth", "Depth"])
    wider = question_id("comp", "target_column", ["Depth", "Zone_Depth", "Elev"])

    assert first == again, "order of the candidates is not part of the question"
    assert first != wider, "a changed candidate set is a different question"


def test_questions_are_not_stored_anywhere(tmp_path: Path) -> None:
    """Derived on read, so a repaired schema cannot leave a campaign shut.

    AGENTS.md rule 2: `apply_card_to_beliefs` stepped once per card, and
    repairing a card afterwards changed nothing. A question list on disk is that
    shape — it outlives its cause.
    """
    profile = _tie_profile(tmp_path / "before")
    assert pending_schema_questions(profile)

    answered = _tie_profile(tmp_path / "after", answers={"target_column": "Depth"})

    assert pending_schema_questions(answered, {"target_column": "Depth"}) == []
    assert not list(tmp_path.rglob("schema_questions.json"))


# --- answers ----------------------------------------------------------------


def test_an_answer_wins_and_carries_its_evidence(tmp_path: Path) -> None:
    """A person's answer is the strongest signal there is, and still a signal.

    It goes through the same resolver as everything else — so the profile shows
    what the *data* said about the column a human chose, beside the fact that
    they chose it.
    """
    profile = _tie_profile(tmp_path, answers={"target_column": "Depth"})
    target = profile.inferences["target_column"]

    assert profile.target_column == "Depth"
    assert target.band == "asserted"
    assert target.confidence == 1.0
    assert [signal.id for signal in target.signals][0] == "operator_answer"
    # The evidence the data offered is kept beside it, not replaced by the answer.
    assert "present_across_train_units" in {signal.id for signal in target.signals}


def test_an_answer_survives_a_profile_rebuild(tmp_path: Path) -> None:
    """It lives beside `profile.json`, not inside it.

    `profile.json` is rebuilt on every `PROFILE_SCHEMA_VERSION` bump; an answer
    that lived in it would be lost on the next profiler upgrade, which is the
    upgrade most likely to change what the question was.
    """
    record_answer(tmp_path, "target_column", "Depth")

    assert (tmp_path / ANSWERS_FILENAME).is_file()
    assert load_answers(tmp_path) == {"target_column": "Depth"}
    assert json.loads((tmp_path / ANSWERS_FILENAME).read_text()) == {"target_column": "Depth"}


def test_only_a_real_question_can_be_answered(tmp_path: Path) -> None:
    """`research schema answer modality image` is not a thing.

    The answer file is not a place to override the profiler generally: it holds
    answers to questions the profiler actually asked.
    """
    with pytest.raises(ValueError, match="not a schema question"):
        record_answer(tmp_path, "modality", "image")
    assert set(BLOCKING_FIELDS) == {"target_column", "id_columns"}


def test_an_unparseable_answer_file_asks_again(tmp_path: Path) -> None:
    """Neither trusted nor fatal: the question comes back."""
    (tmp_path / ANSWERS_FILENAME).write_text('{"target_column": ', encoding="utf-8")

    assert load_answers(tmp_path) == {}


def test_answering_makes_the_profile_stale(tmp_path: Path) -> None:
    """Otherwise the escape is fiction.

    `_profile_is_current` matches on `schema_version`, so a profile built before
    the answer would be reused forever and `research schema answer` would change
    nothing — the exact defect this milestone is named after, re-made inside its
    own mechanism.
    """
    from labpilot.accessor.profiler.report import write_profile
    from labpilot.research_engine.execution.capabilities.workspace.capability import (
        _profile_state,
    )

    profile = _tie_profile(tmp_path)
    write_profile(tmp_path, profile)
    path = tmp_path / "profile.json"

    assert _profile_state(path, root=tmp_path) == "current"
    record_answer(tmp_path, "target_column", "Depth")
    assert _profile_state(path, root=tmp_path) == "stale"
    assert profile.answers_fingerprint != answers_fingerprint(load_answers(tmp_path))


# --- the campaign ------------------------------------------------------------


def test_an_unanswerable_campaign_stops(tmp_path: Path) -> None:
    """Goal 5: unattended, an open question stops the run before it spends a step.

    Mutation check: remove the block from `run_until_stop` and this campaign
    runs its scripted tools, so the assertion on `decisions` fails.
    """
    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))

    trace = harness.run(policy=["generate_plan", "generate_plan"], max_steps=4)

    assert trace.decisions[-1].observe["stop_reason"] == "schema_question"
    assert trace.calls("generate_plan") == 0, "a campaign must not act on a guessed label"
    assert "research schema answer target_column" in trace.stop_reason
    session = harness.store.get_session(trace.session_id)
    assert session is not None
    assert session.status == "waiting", "resumable the moment someone answers"


def test_an_answered_campaign_runs(tmp_path: Path) -> None:
    """The other half: with a channel to ask on, the campaign asks and continues."""
    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))
    asked: list[str] = []

    def prompt(question):
        asked.append(question.field)
        return "Depth"

    trace = harness.run(policy=["generate_plan", None], max_steps=2, schema_prompt=prompt)

    assert asked == ["target_column"]
    assert "schema_question" not in trace.stop_reason
    assert trace.calls("generate_plan") >= 1
    assert load_answers(harness.workspace.root) == {"target_column": "Depth"}
    assert open_questions(harness.workspace.root) == []


def test_a_refused_answer_still_stops(tmp_path: Path) -> None:
    """A prompt that returns nothing is a person declining to guess, not a default."""
    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))

    trace = harness.run(policy=["generate_plan"], max_steps=2, schema_prompt=lambda question: None)

    assert trace.decisions[-1].observe["stop_reason"] == "schema_question"
    assert trace.calls("generate_plan") == 0
    assert load_answers(harness.workspace.root) == {}
