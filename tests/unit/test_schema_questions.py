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

import pandas as pd
import pytest
from helpers.campaign_harness import CampaignHarness, ok
from helpers.dataset_shapes import build_partitioned_without_template, build_strong_signals
from helpers.dataset_sources import DictSource

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
    # Both candidates, each with its own score and signals — including the one
    # the profiler would have used. Listing the provisional by name alone left
    # the operator comparing a number against a bare string, and *are these two
    # equally supported* is the entire question on a tie.
    assert [c.candidate for c in question.candidates] == ["Zone_Depth", "Depth"]
    assert question.candidates[0].confidence == question.candidates[1].confidence
    assert {s.id for s in question.candidates[0].signals} == {
        s.id for s in question.candidates[1].signals
    }


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

    `_profile_state` matches on `schema_version` alone, so a profile built before
    the answer would be served forever and `research schema answer` would change
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


def test_an_answered_campaign_stops_until_the_profile_is_rebuilt(tmp_path: Path) -> None:
    """An answer must reach the *value*, not only the answers file.

    The first version of this test asserted the campaign continued, and passed
    while `profile.json` still named the column the operator had just rejected —
    with the question now closed, so nothing would ever ask again. Recording an
    answer supersedes the description the campaign is holding, and
    `prepare_workspace` is a plan task this loop cannot dispatch, so the honest
    move is to stop and let the next run re-derive.

    The last two assertions are the ones that would have caught it: the profile
    is stale, and re-deriving with the answers on file produces the answered
    target.
    """
    from labpilot.accessor.profiler.tabular import PROFILE_SCHEMA_VERSION
    from labpilot.research_engine.execution.capabilities.workspace.capability import (
        _profile_state,
    )

    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))
    root = harness.workspace.root
    asked: list[str] = []

    def prompt(question):
        asked.append(question.field)
        return "Depth"

    trace = harness.run(policy=["generate_plan", None], max_steps=2, schema_prompt=prompt)

    assert asked == ["target_column"]
    assert trace.decisions[-1].observe["answered"] is True
    assert "answer recorded" in trace.stop_reason
    assert trace.calls("generate_plan") == 0, "not one step on the rejected column"
    assert load_answers(root) == {"target_column": "Depth"}
    assert open_questions(root) == []

    # The description the campaign held is superseded, and re-deriving with the
    # answer on file gives the column the operator chose.
    assert json.loads((root / "profile.json").read_text())["schema_version"] == (
        PROFILE_SCHEMA_VERSION
    )
    assert _profile_state(root / "profile.json", root) == "stale"
    rebuilt = _tie_profile(tmp_path / "rebuild", answers=load_answers(root))
    assert rebuilt.target_column == "Depth"
    assert rebuilt.inferences["target_column"].band == "asserted"


def test_a_refused_answer_still_stops(tmp_path: Path) -> None:
    """A prompt that returns nothing is a person declining to guess, not a default."""
    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))

    trace = harness.run(policy=["generate_plan"], max_steps=2, schema_prompt=lambda question: None)

    assert trace.decisions[-1].observe["stop_reason"] == "schema_question"
    assert trace.calls("generate_plan") == 0
    assert load_answers(harness.workspace.root) == {}


def test_an_answer_that_names_no_column_is_refused(tmp_path: Path) -> None:
    """`operator_answer` is worth 1.00, so an unchecked value is a confident lie.

    Before the check, `answers={"target_column": "Dpeth"}` produced
    `target_column: "Dpeth"` at confidence **1.0**, `asserted`, for a name in no
    table — and took the `equals_target` exclusion with it, because nothing
    equals a target that does not exist, so the leak column returned to
    `feature_columns`.
    """
    data_dir = build_partitioned_without_template(tmp_path)
    source = LocalFileSource(data_dir, DeclaredFacts(answers={"target_column": "Dpeth"}))

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "typo")
    target = profile.inferences["target_column"]

    assert profile.target_column != "Dpeth"
    assert target.band == "uncertain", "the question stays open, so it is asked again"
    assert [claim.claim for claim in target.rejected] == ["Dpeth"]
    assert any(note.code == "answer_refused" for note in profile.notes)


def test_the_cli_refuses_a_column_that_does_not_exist(tmp_path: Path) -> None:
    """The same check where the operator meets it, with the columns listed."""
    from labpilot.accessor.profiler.report import write_profile

    write_profile(tmp_path, _tie_profile(tmp_path / "data"))

    with pytest.raises(ValueError, match="names no column"):
        record_answer(tmp_path, "target_column", "Dpeth")
    assert load_answers(tmp_path) == {}, "a refused answer is not written"


def test_a_composite_key_can_be_answered(tmp_path: Path) -> None:
    """`(store_id, date)` is an ordinary key outside Kaggle.

    A scalar answer would settle a two-column key as half of itself and close
    the question, which is worse than leaving it open.
    """
    from labpilot.accessor.profiler.questions import parse_answer

    assert parse_answer("id_columns", "store_id, date") == ["store_id", "date"]
    with pytest.raises(ValueError, match="takes one column"):
        parse_answer("target_column", "a,b")

    frame = pd.DataFrame({"store_id": [1, 2, 3], "date": ["a", "b", "c"], "sales": [1.0, 2.0, 3.0]})
    source = DictSource(
        {
            "train.csv": frame,
            "test.csv": frame[["store_id", "date"]],
            "sample_submission.csv": frame[["store_id", "sales"]],
        },
        DeclaredFacts(answers={"id_columns": "store_id,date"}),
    )

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "composite")

    assert profile.id_columns == ["store_id", "date"]
    assert profile.id_column == "store_id", "the singular view is the first of them"
    assert profile.inferences["id_columns"].band == "asserted"


def test_a_write_failure_blocks_rather_than_crashes(tmp_path: Path, monkeypatch) -> None:
    """A read-only workspace stops the campaign; it does not end the process.

    `record_answer` used to let `OSError` escape `run_until_stop` entirely — no
    decision record, no session status, just a traceback — where every other
    operator-facing prompt in that loop ends in a stop.
    """
    from labpilot.research_engine.conductor import loop as loop_module

    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok()]})
    harness.seed_profile(_tie_profile(tmp_path / "data"))

    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(loop_module, "record_answer", refuse)
    trace = harness.run(
        policy=["generate_plan"], max_steps=2, schema_prompt=lambda question: "Depth"
    )

    assert trace.decisions[-1].observe["stop_reason"] == "schema_question"
    assert trace.decisions[-1].observe["answered"] is False
    assert trace.calls("generate_plan") == 0


def test_the_profile_is_not_reparsed_every_step(tmp_path: Path, monkeypatch) -> None:
    """Two `stat` calls a step, not a parse of a 14 KB profile.

    `open_questions` validates the whole profile — every column, the evidence
    plane, up to 200 paths — and M17 lets this loop run unbounded.
    """
    from labpilot.research_engine.conductor import loop as loop_module

    harness = CampaignHarness(tmp_path, tools={"generate_plan": [ok(), ok(), ok()]})
    harness.seed_profile(
        TabularProfiler(ProfilerConfig()).profile_directory(
            build_strong_signals(tmp_path / "data"), "strong"
        )
    )
    reads: list[Path] = []
    real = loop_module.open_questions

    def counted(root):
        reads.append(root)
        return real(root)

    monkeypatch.setattr(loop_module, "open_questions", counted)
    trace = harness.run(policy=["generate_plan"] * 3, max_steps=3)

    assert trace.calls("generate_plan") == 3, "the campaign ran three steps"
    assert len(reads) == 1, f"profile parsed {len(reads)} times for 3 steps"
