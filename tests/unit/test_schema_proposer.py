"""The model proposes; the data vetoes.

M22 step 6, and goal 2 of the design: the value plane is byte-identical whether
the proposer is absent, right, or wrong about everything. That is the test that
makes "propose-only" a mechanism rather than a comment, and it is the reason the
proposer may add a signal or an alternative and may never assign a value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers.dataset_shapes import build_no_kaggle_inputs, build_strong_signals

from labpilot.accessor.profiler.proposer import (
    SchemaProposal,
    SchemaProposalAgent,
    apply_proposal,
)
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig


class _Says:
    """A model that answers with whatever it was constructed with."""

    def __init__(self, **payload: object) -> None:
        self._payload = {"target_column": None, "id_columns": [], "reasoning": "", **payload}
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        self.prompts.append(user)
        return json.dumps(self._payload)


class _Explodes:
    def complete(self, system: str, user: str, **kwargs: object) -> str:
        raise RuntimeError("provider down")


def _profile(data_dir: Path, client: object | None = None) -> DatasetProfile:
    config = ProfilerConfig(llm_proposals=client is not None)
    return TabularProfiler(config).profile_directory(data_dir, "hp", llm_client=client)


def _values(profile: DatasetProfile) -> dict:
    """The value plane: everything except the evidence and the prose over it."""
    dumped = json.loads(profile.model_dump_json())
    for derived in ("inferences", "notes", "warnings"):
        dumped.pop(derived, None)
    return dumped


# --- goal 2 ------------------------------------------------------------------


def test_the_value_plane_ignores_the_model(tmp_path: Path) -> None:
    """Absent, right, and wrong about every field: one answer, byte for byte.

    A model asked "which column is the target?" always names one. This is the
    property that makes it safe to ask.
    """
    absent = _values(_profile(build_strong_signals(tmp_path / "a")))
    agrees = _values(
        _profile(
            build_strong_signals(tmp_path / "b"),
            _Says(target_column="SalePrice", id_columns=["Id"]),
        )
    )
    wrong = _values(
        _profile(
            build_strong_signals(tmp_path / "c"),
            _Says(target_column="LotArea", id_columns=["Neighborhood"]),
        )
    )
    invented = _values(
        _profile(build_strong_signals(tmp_path / "d"), _Says(target_column="Nonexistent"))
    )

    assert absent["target_column"] == "SalePrice", "an empty comparison would prove nothing"
    for other in (agrees, wrong, invented):
        assert other == absent


def test_a_failing_proposer_leaves_no_mark_on_the_answer(tmp_path: Path) -> None:
    """A provider that is down must not fail a profile, or change one."""
    absent = _values(_profile(build_strong_signals(tmp_path / "a")))

    profile = _profile(build_strong_signals(tmp_path / "b"), _Explodes())

    assert _values(profile) == absent
    assert any(note.code == "llm_proposal_unavailable" for note in profile.notes)


def test_the_proposer_is_off_unless_asked(tmp_path: Path) -> None:
    """The flag, not the client, is what turns it on."""
    client = _Says(target_column="SalePrice", id_columns=["Id"])

    default = TabularProfiler(ProfilerConfig()).profile_directory(
        build_strong_signals(tmp_path), "hp", llm_client=client
    )

    assert client.prompts == [], "a configured client is not a request to use one"
    assert "llm_proposal_confirmed" not in {
        signal.id for signal in default.inferences["target_column"].signals
    }


# --- the three outcomes ------------------------------------------------------


def test_agreement_is_worth_ten_points_and_no_more(tmp_path: Path) -> None:
    """However emphatic the model, one signal, once."""
    alone = _profile(build_strong_signals(tmp_path / "a"))
    with_model = _profile(
        build_strong_signals(tmp_path / "b"),
        _Says(target_column="SalePrice", id_columns=["Id"], reasoning="certain, obviously"),
    )
    signals = with_model.inferences["target_column"].signals

    assert [s.id for s in signals].count("llm_proposal_confirmed") == 1
    assert with_model.confidence_in("target_column") > alone.confidence_in("target_column")
    # Noisy-OR over one extra 0.10 signal: 1 - (1 - 0.9592) * 0.90.
    assert with_model.confidence_in("target_column") == pytest.approx(0.9633, abs=1e-4)


def test_a_contradicted_claim_is_kept_not_dropped(tmp_path: Path) -> None:
    """A verifier refusing a claim is evidence about the *source*.

    `LotArea` is present in the scoring input, so it cannot be withheld and
    cannot be the label. The claim is recorded with what refuted it, because a
    workspace that silently discards them cannot say why it disbelieved one.
    """
    profile = _profile(build_strong_signals(tmp_path), _Says(target_column="LotArea"))
    rejected = profile.inferences["target_column"].rejected

    assert [claim.claim for claim in rejected] == ["LotArea"]
    assert rejected[0].source == "llm"
    assert "present in the scoring input" in rejected[0].refuted_by
    assert profile.target_column == "SalePrice"


def test_an_invented_column_is_refused(tmp_path: Path) -> None:
    """Every claim faces `column_exists` first, so nothing arrives unchecked."""
    profile = _profile(build_strong_signals(tmp_path), _Says(target_column="Nonexistent"))

    assert profile.inferences["target_column"].rejected[0].refuted_by.startswith("names no column")


def test_a_nomination_reaches_the_question_not_the_answer(tmp_path: Path) -> None:
    """Where nothing could be resolved, the model's guess becomes a candidate.

    Case C — one table, no split, no template — is exactly where a reader of the
    description knows something the files cannot say. It arrives as an
    alternative worth 0.10: visible to whoever answers the question, and far
    below the band that would let it act as the answer.
    """
    profile = _profile(
        build_no_kaggle_inputs(tmp_path),
        _Says(target_column="churned", id_columns=["event_id"]),
    )
    target = profile.inferences["target_column"]

    assert profile.target_column is None, "a nomination is not an answer"
    assert [(a.candidate, a.confidence) for a in target.alternatives] == [("churned", 0.1)]
    assert target.band == "uncertain", "so the campaign still asks"
    assert profile.id_columns == []
    assert [a.candidate for a in profile.inferences["id_columns"].alternatives] == ["event_id"]


# --- what the model is allowed to see ---------------------------------------


def test_the_prompt_withholds_our_answer(tmp_path: Path) -> None:
    """Agreement is only evidence if it was reached independently.

    A model shown `target_column: SalePrice` agrees with it, and that agreement
    is worth nothing. The prompt carries the description, the goal and the
    column *names* — not the answer, not the candidates, not a confidence.
    """
    client = _Says(target_column="SalePrice")

    _profile(build_strong_signals(tmp_path), client)

    assert client.prompts, "the proposer never ran"
    prompt = client.prompts[0]
    assert "SalePrice" in prompt, "it needs the column names to name one"
    assert "target_column" not in prompt
    assert "confidence" not in prompt
    assert "sole_withheld_column" not in prompt


def test_a_proposal_names_columns_and_nothing_else() -> None:
    """The agent's contract, checked without a model.

    `apply_proposal` is a pure function of a profile and a proposal, so the
    outcomes above are testable without a provider at all — which is what makes
    them cheap enough to assert on every shape.
    """
    profile = DatasetProfile(competition="x")

    apply_proposal(profile, SchemaProposal(target_column="anything"))

    assert profile.target_column is None
    assert profile.inferences == {}, "a profile with no inferences has nothing to fold into"
    assert SchemaProposalAgent.output_model is SchemaProposal
