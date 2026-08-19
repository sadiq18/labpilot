"""Questions the profiler is not entitled to answer for you.

M22 step 4. Two of the five answers are ones where **there is no safe default**:
which column is the label, and which is the key. Everything a campaign does
afterwards is a statement about whichever column was picked, and
``_profile_is_current`` reuses ``profile.json``, so a guess made once is frozen
into every later run of that workspace.

So below the acting threshold the profiler stops answering and asks. Interactive,
it asks; unattended, it **blocks** — and there is deliberately no third option.

**Never route a schema question through ``--yes``, ``maybe_approve`` or
``auto_approve``.** Those ask *"may I do the thing you asked for?"*, where
auto-allow is safe by construction because the default is the answer the
operator would have given. This asks *"which of these facts is true?"*, where a
default is a coin flip that outlives the run that made it.

**Questions are derived, never stored.** A question exists exactly while a
blocking field is uncertain and unanswered; repair the schema and it is gone,
with no file to go stale. That is AGENTS.md rule 2 — recompute, never step —
and it is why there is no ``schema_questions.json``. The one durable file is
``schema_answers.json``, which holds only what a human said.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from labpilot.accessor.common.atomic_write import atomic_write_text
from labpilot.accessor.profiler.evidence import Alternative

if TYPE_CHECKING:
    from labpilot.accessor.profiler.tabular import DatasetProfile

__all__ = [
    "ANSWERS_FILENAME",
    "BLOCKING_FIELDS",
    "SchemaQuestion",
    "answers_fingerprint",
    "load_answers",
    "open_questions",
    "pending_schema_questions",
    "record_answer",
]

#: Where an operator's answers live: beside `profile.json`, never inside it.
#: `profile.json` is rebuilt on every `PROFILE_SCHEMA_VERSION` bump, and an
#: answer has to survive a profiler upgrade — it is the one thing in the
#: workspace the profiler did not produce.
ANSWERS_FILENAME = "schema_answers.json"

#: The fields whose uncertainty stops a campaign.
#:
#: Not every field in the schema, and the difference is the point. A capped
#: `disjoint_units` is a *documented assumption* — actionable, never assertable
#: — and asking about it on every ordinary dataset would train an operator to
#: dismiss the question that matters. A missing metric degrades optimisation.
#: A wrong target or key corrupts everything downstream and does it silently.
BLOCKING_FIELDS = ("target_column", "id_columns")


class SchemaQuestion(BaseModel):
    """One thing the profiler could not settle, with the evidence for each answer."""

    #: sha256 over dataset, field and the *candidate set*. A question is never
    #: re-asked; a changed candidate set is genuinely a different question and
    #: should be.
    id: str
    field: str
    #: What the profiler currently has, if anything. Recorded so an operator can
    #: see what would have been used, not so it can be defaulted to.
    provisional: str | None = None
    candidates: list[Alternative] = Field(default_factory=list)
    #: One line: what the profiler saw.
    context: str = ""


def question_id(dataset: str, field: str, candidates: list[str]) -> str:
    payload = "|".join([dataset, field, *sorted(candidates)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pending_schema_questions(
    profile: DatasetProfile, answers: dict[str, str] | None = None
) -> list[SchemaQuestion]:
    """Every blocking field that is uncertain and unanswered.

    Derived on every call from the schema and the answer file, so a repaired
    profile has no questions and a stale file cannot keep a campaign shut.
    """
    answered = answers or {}
    questions: list[SchemaQuestion] = []
    for field in BLOCKING_FIELDS:
        if field in answered:
            continue
        inference = profile.inferences.get(field)
        if inference is None or inference.band != "uncertain":
            continue
        provisional = (
            profile.target_column
            if field == "target_column"
            else (profile.id_columns[0] if profile.id_columns else None)
        )
        candidates = list(inference.alternatives)
        names = [alternative.candidate for alternative in candidates]
        if provisional is not None:
            names.append(provisional)
        questions.append(
            SchemaQuestion(
                id=question_id(profile.competition, field, names),
                field=field,
                provisional=provisional,
                candidates=candidates,
                context=_context_for(profile, field, inference.confidence),
            )
        )
    return questions


def _context_for(profile: DatasetProfile, field: str, confidence: float) -> str:
    if field == "target_column":
        return (
            f"{len(profile.train_only_columns)} column(s) are withheld at scoring; "
            f"the best-evidenced one scores {confidence:.2f}"
        )
    return f"the key was inferred at {confidence:.2f} confidence"


def answers_fingerprint(answers: dict[str, str]) -> str:
    """A stamp for the answers a profile was built with.

    Without it, answering a question changes nothing: `_profile_is_current`
    matches on `schema_version`, so the profile built from the *old* answers is
    reused forever and the escape the operator was offered is fiction — the
    defect this milestone is named after, re-made in its own mechanism.
    """
    if not answers:
        return ""
    payload = json.dumps(answers, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def open_questions(root: Path) -> list[SchemaQuestion]:
    """The questions a workspace has open, or none.

    A missing or unreadable profile has no questions — a campaign in that state
    has a different problem, and reporting a schema question would send the
    operator to the wrong place.
    """
    from labpilot.accessor.profiler.tabular import DatasetProfile

    path = Path(root) / "profile.json"
    try:
        profile = DatasetProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []
    return pending_schema_questions(profile, load_answers(root))


def answers_path(root: Path) -> Path:
    return Path(root) / ANSWERS_FILENAME


def load_answers(root: Path) -> dict[str, str]:
    """What a human has already settled, or nothing.

    Unreadable is the same as absent here on purpose: an answer file nobody can
    parse must not stop a campaign *and* must not be silently trusted, so the
    questions come back and the operator answers again.
    """
    path = answers_path(root)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {
        str(field): str(value)
        for field, value in stored.items()
        if isinstance(field, str) and value is not None
    }


def record_answer(root: Path, field: str, value: str) -> dict[str, str]:
    """Persist one answer and return the full set.

    Written whole rather than appended, and atomically, so a crash mid-write
    leaves the previous answers rather than half of them.
    """
    if field not in BLOCKING_FIELDS:
        raise ValueError(f"{field!r} is not a schema question; expected one of {BLOCKING_FIELDS}")
    answers = load_answers(root)
    answers[field] = value
    atomic_write_text(answers_path(root), json.dumps(answers, indent=2, sort_keys=True) + "\n")
    return answers
