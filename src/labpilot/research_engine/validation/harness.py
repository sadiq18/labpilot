"""A validator for objectives that are not competitions.

The second implementation of `HypothesisValidator`, and the point of M12: it
exists to prove the seam carries something that is not Kaggle. Every assumption
the Kaggle path makes is absent here.

* **No folds.** `_primary_cv_keyed` never runs, and no `cv_` key appears
  anywhere. The score is whatever the harness computed.
* **No `cv_std`.** Stability comes back `UNKNOWN` rather than fabricated, and
  `_decide` has to reach a verdict on one number.
* **No submission and no leaderboard.** `secondary` is `None`, so the
  leaderboard branch of `_decide` is never taken.
* **No `competition.json`.** There is no file anywhere that could answer "which
  way is better", which is the whole reason this validator was chosen over a
  local-dataset one for the second implementation: **direction is stated by the
  thing that computed the score**, and it cannot be recovered from a contract
  even in principle.

`result.json` is the entire contract:

    {"score": 0.82, "metric": "pass_rate", "direction": "maximize"}

**This reads the result; it does not run the harness.** The design sketched
"runs a script and reads a result.json", and the symmetry with `KaggleCvValidator`
is worth more: that one does not train either — training is a task the execution
machinery already owns, and `validate` reads what it wrote. Keeping the split
means a harness is scheduled, retried, budgeted and sandboxed by the same code
as everything else, instead of a subprocess spawned from inside a comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
    _slug,
    normalize_direction,
)
from labpilot.research_engine.validation.models import ValidationResult

logger = logging.getLogger(__name__)

SOURCE = "harness"

#: What a harness *produced*. Distinct from `metrics.json` so a workspace can be
#: read without a flag to set or anything to keep in sync.
RESULT_FILE = "result.json"

#: What a harness *promises*, declared before anything runs. Two files rather
#: than one because they answer at different times: a campaign has to know its
#: objective at launch, and `result.json` does not exist until a run has already
#: happened. Selecting on the result alone meant a fresh harness workspace was
#: read as a Kaggle one until its first comparison.
OBJECTIVE_FILE = "harness.json"


def handles(workspace_root: Path) -> bool:
    """Whether this workspace is a harness workspace.

    Keyed on the *declaration*, not the result, so the answer is the same before
    and after a run. Deliberately exclusive of `metrics.json`: a workspace
    holding both is ambiguous, and guessing which objective a campaign is judged
    on is the failure this milestone exists to remove — so the established path
    wins rather than the newer one.
    """
    root = Path(workspace_root)
    return (root / OBJECTIVE_FILE).is_file() and not (root / "metrics.json").is_file()


def stated_objective(workspace_root: Path) -> tuple[str | None, str | None]:
    """`(metric, direction)` a harness promises, for the launch preflight.

    The campaign gate refuses to start when it cannot justify the objective, and
    it could only ever read a `competition.json` — so a benchmark workspace was
    told to *"set evaluation_metric in competition.json"*, which is advice from
    another domain for a file it will never have. That is exit criterion 3
    failing: the same `research conduct` phrasing did not work across domains.

    Returned raw. `resolve_objective` decides what a stated direction is worth,
    and it is the same layer that catches a harness whose declaration and result
    disagree — the contradiction case, which blocks rather than picking a side.
    """
    path = Path(workspace_root) / OBJECTIVE_FILE
    body = _load(path)
    if not isinstance(body, dict):
        return None, None
    metric = str(body.get("metric") or "") or None
    direction = normalize_direction(body.get("direction"))
    return metric, None if direction == "unknown" else direction


def result_from_payload(
    payload: Any, *, artifacts: dict[str, str] | None = None
) -> ValidationResult:
    """Read one `result.json` body into a `ValidationResult`.

    Every field is optional in the sense that a missing or malformed one yields
    `None` and a line of provenance rather than an exception. A harness is
    somebody else's script; it will get this wrong, and the useful response is a
    result that says what was wrong, not a traceback inside a comparison.

    The three refusals below each map onto a guard the evidence layer already
    has, so a bad `result.json` is stopped by machinery that already exists:

    * no score      -> `_found` returns None -> `missing_control` / no `cv_gain`
    * no metric     -> `_same_metric` refuses; two unnamed scores are not one metric
    * no direction  -> `build_evidence_card` raises rather than guessing a sign
    """
    provenance: list[str] = []
    body = payload if isinstance(payload, dict) else {}
    if not isinstance(payload, dict):
        provenance.append(f"{RESULT_FILE} is not a JSON object; nothing could be read from it")

    raw_score = body.get("score")
    score: float | None = None
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        # `bool` is an `int` in Python, and `{"score": true}` is a harness bug
        # worth naming rather than scoring as 1.0.
        if "score" in body:
            provenance.append(f"score {raw_score!r} is not a number")
        else:
            provenance.append("no score reported")
    else:
        score = float(raw_score)
        provenance.append(f"score {score} read from {RESULT_FILE}")

    metric = _slug(str(body.get("metric") or ""))
    if metric:
        provenance.append(f"measured on {metric!r}, as stated by the harness")
    else:
        provenance.append("the harness did not name the metric it measured")

    direction = normalize_direction(body.get("direction"))
    if direction == "unknown":
        provenance.append(
            f"direction {body.get('direction')!r} is not one the harness could state"
        )
    else:
        provenance.append(f"direction {direction}, stated by the harness that computed the score")

    return ValidationResult(
        score=score,
        metric=metric,
        direction=None if direction == "unknown" else direction,
        source=SOURCE,
        provenance=provenance,
        artifacts=dict(artifacts or {}),
        raw=dict(body),
        # No leaderboard, no held-out set: one number is all a harness reports.
        secondary=None,
    )


class HarnessValidator:
    """Scores a hypothesis from whatever the workspace's harness wrote.

    Implements `HypothesisValidator`. The second implementation, and the one
    that makes the protocol's cost visible: it needs no `competition.json`, no
    knowledge directory and no metric registry, so anything it cannot do is
    something the seam genuinely requires rather than something Kaggle happened
    to provide.
    """

    source = SOURCE

    def validate(
        self, hypothesis_id: str | None, workspace: Any, context: Any
    ) -> ValidationResult:
        root = _workspace_root(workspace)
        path = root / RESULT_FILE
        return result_from_payload(_load(path), artifacts={"result": str(path)})


def _workspace_root(workspace: Any) -> Path:
    """Same rule as the Kaggle validator: ask the type, do not probe for `.root`.

    `pathlib.Path` has a `root` property returning ``"/"``, so
    `getattr(workspace, "root", workspace)` silently reads the filesystem root
    when handed the obvious thing.
    """
    if isinstance(workspace, str | Path):
        return Path(workspace)
    return Path(workspace.root)


def _load(path: Path) -> Any:
    """The file's body, or `None` when it cannot be read.

    Unreadable and malformed are the same answer here — "the harness told us
    nothing" — because the distinction changes nothing a caller can act on, and
    `result_from_payload` turns either into a result that refuses rather than a
    raise inside a comparison.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.info("%s unreadable at %s: %s", RESULT_FILE, path, exc)
        return None
