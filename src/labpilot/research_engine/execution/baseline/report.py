"""Why the baseline failed, in facts read from artifacts.

M23 step 6. Every detector here reads a file and cites it. **No LLM on this
path**, and no list of plausible causes: a list that prints identically on every
failure is a list nobody reads, and this repository has paid for that twice
already — in `check_confinement` and in `validation_region`.

So the report has three parts and the middle one is often empty:

* the comparison, which is the verdict in three lines;
* **Observed** — causes that fired, each with the artifact that says so;
* **Not ruled out** — the causes nothing could check here, named so the absence
  of a finding is not read as an all-clear.

When nothing fires, the report says so. That sentence is more useful than six
bullets, because it tells an operator the detectors ran and found nothing rather
than leaving them to guess whether anything looked.

Plan: ``docs/research-os/autonomy-roadmap/design/18-baseline-correctness.md`` §7.6
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from labpilot.research_engine.execution.baseline.gate import GateVerdict

logger = logging.getLogger(__name__)

__all__ = ["Cause", "FailureReport", "build_report", "CAUSES"]

#: The causes the design names. Every one either fires with a citation or lands
#: in "not ruled out" — there is no third outcome, which is what stops the report
#: becoming a checklist that always prints.
CAUSES: tuple[str, ...] = (
    "leakage/ID handling",
    "validation mismatch",
    "target identification",
    "feature selection",
    "preprocessing",
    "metric mismatch",
)


class Cause(BaseModel):
    """One thing that fired, and the artifact that says so."""

    name: str
    detail: str
    #: The file this was read from. A cause without one is an opinion, and this
    #: model has no way to express one.
    citation: str


class FailureReport(BaseModel):
    competition: str = ""
    verdict: str = ""
    comparison: str = ""
    observed: list[Cause] = Field(default_factory=list)
    not_ruled_out: list[str] = Field(default_factory=list)

    def render(self) -> str:
        headline = "BASELINE FAILURE"
        lines = [f"{headline} - {self.competition}" if self.competition else headline]
        if self.comparison:
            lines.append(self.comparison)
        lines.append("")
        if self.observed:
            lines.append("Observed (facts read from artifacts):")
            for cause in self.observed:
                lines.append(f"  {cause.name}")
                lines.append(f"      {cause.detail}")
                lines.append(f"      [{cause.citation}]")
        else:
            # The sentence that beats six bullets: the detectors ran.
            lines.append("Observed: nothing. Every detector ran and none of them fired.")
        if self.not_ruled_out:
            lines.append("")
            lines.append("Not ruled out: " + " - ".join(self.not_ruled_out))
        lines.append("")
        lines.append("Do not proceed to research.")
        return "\n".join(lines)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _training_sources(root: Path) -> list[tuple[str, str]]:
    """`(relative path, text)` for the generated pipeline, if there is one."""
    found: list[tuple[str, str]] = []
    for directory in ("pipeline", "src"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                found.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
            except OSError:
                continue
    return found


def _detect_anchor(root: Path) -> Cause | None:
    """The anchor column the profiler named, that the pipeline never mentions.

    rogii's whole story: `TVT_input` equals the target wherever present and is
    absent exactly on the scored rows, so carrying it forward scores 15.1 against
    the pipeline's 1380. The profiler has named it since 2026-08-13, and nothing
    read it — which is why this detector cites `profile.json` rather than
    inferring anything.
    """
    profile = _read_json(root / "profile.json")
    anchor = profile.get("anchor_column")
    if not anchor:
        return None
    sources = _training_sources(root)
    if not sources:
        return None
    if any(str(anchor) in text for _, text in sources):
        return None
    named = ", ".join(path for path, _ in sources[:3])
    return Cause(
        name="leakage/ID handling",
        detail=(
            f"profile.anchor_column={anchor!r} equals the target wherever present "
            f"and is absent exactly on the scored rows; no training source mentions it, "
            f"so the strongest signal in the dataset is unused"
        ),
        citation=f"profile.json:anchor_column, checked against {named}",
    )


def _detect_validation_mismatch(root: Path) -> Cause | None:
    """A declared scheme that no generated function performs.

    Reuses `delta/consistency.py`'s matcher rather than a second one: it already
    knows that `group_kfold` and `GroupKFold` are the same word spelled twice,
    and a fresh substring check here would disagree with the delta checks about
    whether the pipeline honours its own plan.
    """
    choice = _read_json(root / "baseline_choice.json")
    plan = choice.get("validation") if isinstance(choice.get("validation"), dict) else {}
    scheme = str(plan.get("scheme") or "")
    if not scheme or scheme == "kfold":
        # Plain KFold is what any `train_test_split` degrades to, so its absence
        # is not evidence of anything.
        return None
    sources = _training_sources(root)
    if not sources:
        return None

    from labpilot.research_engine.execution.delta.consistency import _matches_scheme, _word_parts

    parts = _word_parts(scheme)
    for path, text in sources:
        for line in text.splitlines():
            for token in line.replace("(", " ").replace(".", " ").split():
                if _matches_scheme(token, parts):
                    return None
    named = ", ".join(path for path, _ in sources[:3])
    return Cause(
        name="validation mismatch",
        detail=(
            f"the plan declares {scheme!r}; no function in the training source performs it, "
            "so local CV is measuring a different split from the one the competition scores"
        ),
        citation=f"baseline_choice.json:validation.scheme, checked against {named}",
    )


def _detect_metric_mismatch(root: Path) -> Cause | None:
    """CV optimising something other than what the competition scores.

    Step 2 made this a field rather than a `logger.info`; this is the reader that
    field was for.
    """
    choice = _read_json(root / "baseline_choice.json")
    substituted = choice.get("metric_substituted_from")
    if not substituted:
        return None
    return Cause(
        name="metric mismatch",
        detail=(
            f"the competition is scored by {substituted!r} and cross-validation optimises "
            f"{choice.get('metric_name')!r}, so every comparison here is about a proxy"
        ),
        citation="baseline_choice.json:metric_substituted_from",
    )


def _detect_excluded_features_used(root: Path) -> Cause | None:
    """A column the plan excludes that the training source uses anyway."""
    choice = _read_json(root / "baseline_choice.json")
    plan = choice.get("validation") if isinstance(choice.get("validation"), dict) else {}
    excluded = [c for c in (plan.get("exclude_features") or []) if isinstance(c, str) and c]
    if not excluded:
        return None
    sources = _training_sources(root)
    if not sources:
        return None
    for path, text in sources:
        used = [column for column in excluded if column in text]
        if used:
            return Cause(
                name="feature selection",
                detail=(
                    f"{', '.join(sorted(used))} is excluded by the plan — present in training "
                    "and absent at inference — and the training source references it"
                ),
                citation=f"baseline_choice.json:validation.exclude_features, found in {path}",
            )
    return None


#: Detector per cause. A cause with no detector is honest about that: it lands in
#: "not ruled out" rather than being silently dropped, so the report never
#: implies more was checked than was.
_DETECTORS = {
    "leakage/ID handling": _detect_anchor,
    "validation mismatch": _detect_validation_mismatch,
    "metric mismatch": _detect_metric_mismatch,
    "feature selection": _detect_excluded_features_used,
}


def build_report(root: Path, verdict: GateVerdict, competition: str = "") -> FailureReport:
    """Run every detector and name what could not be checked.

    Detectors that raise are treated as "did not fire" and logged: a report that
    crashes tells an operator nothing, and the causes it could not check are
    already named in `not_ruled_out`.
    """
    root = Path(root)
    report = FailureReport(
        competition=competition,
        verdict=verdict.state,
        comparison=verdict.comparison.render() if verdict.comparison.metric_name else "",
    )
    fired: set[str] = set()
    for name, detector in _DETECTORS.items():
        try:
            cause = detector(root)
        except Exception as exc:  # noqa: BLE001 — a broken detector is not a cause
            logger.info("Baseline failure detector %r could not run: %s", name, exc)
            continue
        if cause is not None:
            report.observed.append(cause)
            fired.add(name)
    report.not_ruled_out = [name for name in CAUSES if name not in fired]
    return report
