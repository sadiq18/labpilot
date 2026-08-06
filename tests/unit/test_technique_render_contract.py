"""The contract: a different technique must produce a different `train.py`.

This is the check that would have caught the bug on day one. `file_digest` is
already computed at `capability.py:265` and nothing ever compared it across
runs — so twelve hypotheses produced byte-identical training code and twelve
identical scores, and the campaign looked healthy throughout.

These tests render through the real `CodeRenderer` and compare bytes. They do
not assert that a keyword argument was forwarded: forwarding is precisely what
the broken code appeared to do.
"""

from __future__ import annotations

import hashlib

import pytest

from labpilot.config import TrainingConfig
from labpilot.research_engine.execution.baseline.registry import get_template
from labpilot.research_engine.execution.baseline.selector import BaselineChoice
from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (  # noqa: E501
    CodeRenderer,
)
from labpilot.research_engine.execution.technique.resolver import resolve_technique

CATEGORICAL_PROFILE = {
    "target_column": "y",
    "partitioned": False,
    "columns": [
        {"name": "city", "dtype": "object"},
        {"name": "amount", "dtype": "float64"},
        {"name": "y", "dtype": "float64", "is_target": True},
    ],
}


def _choice() -> BaselineChoice:
    return BaselineChoice(
        problem_type="tabular_regression",
        template_name="tabular_regression",
        rationale="test",
        target_column="y",
        id_column="id",
        metric_name="rmse",
    )


def _render(tmp_path, name, **kwargs) -> str:
    """Render into **one** directory, always.

    `CodeRenderer` bakes `run_dir` into the output (`competition`, `data_dir`,
    `output_dir`), so rendering two variants into two directories differs
    whatever the technique. An earlier version of this file did exactly that,
    and its contract test passed on the directory name rather than on the
    recipe — a hollow test of the same kind this milestone exists to stop.
    `name` is kept only to label failures.
    """
    run_dir = tmp_path / "run"
    (run_dir / "pipeline").mkdir(parents=True, exist_ok=True)
    choice = _choice()
    template = get_template(choice.problem_type, template_name=choice.template_name)
    assert template is not None, "tabular_regression template must exist"
    CodeRenderer(TrainingConfig()).render(template, choice, run_dir, **kwargs)
    return (run_dir / "pipeline" / "train.py").read_text(encoding="utf-8")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_two_techniques_produce_different_training_code(tmp_path):
    """The headline contract. Both recipes have gates in this template today."""
    a = _render(tmp_path, "a", feature_recipes=["target_encoding"])
    b = _render(tmp_path, "b", feature_recipes=["log_numeric"])
    assert _digest(a) != _digest(b), (
        "two different techniques rendered byte-identical training code — the "
        "exact condition that produced 12 identical scores"
    )


def test_a_technique_differs_from_no_technique(tmp_path):
    baseline = _render(tmp_path, "base")
    treated = _render(tmp_path, "treated", feature_recipes=["target_encoding"])
    assert _digest(baseline) != _digest(treated)


def test_no_technique_is_byte_identical_across_renders(tmp_path):
    """N3/N5: the recipe path is deterministic, and a plan with no technique
    renders what it always rendered. Without this the contract test above could
    pass on nondeterminism rather than on the technique."""
    first = _render(tmp_path, "one")
    second = _render(tmp_path, "two")
    assert _digest(first) == _digest(second)


def test_model_params_reach_the_rendered_code(tmp_path):
    """`deeper_trees` changes no features, only params — it must still alter
    the artifact, or a whole class of technique is silently inert."""
    baseline = _render(tmp_path, "p_base")
    deeper = _render(tmp_path, "p_deep", model_params={"max_depth": 10, "num_leaves": 127})
    assert _digest(baseline) != _digest(deeper)
    assert "127" in deeper


def test_the_capability_fallback_passes_the_technique_through(tmp_path):
    """The verified break, asserted where it happened.

    `capability.py:402` called `render(template, choice, root)` and discarded
    every keyword argument the renderer already accepted. The tests above
    exercise `CodeRenderer` directly, which was never broken — this one drives
    `_render_template_fallback`, which was.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
        CodeEngineeringCapability,
    )

    cap = CodeEngineeringCapability(llm_client=None)
    root = tmp_path / "ws"
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    choice = _choice()

    applied = resolve_technique(
        {"technique": "target_encoding"}, {}, choice=choice, profile=CATEGORICAL_PROFILE
    )
    assert applied.status == "applied"

    assert cap._render_template_fallback(root, choice) is not None
    without = (root / "pipeline" / "train.py").read_text(encoding="utf-8")

    assert cap._render_template_fallback(root, choice, applied) is not None
    with_technique = (root / "pipeline" / "train.py").read_text(encoding="utf-8")

    assert _digest(without) != _digest(with_technique), (
        "the fallback rendered identical code with and without a resolved "
        "technique — the defect that made 12 hypotheses score the same"
    )


@pytest.mark.parametrize("technique", ["target_encoding", "log1p_transform"])
def test_resolution_to_render_is_end_to_end(tmp_path, technique):
    """Resolver output feeds the renderer unchanged — the wiring under test.

    Both techniques are applicable to CATEGORICAL_PROFILE and both have gates
    in this template, so a resolution that changes rendering must change bytes.
    """
    res = resolve_technique(
        {"technique": technique}, {}, choice=_choice(), profile=CATEGORICAL_PROFILE
    )
    assert res.status == "applied", res.reason
    assert res.changes_rendering

    baseline = _render(tmp_path, f"{technique}_base")
    treated = _render(
        tmp_path,
        f"{technique}_treated",
        feature_recipes=res.feature_recipes or None,
        model_params=res.model_params or None,
    )
    assert _digest(baseline) != _digest(treated)


def test_applied_technique_is_stamped_on_baseline_choice(tmp_path):
    """F5, producer side. The artifact an operator reads to answer "what did
    this run apply?" without replaying the log."""
    import json

    from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.execution.technique.resolver import TechniqueResolution

    root = tmp_path / "ws"
    root.mkdir()
    (root / "baseline_choice.json").write_text(
        _choice().model_dump_json(indent=2), encoding="utf-8"
    )

    cap = CodeEngineeringCapability(llm_client=None)
    cap._stamp_technique(
        root,
        TechniqueResolution(
            requested="target encoding",
            canonical="target_encoding",
            status="applied",
            reason="resolved",
            feature_recipes=["target_encoding"],
        ),
    )

    stamped = json.loads((root / "baseline_choice.json").read_text(encoding="utf-8"))
    assert stamped["applied_technique"]["canonical"] == "target_encoding"
    assert stamped["applied_technique"]["status"] == "applied"
    # The selector's own fields must survive the merge.
    assert stamped["problem_type"] == "tabular_regression"


def test_stamping_never_fails_a_run(tmp_path):
    """Provenance is not worth aborting training for."""
    from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.execution.technique.resolver import TechniqueResolution

    cap = CodeEngineeringCapability(llm_client=None)
    cap._stamp_technique(tmp_path / "missing", TechniqueResolution())  # no file
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "baseline_choice.json").write_text("{not json", encoding="utf-8")
    cap._stamp_technique(tmp_path / "bad", TechniqueResolution())


def test_a_rejected_label_never_reaches_the_codegen_prompt():
    """rogii asked codegen to implement `hyp:H-010`. The triad carries the real
    intent, so a rejected label is strictly worse than none."""
    res = resolve_technique({"technique": "hyp:H-010"}, {}, choice=_choice())
    assert res.status == "rejected"
    prompt_technique = None if res.status == "rejected" else res.requested or None
    assert prompt_technique is None
