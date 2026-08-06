"""Rendered `train.py` must contain Python literals, not JSON ones.

The templates used to interpolate constants with `| tojson`, which spells the
singletons `null` / `true` / `false`. Those are valid Python *names*, so the
output passed `validate_python_syntax` and only failed at import:

    NameError: name 'null' is not defined

It needed a None-valued field to bite, so competitions that happened to set
every optional field never saw it. These tests render every template with the
optional `BaselineChoice` fields left None — the shape that triggers it.
"""

from __future__ import annotations

import ast

import pytest

from labpilot.config import TrainingConfig
from labpilot.research_engine.execution.baseline.registry import list_templates
from labpilot.research_engine.execution.baseline.selector import BaselineChoice
from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (  # noqa: E501
    CodeRenderer,
    py_literal,
)

JSON_ONLY_NAMES = {"null", "true", "false"}

TEMPLATES = [t.name for t in list_templates()]


def _bare_json_names(source: str) -> set[str]:
    """Names the module reads but never binds, restricted to the JSON spellings.

    Checking the parse tree rather than the text avoids matching `null` inside
    a docstring or a column name, which would be a false alarm.
    """
    tree = ast.parse(source)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in JSON_ONLY_NAMES
    }


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_renders_python_literals_when_optional_fields_are_none(tmp_path, template_name):
    """Every template, with every None-able field actually None."""
    template = next(t for t in list_templates() if t.name == template_name)
    choice = BaselineChoice(
        problem_type=template.problem_type,
        template_name=template_name,
        rationale="test",
        # Left at their None defaults on purpose: target_column, id_column,
        # train_file, test_file, sample_submission_file, text_column,
        # image_dir, image_column.
    )

    run_dir = tmp_path / template_name
    pipeline_dir = CodeRenderer(TrainingConfig()).render(template, choice, run_dir)
    source = (pipeline_dir / "train.py").read_text(encoding="utf-8")

    assert not _bare_json_names(source), (
        f"{template_name}/train.py references JSON literals that do not exist "
        f"in Python: {sorted(_bare_json_names(source))}"
    )


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_rendered_module_imports_without_nameerror(tmp_path, template_name):
    """The failure the parse check stands in for: `null` is syntactically fine,
    so only executing the module surfaces it. Constants live above the first
    function, so compiling and running that prefix is enough — and avoids
    importing torch/transformers for the deep templates."""
    template = next(t for t in list_templates() if t.name == template_name)
    choice = BaselineChoice(
        problem_type=template.problem_type,
        template_name=template_name,
        rationale="test",
    )

    run_dir = tmp_path / template_name
    pipeline_dir = CodeRenderer(TrainingConfig()).render(template, choice, run_dir)
    tree = ast.parse((pipeline_dir / "train.py").read_text(encoding="utf-8"))
    constants = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    module = ast.Module(body=constants, type_ignores=[])
    exec(compile(module, "train.py", "exec"), {"Path": type(run_dir)})


def test_py_literal_round_trips_the_values_templates_carry():
    for value in [None, True, False, "col", 3, 0.5, ["a", None], {"k": None}]:
        assert ast.literal_eval(py_literal(value)) == value
