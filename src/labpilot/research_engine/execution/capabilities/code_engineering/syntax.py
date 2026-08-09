"""Does the generated pipeline parse?

Lived in `offline_codegen.renderer` beside the Jinja baseline pack, and
outlived it: M19 §2 deleted the templates when delta became the default, but
"is this file valid Python" is asked of *every* proposal regardless of what
produced it — the smoke gate calls it on code the LLM wrote, and `apply` calls
`ast.parse` for the same reason.
"""

from __future__ import annotations

import ast
from pathlib import Path


def validate_python_syntax(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        ast.parse(path.read_text())
    except SyntaxError as exc:
        errors.append(f"{path}: {exc}")
    return errors


def validate_pipeline(pipeline_dir: Path) -> list[str]:
    errors: list[str] = []
    train_script = pipeline_dir / "train.py"
    if not train_script.exists():
        errors.append(f"Missing train.py in {pipeline_dir}")
    else:
        errors.extend(validate_python_syntax(train_script))
    return errors
