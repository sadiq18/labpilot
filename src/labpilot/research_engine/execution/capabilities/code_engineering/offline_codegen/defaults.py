"""Default tabular model params for template offline codegen.

TODO: When template-based offline codegen is removed, delete this module and
any Jinja wiring that exists only to feed CodeRenderer.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TABULAR_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}
