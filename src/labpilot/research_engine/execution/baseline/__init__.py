"""Baseline template registry and selection (Research Engineer)."""

from labpilot.research_engine.execution.baseline.registry import (
    BaselineTemplate,
    get_template,
    get_templates_root,
    list_templates,
)
from labpilot.research_engine.execution.baseline.selector import (
    BaselineChoice,
    BaselineSelector,
)

__all__ = [
    "BaselineChoice",
    "BaselineSelector",
    "BaselineTemplate",
    "get_template",
    "get_templates_root",
    "list_templates",
]
