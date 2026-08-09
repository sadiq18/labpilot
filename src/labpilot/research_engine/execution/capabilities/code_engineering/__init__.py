"""Code Engineering — LLM proposes full code; platform applies it.

The Jinja baseline pack it used to own was deleted with M19 §2, in the change
that made ``delta`` the default.
"""

from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.capabilities.code_engineering.syntax import (
    validate_pipeline,
    validate_python_syntax,
)

__all__ = [
    "CodeEngineeringCapability",
        "validate_pipeline",
    "validate_python_syntax",
]
