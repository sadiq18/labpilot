"""Code Engineering — LLM proposes full code; platform applies it.

Also owns offline Jinja scaffolds (``offline_codegen`` + ``templates/``).
"""

from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen import (
    CodeRenderer,
    validate_pipeline,
    validate_python_syntax,
)

__all__ = [
    "CodeEngineeringCapability",
    "CodeRenderer",
    "validate_pipeline",
    "validate_python_syntax",
]
