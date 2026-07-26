"""Offline / rule_engine code generation (Jinja baseline scaffolds).

Primary Code Engineering path is LLM → typed ``CodeProposal`` → apply.
This package is the deterministic fallback: render full train scripts from
templates under ``code_engineering/templates/``.
"""

from labpilot.research_engine.execution.capabilities.code_engineering.offline_codegen.renderer import (
    CodeRenderer,
    validate_pipeline,
    validate_python_syntax,
)

__all__ = ["CodeRenderer", "validate_pipeline", "validate_python_syntax"]
