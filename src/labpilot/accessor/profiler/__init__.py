"""Dataset profiling (EDA) shared by Workspace and Intelligence.

Produces ``DatasetProfile`` / ``profile.json`` from competition data dirs.
Accessor-owned so Execution and Intelligence can share it without importing
each other.
"""

from labpilot.accessor.profiler.report import load_profile, write_profile
from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile, TabularProfiler

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "TabularProfiler",
    "load_profile",
    "write_profile",
]
