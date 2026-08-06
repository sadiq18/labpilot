"""Label semantics shared by evidence attribution and the hypothesis ledger.

One rule lives here because it was independently implemented twice and was
wrong or missing both times.

A *record reference* is a tag that points at a stored record — ``hyp:H-010``,
``fork:H-003`` — rather than naming a method. ``outcome.py`` appends
``hyp:{hypothesis_id}`` to an experiment's tags for provenance, so these travel
alongside genuine technique names and must be filtered wherever tags are read
as techniques.

Measured on the rogii workspace before this module existed:

* ``hyp:H-010`` was the most common "technique" in the knowledge base — 11
  durable records, ahead of every real one;
* ``techniques.name`` held five of them (``hyp:H-010`` … ``hyp:H-BASELINE``);
* a belief recorded ``hyp:H-010`` with ``effect='negative'``,
  ``status='rejected'`` — a fabricated *failure* of a technique that never
  existed;
* six of ten plans carried one in ``metadata['technique']``, so codegen was
  being asked to implement ``hyp:H-010``.

The two prior attempts, and why each failed:

1. ``evidence/builder.py`` filtered ``fork:`` and simply omitted ``hyp:``.
2. ``ledger.py::_index_technique`` tested ``normalize_label(name)`` — which
   strips non-alphanumerics, turning ``hyp:H-010`` into ``hyph010`` — against a
   prefix *containing a colon*. That comparison can never be true, so the guard
   read as protection while doing nothing.

Hence the rule below deliberately operates on the **raw** label, and callers
must not pass a normalised one.
"""

from __future__ import annotations

#: Prefixes marking a reference to a stored record rather than a method name.
RECORD_REFERENCE_PREFIXES = ("hyp:", "fork:")


def is_record_reference(label: str) -> bool:
    """True when ``label`` points at a record instead of naming a technique.

    Pass the raw label. Normalising first removes the delimiter these prefixes
    depend on, which is exactly how the previous guard was defeated.
    """
    return str(label).strip().lower().startswith(RECORD_REFERENCE_PREFIXES)
