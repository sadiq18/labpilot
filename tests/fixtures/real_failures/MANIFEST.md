# Real failure corpus

M20 exit criterion 3. Artifacts that a gate **let through**, kept verbatim so the
guard written to reject them can be proven against the thing itself.

The rule this exists to enforce is in `15-gates-must-fail.md`: *"do not test the
guard against a synthetic bad input when a real one exists."* A hand-written
"truncated file" would have had no `# /// script` block at all, passed the check,
and taught nothing — the real one is truncated **inside** the block, which is
exactly why it slipped past `ast.parse`.

Each entry is dated, sourced, and says which gate it defeated.

| file | bytes | date | source | the gate it passed |
|---|---|---|---|---|
| `truncated_train_py.txt` | 624 | 2026-08-08 | rogii, codegen output written to `pipeline/train.py` | `ast.parse` — 624 bytes of docstring and half a comment, syntactically valid, no code. `run_smoke_test` passed it too. Defects 6 and 8 |
| `stdlib_dependency_block.txt` | 248 | 2026-08-08 | rogii, codegen output | The PEP 723 block declares `glob`, which is stdlib. `uv` refused **all six** dependencies and the run never started. Defect 11 |
| `tqdm_flood_stderr.txt` | 6380 | 2026-08-08 | rogii, `run_training` stderr | 96 progress frames ahead of the traceback. `error[:1500]` stored the bar and dropped the diagnosis. Defect 9 |
| `record_reference_technique.txt` | 10 | 2026-08-08 | rogii, `.labpilot/skills/*.md` overlay | A hypothesis id where a technique name belongs, carried into six agents' system prompts on every run. Defect 1 |

## Adding to it

An artifact belongs here when it **actually occurred** and a gate **actually
passed it**. Not a case someone imagined a gate might miss — those are ordinary
unit tests and belong beside the code.

Give the **byte count**, the date, the workspace it came from, and the gate it
defeated. The size is not decoration: `truncated_train_py.txt` sat here for a day
as a **79-byte fragment typed from memory** while the manifest and three
docstrings called it 624 bytes — the corpus that exists so guards face the real
artifact, holding a paraphrase. Reported on PR #120. The real file was recovered
from `artifacts/code_backups/train_E-167.py`, and
`test_every_artifact_is_the_size_the_manifest_claims` now makes the drift
impossible to repeat.

Store it as `.txt` even when the content is Python: these are data, not modules, and the
repo's linters should not hold generated competition output to hand-written
standards.
