#!/usr/bin/env bash
# Run the unit suite as if this were a different machine.
#
# The local stand-in for a second CI box. Mutation testing proves an assertion
# is wired to its code; it cannot prove the assertion means the same thing
# elsewhere. This shifts the ambient values that tests accidentally encode —
# core count and terminal width — and re-runs.
#
# Measured worth: four tests on the M11 fan-out branch passed on a ten-core
# laptop and failed on a two-core CI runner. All four reproduce here in seconds.
#
#   scripts/hostile-test.sh                 # the default sweep
#   scripts/hostile-test.sh tests/unit/test_fanout_branches.py
set -euo pipefail

cd "$(dirname "$0")/.."
targets=("${@:-tests/unit}")

# 1 and 2 cores because those are the shapes that broke: 1 exposes a cap that
# degenerates to no parallelism, 2 is what the CI runner actually has.
# COLUMNS=40 is below the width where Rich truncates long option names.
run() {
  local label="$1"; shift
  echo "── ${label} ─────────────────────────────────────────"
  if env "$@" uv run pytest -p plugins.hostile_env -q "${targets[@]}"; then
    echo "   ok"
  else
    echo "   FAILED under ${label} — the suite is asserting the machine, not the code" >&2
    return 1
  fi
}

failed=0
run "1 core"              FAKE_CPUS=1 || failed=1
run "2 cores (CI's shape)" FAKE_CPUS=2 || failed=1
run "40-column terminal"   COLUMNS=40  || failed=1

if [ "$failed" -ne 0 ]; then
  echo
  echo "At least one hostile environment failed. That is a real defect: the same" >&2
  echo "code passes here and fails on a machine shaped differently." >&2
  exit 1
fi
echo
echo "All hostile environments green."
