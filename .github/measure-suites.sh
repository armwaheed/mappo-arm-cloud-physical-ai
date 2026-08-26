#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run every offline test suite in the tree, and report how many tests each directory ran.
#
#   bash .github/measure-suites.sh            # run everything, print the inventory
#   bash .github/measure-suites.sh --write    # ...and rewrite .github/test-inventory.tsv
#   bash .github/measure-suites.sh --check    # ...and fail if it disagrees with that file
#
# WHY THIS IS A SCRIPT AND NOT A `run:` BLOCK. The per-directory test counts used to live
# in AGENTS.md, and CI re-measured them and failed on a mismatch. That caught real drift —
# the block read 189 for a directory that measures 196, 17 for one that measures 60, never
# listed three Go2 directories or detector/labels at all, and named ONE file in a directory
# of ten, so 41 tests existed that the documented command never ran. But it also made
# AGENTS.md a file that EVERY PR adding a test has to edit, and two changes collided on it
# in one night; one had to hand its count diff to another to apply.
#
# The counts now live in .github/test-inventory.tsv, which is generated. The point of a
# generated file is that nobody types a number into it — so the generator has to be
# runnable by a person, not only by CI, or the loop becomes "push, read the CI error, paste
# the number", which is worse than editing AGENTS.md. Hence one implementation, two callers:
# the workflow runs THIS FILE, and so do you. A second copy of the discovery loop living in
# the workflow is the same bug as two ruff.toml files that are meant to be identical.
#
# WHAT IT COUNTS. Every suite prints one `  ok  <name>` line per test. The count for a
# directory is those lines, summed over its test files. A file that exits 0 without
# printing one is an error, not a zero: that is the shape of a green tick over a suite that
# ran nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="$ROOT/.github/test-inventory.tsv"
PY="${PY:-python3}"

mode="print"
case "${1:-}" in
  "")        mode="print" ;;
  --write)   mode="write" ;;
  --check)   mode="check" ;;
  *) echo "usage: $0 [--write|--check]" >&2; exit 2 ;;
esac

# ---- annotations -----------------------------------------------------------------------
# ::warning:: and ::error:: land on the GitHub run summary rather than on line 400 of a
# collapsed log. Off the runner they would be noise, so they degrade to plain prose.
in_ci() { [ -n "${GITHUB_ACTIONS:-}" ]; }
warn()  { if in_ci; then printf '::warning file=%s::%s\n' "$1" "$2"; else printf 'WARNING  %s: %s\n' "$1" "$2"; fi; }
err()   { if in_ci; then printf '::error file=%s::%s\n' "$1" "$2"; else printf 'ERROR    %s: %s\n' "$1" "$2"; fi; }
bare()  { if in_ci; then printf '::error::%s\n' "$1"; else printf 'ERROR    %s\n' "$1"; fi; }
grp()   { if in_ci; then printf '::group::%s\n' "$1"; else printf -- '---- %s\n' "$1"; fi; }
egrp()  { if in_ci; then printf '::endgroup::\n'; fi; }

# ---- which files are not run, and why ---------------------------------------------------
# ENUMERATED, AND THAT DIRECTION IS SAFE: a file nobody has listed here gets RUN. Never
# invert it into an allow-list. The discovery below globs for the same reason.
py_minor="$("$PY" -c 'import sys; print(sys.version_info[1])')"
py_major="$("$PY" -c 'import sys; print(sys.version_info[0])')"
authoritative="no"
if [ "$py_major" -gt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -ge 11 ]; }; then
  authoritative="yes"
fi

skip_reason() {
  case "$1" in
    ./dashboard/test_robot_driver.py)
      echo "imports device_connect_edge — the Device Connect edge package is not on PyPI before launch. Fails identically on main."
      return ;;
    ./robot-stack/unitree/go2/deploy/test_go2_robot_io.py)
      echo "imports arm_dc_robotkit — the shared core is not on PyPI before launch. Fails identically on main."
      return ;;
  esac
  # dashboard/ needs Python >= 3.11 (Device Connect). drive_bridge.py is the deliberate
  # exception: it is the 3.8 half of the two-env bridge, and running its suite on 3.8 is
  # the only thing that proves the split still holds.
  case "$1" in
    ./dashboard/test_drive_bridge.py) : ;;
    ./dashboard/*)
      if [ "$authoritative" = "no" ]; then
        echo "dashboard/ requires Python >= 3.11 (Device Connect); this interpreter is $py_major.$py_minor"
      fi ;;
  esac
}

# ---- the interpreter has to be the one the inventory was measured with ------------------
# A missing dependency reads as a pass here: three tests in dashboard/test_camera_source.py
# print "  skip  " and then "  ok  " without Pillow. So --write and --check refuse to run
# on an interpreter or a dependency set that cannot produce the authoritative numbers,
# rather than writing a smaller inventory that looks like a deletion of tests.
if [ "$mode" != "print" ]; then
  if [ "$authoritative" = "no" ]; then
    bare "--$mode needs Python >= 3.11 (this is $py_major.$py_minor). On 3.8, dashboard/ is skipped and the inventory would be short by that many tests."
    exit 2
  fi
  missing=""
  for mod in numpy cv2 PIL pytest aiohttp; do
    "$PY" -c "import $mod" >/dev/null 2>&1 || missing="$missing $mod"
  done
  if [ -n "$missing" ]; then
    bare "--$mode needs these importable and they are not:$missing. Install: pip install numpy opencv-python-headless Pillow pytest aiohttp"
    exit 2
  fi
fi
"$PY" - <<'EOF' || true
import sys
mods = []
for name, attr in (("numpy", "__version__"), ("cv2", "__version__"),
                   ("PIL", "__version__"), ("pytest", "__version__")):
    try:
        mods.append("%s %s" % (name, getattr(__import__(name), attr)))
    except Exception:
        mods.append("%s MISSING" % name)
print("python %s  %s" % (sys.version.split()[0], "  ".join(mods)))
EOF

# ---- run ---------------------------------------------------------------------------------
# GLOBBED, NEVER ENUMERATED. AGENTS.md used to name test_lite3_state_probe.py explicitly in
# a directory that holds ten test files, and the other nine were never run. `find` cannot
# make that mistake. Each suite runs from its OWN directory, because each puts that
# directory on sys.path to import its siblings; from the repo root they die on the import
# rather than on a test.
cd "$ROOT" || exit 1
status=0; ran=0; skipped=0; total=0
counts_raw="$(mktemp)"; generated="$(mktemp)"; diffout="$(mktemp)"
trap 'rm -f "$counts_raw" "$generated" "$diffout"' EXIT

# ⚠️ DISCOVERY IS `git ls-files`, NOT `find`. A plain `find` walks gitignored directories,
# and this repository keeps agent worktrees under `.claude/worktrees/` -- full copies of the
# tree at older commits. Measured: `find` reported 3,171 tests across 45 directories where
# `git ls-files` reports 1,046 across 12, because 33 of those 45 were worktree copies. CI
# checks out fresh and never sees them, so `find` makes the script answer differently on a
# developer's machine than in CI -- and the developer is the one who runs `--write`, so the
# polluted number is the one that gets committed. Tracked files are what CI has, so tracked
# files are what this measures.
# The `./` prefix is load-bearing: skip_reason() matches on it, because `find .` emitted
# it and the skip list was written against that. Dropping it silently un-skips every
# entry in that list -- measured: both skipped suites ran, both died at import, and the
# run still reported success.
for test in $(git ls-files '*/test_*.py' 'test_*.py' | grep -v '/__pycache__/' | sed 's|^|./|' | sort); do
  dir="$(dirname "$test")"
  reason="$(skip_reason "$test")"
  if [ -n "$reason" ]; then
    warn "${test#./}" "SKIPPED — $reason"
    skipped=$((skipped + 1))
    continue
  fi
  grp "$test"
  if out="$( (cd "$dir" && "$PY" "$(basename "$test")") 2>&1 )"; then rc=0; else rc=$?; fi
  printf '%s\n' "$out"
  egrp
  # awk rather than `grep -c`: grep exits 1 when the count is zero, which is the case this
  # loop most needs to survive, and GitHub runs `run:` blocks under `-e -o pipefail`.
  n="$(printf '%s\n' "$out" | awk '/^  ok  /{n++} END{print n+0}')"
  if [ "$rc" -ne 0 ]; then
    err "${test#./}" "FAILED (exit $rc) after $n test(s)"
    status=1
  elif [ "$n" -eq 0 ]; then
    err "${test#./}" "exited 0 but printed no '  ok  ' line — it ran no tests"
    status=1
  fi
  printf '%s\n' "$out" | grep '^  skip  ' | while IFS= read -r s; do
    warn "${test#./}" "a test degraded rather than ran —${s#*skip}"
  done
  printf '%s\t%s\n' "${dir#./}" "$n" >> "$counts_raw"
  ran=$((ran + 1)); total=$((total + n))
done

if [ "$ran" -eq 0 ]; then
  bare "no test suite was run — the discovery loop found nothing"
  exit 1
fi
echo "ran $ran files, skipped $skipped, $total tests passed"

# ---- the inventory -----------------------------------------------------------------------
# One line per directory, TAB separated, sorted, AND DELIBERATELY WITHOUT A TOTAL. A total
# is one line that every single test-adding change has to rewrite, which is exactly the
# collision this file exists to remove; two changes touching different directories now
# touch different lines. The total is printed above and by CI instead.
#
# Not column-aligned for the same reason: padding would make one count crossing a digit
# boundary rewrite every other line.
{
  cat <<'HEADER'
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GENERATED — DO NOT EDIT BY HAND. Regenerate with:
#
#     bash .github/measure-suites.sh --write
#
# One line per directory that runs tests: <directory> TAB <number of "  ok  " lines>.
# CI re-measures this on every pull request and fails if a number here disagrees, in
# either direction, including a directory of tests this file does not list. Do not edit a
# number to make CI pass; the number is the measurement.
#
# There is no total line on purpose: a total is a line every test-adding change collides on.
HEADER
  awk -F'\t' '{c[$1]+=$2} END {for (d in c) printf "%s\t%d\n", d, c[d]}' "$counts_raw" | sort
} > "$generated"

# The measurement itself, for a caller that wants it whatever the mode and whatever the
# outcome — CI prints it as the run summary even when a suite failed.
if [ -n "${MEASURED_OUT:-}" ]; then cp "$generated" "$MEASURED_OUT"; fi

case "$mode" in
  print)
    grep -v '^#' "$generated" | grep -v '^$'
    ;;
  write)
    cp "$generated" "$INVENTORY"
    echo "wrote ${INVENTORY#"$ROOT"/}"
    ;;
  check)
    if [ ! -f "$INVENTORY" ]; then
      bare "${INVENTORY#"$ROOT"/} does not exist — run: bash .github/measure-suites.sh --write"
      status=1
    elif ! diff -u "$INVENTORY" "$generated" > "$diffout" 2>&1; then
      echo "-committed .github/test-inventory.tsv   +measured by this run"
      cat "$diffout"
      err ".github/test-inventory.tsv" "stale — a count here is not what this run measured, or a directory is missing. Regenerate: bash .github/measure-suites.sh --write"
      status=1
    else
      echo "inventory agrees: $total tests across $(grep -cv '^#' "$INVENTORY") directories"
    fi
    ;;
esac

exit $status
