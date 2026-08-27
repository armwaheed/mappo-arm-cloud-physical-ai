#!/usr/bin/env bash
# Reconstruct /home/unitree/mappo-main -- the tree the lab Go2 actually ran until
# 2026-08-27 -- byte for byte, from this repository's own object store.
#
# The tree matched no commit. It did not need to be committed: every one of its 134
# files is a blob git already holds, so the manifest beside this script is the whole
# artefact. Verified: rebuilding from it and diffing against the tarball pulled off
# the robot reports no differences.
#
#   bash rebuild.sh /tmp/mappo-main-reconstructed
set -euo pipefail
DEST="${1:?usage: rebuild.sh <destination directory>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/manifest.tsv"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"

[ -f "$MANIFEST" ] || { echo "rebuild: $MANIFEST is missing" >&2; exit 1; }
mkdir -p "$DEST"

n=0
missing=0
while IFS=$'\t' read -r rel blob _commit; do
  [ -n "$rel" ] || continue
  mkdir -p "$DEST/$(dirname "$rel")"
  if git -C "$ROOT" cat-file blob "$blob" > "$DEST/$rel" 2>/dev/null; then
    n=$((n + 1))
  else
    echo "rebuild: MISSING BLOB $blob for $rel" >&2
    missing=$((missing + 1))
  fi
done < "$MANIFEST"

echo "rebuild: wrote $n file(s) to $DEST"
if [ "$missing" -gt 0 ]; then
  echo "rebuild: $missing blob(s) absent from this clone -- fetch all refs and retry" >&2
  exit 1
fi
