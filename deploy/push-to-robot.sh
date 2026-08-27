#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy this repository to a robot so that the robot can afterwards say which commit it
# is running -- and prove it, from the bytes on its own disk, with no `.git` and no `git`.
#
# Usage:
#   bash deploy/push-to-robot.sh <user@host> <dest> [commit] [-- <ssh options>]
#   bash deploy/push-to-robot.sh unitree@192.168.123.18 /home/unitree/mappo-run
#
# What it does, and why each step is here rather than in a one-line rsync:
#
#   1. Refuses a dirty working tree unless ALLOW_DIRTY=1. A deploy from uncommitted work
#      produces a tree id that resolves to nothing, which is honest but useless.
#   2. Ships `git archive <commit>` -- TRACKED FILES ONLY, at that commit. Not the working
#      directory: a stray .pyc or a 4.8 MB debugging GIF in the source tree would land on
#      the robot and change the tree id away from the commit's.
#   3. Extracts into a STAGING directory, stamps it, verifies it there, and only then
#      swaps it into place. A deploy that fails half way through never leaves a robot
#      running a tree that is partly one commit and partly another -- which, judging by
#      ~/mappo-main on the lab Go2, is how a tree comes to span 83 commits.
#   4. Moves any existing tree aside instead of deleting it. Nothing on a robot is
#      deleted by this script, ever.
#   5. Re-verifies over SSH after the swap and FAILS THE DEPLOY if the robot disagrees.
#      A deploy that does not check itself is a deploy that reports success when `tar`
#      ran out of disk.
#
# The verifier ships inside the tree it verifies and is covered by the stamp, so editing
# it changes the tree id. That is not a proof against a determined editor -- they can edit
# both -- and `tree_stamp.py audit` off-robot is the half that catches that; see the
# module docstring, which does not oversell this.

set -euo pipefail

HOST="${1:?usage: push-to-robot.sh <user@host> <dest> [commit]}"
DEST="${2:?usage: push-to-robot.sh <user@host> <dest> [commit]}"
COMMIT="${3:-HEAD}"

ROOT="$(git rev-parse --show-toplevel)"
STAMPER="$ROOT/robot-stack/preflight/tree_stamp.py"
SSH=(ssh -o StrictHostKeyChecking=no)
SCP=(scp -o StrictHostKeyChecking=no)
if [ -n "${ROBOT_SSHPASS:-}" ]; then
  SSH=(sshpass -p "$ROBOT_SSHPASS" "${SSH[@]}")
  SCP=(sshpass -p "$ROBOT_SSHPASS" "${SCP[@]}")
fi

if [ -n "$(git -C "$ROOT" status --porcelain)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
  echo "push-to-robot: working tree is dirty. Commit, or re-run with ALLOW_DIRTY=1 to" >&2
  echo "  deploy $COMMIT anyway -- the stamp will say dirty=true and its tree id will" >&2
  echo "  not resolve to anything in the repository." >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi

SHA="$(git -C "$ROOT" rev-parse "$COMMIT")"
STAGE="$DEST.staging-${SHA:0:12}"
ASIDE="$DEST.superseded-$(date -u +%Y%m%dT%H%M%SZ)"

echo "push-to-robot: $SHA -> $HOST:$DEST"

# 1. staging directory, tracked files only, at the named commit.
"${SSH[@]}" "$HOST" "rm -rf '$STAGE' && mkdir -p '$STAGE'"
git -C "$ROOT" archive --format=tar "$SHA" | "${SSH[@]}" "$HOST" "tar -x -C '$STAGE'"

# 2. the stamp is computed HERE, where git is, and copied over. The robot never needs it.
TMP="$(mktemp -t mappo-stamp.XXXXXX)"
trap 'rm -f "$TMP"' EXIT
python3 "$STAMPER" stamp "$ROOT" > "$TMP"
"${SCP[@]}" -q "$TMP" "$HOST:$STAGE/.mappo-stamp.json"

# 3. verify in staging, before anything in $DEST is touched.
echo "push-to-robot: verifying the staged tree on the robot..."
"${SSH[@]}" "$HOST" "python3 '$STAGE/robot-stack/preflight/tree_stamp.py' verify '$STAGE'"

# 4. swap. The old tree is MOVED, never removed.
"${SSH[@]}" "$HOST" "
  set -e
  if [ -e '$DEST' ]; then mv '$DEST' '$ASIDE'; echo 'push-to-robot: previous tree kept at $ASIDE'; fi
  mv '$STAGE' '$DEST'"

# 5. verify again, in place, and let a failure fail the deploy.
echo "push-to-robot: verifying in place..."
"${SSH[@]}" "$HOST" "python3 '$DEST/robot-stack/preflight/tree_stamp.py' verify '$DEST'"

echo "push-to-robot: done. Ask the robot what it is running with:"
echo "    ${SSH[*]} $HOST \"python3 '$DEST/robot-stack/preflight/tree_stamp.py' verify '$DEST'\""
