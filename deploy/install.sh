#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Bring the whole demo up on a machine: the Go2 control stack's environment, the MAPPO
# policy package on top of it, and the checks that say whether either of them actually
# works here.
#
# It does NOT reimplement the robot environment. Step 1 calls the stack's own
# robot-stack/unitree/go2/install/install.sh, because a second copy of the CycloneDDS
# and unitree_sdk2py recipe is a copy that will drift — that has already happened three
# times in this project (see PROVENANCE.md).
#
# Idempotent: run it again after a change and it re-checks rather than re-installs. Every
# path it creates is recorded in a manifest so uninstall.sh can remove exactly those and
# nothing else.
#
# Usage:
#   ./install.sh                          # full install (robot env + policy)
#   ./install.sh --verify                 # ...and probe the live robot's DDS (no motion)
#   ./install.sh --policy-only            # policy + offline checks; no robot SDK
#   ./install.sh --robotkit ~/robotkit    # also deploy the shared core the stack imports
#   ENV_DIR=~/my-env ./install.sh
#
# On the Go2's Jetson, run it from a checkout on the robot. On a workstation, use
# --policy-only: everything except the DDS layer runs there, including the closed-loop
# simulation, which is the thing that gates a live run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/robotics-connect-go2}"
SDK_REPO="${SDK_REPO:-$HOME/unitree_sdk2_python}"
MANIFEST="${MANIFEST:-$HOME/.mappo-go2-deploy.manifest}"
IFACE="${IFACE:-eth0}"
POLICY_ONLY=0
VERIFY=0
ROBOTKIT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --policy-only) POLICY_ONLY=1 ;;
        --verify) VERIFY=1 ;;
        --robotkit) ROBOTKIT="${2:?--robotkit needs a path}"; shift ;;
        --env-dir) ENV_DIR="${2:?--env-dir needs a path}"; shift ;;
        --iface) IFACE="${2:?--iface needs a name}"; shift ;;
        -h|--help) sed -n '5,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

POLICY_DIR="$REPO_ROOT/policy"
CHECKPOINT="$POLICY_DIR/models/mappo_actor_3agent_1910000.npz"
# Recorded here as well as in policy/PROVENANCE.md, because this is the copy that runs on
# the robot and a truncated scp is silent: numpy loads a short npz as a KeyError several
# frames from the cause, if it fails at all.
CHECKPOINT_SHA256="7327f72401adfdfa1931a516e85aeee62b5bee0e06e976c13600515ca2d2ca11"

step() { echo ""; echo "── $* ──"; }
fail() { echo "FAIL  $*" >&2; exit 1; }

# Manifest lines are appended as `key value`; uninstall.sh reads the last value for each
# key, so re-running install just updates it.
record() { printf '%s %s\n' "$1" "$2" >>"$MANIFEST"; }

step "0/6  where things are"
echo "    repo        $REPO_ROOT"
echo "    env         $ENV_DIR"
echo "    manifest    $MANIFEST"
[ -f "$POLICY_DIR/physical_ai_mappo.py" ] || fail "no policy package at $POLICY_DIR"

if [ ! -f "$MANIFEST" ]; then
    printf '# mappo-go2 deploy manifest v1 — written by deploy/install.sh\n' >"$MANIFEST"
fi
record installed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
record repo "$REPO_ROOT"
record env_dir "$ENV_DIR"

step "1/6  the checkpoint"
[ -f "$CHECKPOINT" ] || fail "no checkpoint at $CHECKPOINT"
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
else
    ACTUAL="$(shasum -a 256 "$CHECKPOINT" | cut -d' ' -f1)"
fi
[ "$ACTUAL" = "$CHECKPOINT_SHA256" ] || fail \
    "checkpoint sha256 is $ACTUAL, expected $CHECKPOINT_SHA256 — the file is not the one
     this repository was tested against. Re-clone rather than guessing."
echo "    PASS  $(basename "$CHECKPOINT") matches its recorded sha256"

step "2/6  the robot environment"
if [ "$POLICY_ONLY" = 1 ]; then
    echo "    SKIP  --policy-only: no venv, no CycloneDDS, no unitree_sdk2py."
    echo "          Using $(command -v python3): $(python3 -V 2>&1)"
    PYTHON=python3
else
    [ -d "$ENV_DIR" ] && ENV_EXISTED=1 || ENV_EXISTED=0
    [ -d "$SDK_REPO/.git" ] && SDK_EXISTED=1 || SDK_EXISTED=0
    ENV_DIR="$ENV_DIR" SDK_REPO="$SDK_REPO" IFACE="$IFACE" \
        "$REPO_ROOT/robot-stack/unitree/go2/install/install.sh"
    # Only what THIS run created is removable later. An environment that was already
    # there belongs to whoever made it.
    record created_env "$((1 - ENV_EXISTED))"
    record created_sdk_clone "$((1 - SDK_EXISTED))"
    record sdk_repo "$SDK_REPO"
    PYTHON="$ENV_DIR/bin/python3"
fi

step "3/6  the policy package"
"$PYTHON" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$PYTHON" -m pip install --quiet -r "$POLICY_DIR/requirements.txt"
echo "    numpy $("$PYTHON" -c 'import numpy; print(numpy.__version__)')"

if [ -n "$ROBOTKIT" ]; then
    step "3b/6  the shared core the stack imports"
    # Deployed as a directory rather than pip-installed: its packaging declares
    # requires-python >= 3.10 and the Go2's Jetson is on 3.8, so pip refuses it.
    # See PROVENANCE.md, "Deploying to the robot".
    DEPS_DIR="${DEPS_DIR:-$HOME/deps}"
    [ -d "$DEPS_DIR/arm_dc_robotkit" ] && KIT_EXISTED=1 || KIT_EXISTED=0
    mkdir -p "$DEPS_DIR/arm_dc_robotkit"
    rsync -a "${ROBOTKIT%/}/lib/" "$DEPS_DIR/arm_dc_robotkit/"
    record robotkit_dir "$DEPS_DIR/arm_dc_robotkit"
    record created_robotkit "$((1 - KIT_EXISTED))"
    echo "    deployed to $DEPS_DIR/arm_dc_robotkit — export PYTHONPATH=$DEPS_DIR"
fi

step "4/6  smoke check: can this machine run one inference"
( cd "$POLICY_DIR" && "$PYTHON" basic_test.py )

step "5/6  offline tests"
# These prove the TREE is intact, not that the robot is. They are the same suites the
# repository's own pre-flight runs, and they are fast enough to be worth running here:
# a truncated checkout or a half-applied patch shows up as a failure rather than as
# strange behaviour in the arena tomorrow.
FAILED=0
for suite in "$POLICY_DIR"/test_*.py; do
    name="$(basename "$suite")"
    if ( cd "$POLICY_DIR" && "$PYTHON" "$name" >/dev/null 2>&1 ); then
        echo "    PASS  policy/$name"
    else
        echo "    FAIL  policy/$name"; FAILED=1
    fi
done
for suite in "$REPO_ROOT"/integration/test_*.py; do
    name="$(basename "$suite")"
    if ( cd "$REPO_ROOT/integration" && "$PYTHON" "$name" >/dev/null 2>&1 ); then
        echo "    PASS  integration/$name"
    else
        echo "    FAIL  integration/$name"; FAILED=1
    fi
done
[ "$FAILED" = 0 ] || fail "an offline suite failed — run it directly to see the output"

step "6/6  the live robot"
if [ "$VERIFY" = 1 ] && [ "$POLICY_ONLY" != 1 ]; then
    # The stack's own read-only probe, for the same reason step 2 calls its installer:
    # a second copy of "is the robot's DDS visible" would be one more thing to keep in
    # step with the SDK. It subscribes to rt/lowstate and rt/sportmodestate and passes
    # only if a frame arrives. No motion.
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    IFACE="$IFACE" REPO_ROOT="$REPO_ROOT/robot-stack" \
        "$REPO_ROOT/robot-stack/unitree/go2/install/verify.sh" \
        || fail "the robot's DDS is not visible from here — check the cable, the iface
     (--iface, currently $IFACE), and that the robot is powered and out of damping mode"
else
    echo "    SKIP  pass --verify to probe the robot's DDS (read-only, no motion)"
fi

if [ "$POLICY_ONLY" = 1 ]; then
    cat <<EOF

────────────────────────────────────────────────────────────────────────────────
Installed (policy only). Manifest: $MANIFEST

This machine has no robot environment, which is the right shape for a workstation: the
part that gates a live run does not need one.

    cd $REPO_ROOT/integration
    python3 closed_loop_sim.py --seeds 30 --scale 2.5 --command-scale 1.0
    python3 replay_mappo.py ../evidence/sample_telemetry.jsonl --scale 1.5 2.5
    python3 mappo_shadow.py <a run.jsonl copied off the robot>

Run install.sh without --policy-only on the Go2 itself. Read deploy/README.md first.
────────────────────────────────────────────────────────────────────────────────
EOF
    exit 0
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────────
Installed. Manifest: $MANIFEST   (deploy/uninstall.sh reads it)

Read deploy/README.md before the robot moves. The short version, in order:

  1. CLEAR THE GATE — off the robot, no hardware needed:
       cd $REPO_ROOT/integration
       python3 closed_loop_sim.py --seeds 30 --scale 2.5 --command-scale 1.0

  2. SHADOW — on the robot, the planner drives, the policy only watches:
       source $ENV_DIR/bin/activate
       cd $REPO_ROOT/robot-stack/unitree/go2/visual_nav
       python3 visual_nav.py --live --telemetry run.jsonl --robot-radius 0.25 ...
       # in a second shell:
       cd $REPO_ROOT/integration
       python3 mappo_shadow.py run.jsonl --follow --out shadow.jsonl

  3. DRIVE — only after 1 and 2, and only with an operator on the remote:
       cd $REPO_ROOT/integration
       python3 mappo_drive.py --live --robot-radius 0.25 --telemetry drive.jsonl ...

robot-stack/SAFETY.md governs anything that moves a leg. It is not optional.
────────────────────────────────────────────────────────────────────────────────
EOF
