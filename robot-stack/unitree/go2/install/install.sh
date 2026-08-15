#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Bring up the unitree_sdk2py environment the Go2 control stack needs, on
# the robot's onboard Jetson (or any host on the Go2 net). Creates a venv, installs
# CycloneDDS + unitree_sdk2py, and (with --verify) runs read-only probes that PASS
# only if the robot's DDS is actually visible.
#
# Usage:
#   ./install.sh                 # create the env + install the SDK
#   ./install.sh --verify        # ...and confirm rt/lowstate + rt/sportmodestate are live
#   IFACE=eth0 ./install.sh --verify

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/robotics-connect-go2}"
SDK_REPO="${SDK_REPO:-$HOME/unitree_sdk2_python}"
IFACE="${IFACE:-eth0}"
VERIFY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

#: True when the venv can already import NAME. The Go2 is not allowed on an
#: internet-connected network, so on the machine this script is actually written for
#: there is no package index to reach. "Already importable" therefore has to count as
#: installed — reaching for pip first fails on the primary target.
have() { python3 -c "import $1" >/dev/null 2>&1; }

echo "[1/4] Python venv at $ENV_DIR"
if [ -x "$ENV_DIR/bin/python3" ]; then
    # NEVER re-run `python3 -m venv` over an environment that already exists. It
    # rewrites pyvenv.cfg, and the line that matters on the Jetson is
    # include-system-site-packages: numpy, OpenCV and the CycloneDDS bindings are
    # distro packages there with no aarch64 wheels to fall back on, so flipping it to
    # false severs all three at once and the whole control stack stops importing.
    #
    # The rewrite also happens BEFORE the ensurepip step that fails on this image, so
    # a re-run reports an error about pip while the damage it has already done is to
    # something else entirely. That is the worst shape a bug can have in an installer:
    # it leaves the machine less working than it found it, and the message on the way
    # out names the wrong thing.
    echo "    -> reusing the existing environment (an installer must not rewrite one)"
else
    # --system-site-packages is not optional here, for the reason above: an isolated
    # venv on the Jetson cannot import numpy, OpenCV or CycloneDDS at all.
    # --without-pip is the fallback for images whose python3.8 ships no ensurepip —
    # the Jetson's does not — and the system pip installs into the venv anyway once
    # the venv is the active one.
    python3 -m venv --system-site-packages "$ENV_DIR" \
        || python3 -m venv --system-site-packages --without-pip "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
if python3 -m pip --version >/dev/null 2>&1; then
    pip install --quiet --upgrade pip wheel \
        || echo "    -> pip/wheel upgrade skipped (no package index reachable)"
else
    echo "    -> no pip in this venv; relying on what is already importable"
fi

echo "[2/4] CycloneDDS bindings"
if have cyclonedds; then
    echo "    -> cyclonedds already importable; leaving it alone"
else
    # The Go2 Jetson already ships a built CycloneDDS (~/cyclonedds_ws). Link the Python
    # bindings against it instead of rebuilding the C library from source (the usual
    # aarch64 gotcha). Fall back to a source build if that workspace isn't present.
    CDDS_INSTALL="$HOME/cyclonedds_ws/install"
    if [ -d "$CDDS_INSTALL" ]; then
        export CYCLONEDDS_HOME="$CDDS_INSTALL"
        echo "    -> using existing CycloneDDS at $CYCLONEDDS_HOME"
    else
        echo "    -> no ~/cyclonedds_ws; pip will build cyclonedds from source (needs cmake)"
    fi
    pip install --quiet "cyclonedds==0.10.2" numpy
fi

echo "[3/4] unitree_sdk2py"
if have unitree_sdk2py; then
    echo "    -> unitree_sdk2py already importable; leaving it alone"
else
    if [ ! -d "$SDK_REPO/.git" ]; then
        git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python "$SDK_REPO"
    fi
    pip install --quiet -e "$SDK_REPO"
fi

# Report what the environment can actually import, by importing it. Every step above
# can be skipped for a good reason, and a script that only prints the steps it took
# says nothing about whether the result works.
MISSING=""
for module in numpy cv2 cyclonedds unitree_sdk2py; do
    have "$module" || MISSING="$MISSING $module"
done
if [ -n "$MISSING" ]; then
    echo "[!] the environment cannot import:$MISSING" >&2
    echo "    Nothing was removed; the stack will not run until these resolve." >&2
    exit 1
fi

echo "[4/4] done. Env: $ENV_DIR"
echo "    activate:  source $ENV_DIR/bin/activate"
echo "    DDS iface: $IFACE   (export CYCLONEDDS_URI=file://$REPO_ROOT/unitree/go2/cyclonedds.xml)"

if [ "$VERIFY" = 1 ]; then
    echo ""
    echo "[verify] read-only probe of the live robot (no motion)…"
    IFACE="$IFACE" REPO_ROOT="$REPO_ROOT" "$(dirname "${BASH_SOURCE[0]}")/verify.sh"
fi
