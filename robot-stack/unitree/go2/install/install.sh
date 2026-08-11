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

echo "[1/4] Python venv at $ENV_DIR"
python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --quiet --upgrade pip wheel

echo "[2/4] CycloneDDS bindings"
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

echo "[3/4] unitree_sdk2py"
if [ ! -d "$SDK_REPO/.git" ]; then
    git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python "$SDK_REPO"
fi
pip install --quiet -e "$SDK_REPO"

echo "[4/4] done. Env: $ENV_DIR"
echo "    activate:  source $ENV_DIR/bin/activate"
echo "    DDS iface: $IFACE   (export CYCLONEDDS_URI=file://$REPO_ROOT/unitree/go2/cyclonedds.xml)"

if [ "$VERIFY" = 1 ]; then
    echo ""
    echo "[verify] read-only probe of the live robot (no motion)…"
    IFACE="$IFACE" REPO_ROOT="$REPO_ROOT" "$(dirname "${BASH_SOURCE[0]}")/verify.sh"
fi
