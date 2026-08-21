#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare a demo host: a venv, the dashboard's dependencies, and the demo's assets.
#
# A VENV IS NOT OPTIONAL on a current Ubuntu. 24.04 marks the system Python
# externally-managed (PEP 668), so `pip install` into it refuses outright — and using
# --break-system-packages on a host that runs other services is how you break the other
# services. Everything here lives under $RUN and nothing is installed system-wide.
set -euo pipefail

RUN="${MAPPO_DEMO_RUN:-$HOME/.mappo-demo}"
VENV="${MAPPO_DEMO_VENV:-$RUN/venv}"
PYBIN="${MAPPO_PYTHON:-python3}"

echo "==> $("$PYBIN" --version) at $(command -v "$PYBIN")"
# device-connect-edge requires >= 3.11. Fail here with the reason rather than inside pip.
"$PYBIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"device-connect needs Python >= 3.11; this is {sys.version.split()[0]}")
PY

mkdir -p "$RUN/logs" "$RUN/served-models" "$RUN/frames" "$RUN/policy/models"
"$PYBIN" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
# Pillow is what burns the REPLAY label into the camera frames; without it the feed still
# works and is silently unlabelled, which is the one outcome worth paying a dependency for.
"$VENV/bin/pip" install --quiet \
    device-connect-edge device-connect-agent-tools aiohttp numpy Pillow

echo "==> installed into $VENV"
"$VENV/bin/python" - <<'PY'
import device_connect_edge, device_connect_agent_tools, aiohttp, numpy, PIL
print("    device-connect-edge", device_connect_edge.__version__ if hasattr(device_connect_edge, "__version__") else "ok")
print("    agent-tools, aiohttp, numpy, Pillow: ok")
PY
echo
echo "Next: copy the policy package, checkpoints and replay frames into:"
echo "  $RUN/policy          (config.json + models/)"
echo "  $RUN/served-models   (.npz the model server offers)"
echo "  $RUN/frames          (.jpg the camera replays)"
echo "Then: deploy/demo/run_demo.sh start"
