#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Read-only self-check: confirm the Go2's DDS is visible from this host. Subscribes
# to rt/lowstate + rt/sportmodestate and PASSES only if a frame arrives. NO motion.
#
# Usage:  IFACE=eth0 ./verify.sh     (run inside the install venv)

set -euo pipefail
IFACE="${IFACE:-eth0}"

python3 - "$IFACE" <<'PY'
import sys, time
iface = sys.argv[1]
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
except Exception as exc:
    print(f"FAIL  unitree_sdk2py import: {exc!r}")
    sys.exit(1)

ChannelFactoryInitialize(0, iface)
seen = {"lowstate": False, "sportmodestate": False}

def _mk(key):
    def _cb(_msg):
        seen[key] = True
    return _cb

subs = [
    ChannelSubscriber("rt/lowstate", LowState_),
    ChannelSubscriber("rt/sportmodestate", SportModeState_),
]
subs[0].Init(_mk("lowstate"), 10)
subs[1].Init(_mk("sportmodestate"), 10)

deadline = time.monotonic() + 5.0
while time.monotonic() < deadline and not all(seen.values()):
    time.sleep(0.05)

for topic, ok in seen.items():
    print(f"  {'PASS' if ok else 'FAIL'}  rt/{topic}")
sys.exit(0 if all(seen.values()) else 1)
PY

echo "[verify] arm topics (optional; only if a D1 is attached):"
python3 - "$IFACE" <<'PY' || true
import sys
iface = sys.argv[1]
# The arm uses the unitree_arm ArmString IDL (D1 SDK / vendored). Just report whether
# the D1 module can resolve its type — a full live check needs the arm powered.
sys.path.insert(0, __import__("os").path.join(
    __import__("os").environ.get("REPO_ROOT", "."), "unitree", "go2", "d1_arm"))
try:
    import d1_arm
    d1_arm._resolve_armstring()
    print("  PASS  D1 ArmString IDL resolvable (rt/arm_Command ready)")
except Exception as exc:
    print(f"  SKIP  D1 arm not set up: {exc!r}")
PY
