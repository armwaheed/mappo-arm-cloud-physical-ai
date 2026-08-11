# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable environment bootstrap for the robotics-connect Go2 stack on the
# robot's Jetson. Source it before running anything that talks to the robot:
#
#     source /home/unitree/robotics-connect/unitree/go2/install/setup_env.sh
#     python3 examples/go2_marker_grab_smoke.py
#
# WHY LD_LIBRARY_PATH is load-bearing (do not drop it):
#   unitree_sdk2py's RPC clients (SportClient, VideoClient, MotionSwitcherClient)
#   SEGFAULT while serializing a Request_ if the cyclonedds Python binding loads a
#   *different* build of libddsc than the one it was compiled against. This Jetson
#   ships two 0.10.2 builds — the ROS2 workspace build under ~/cyclonedds_ws and a
#   stray /usr/local/lib/libddsc.so.0. ldconfig resolves to /usr/local by default,
#   which is the WRONG one, so the crash is the default. Putting the ws lib dir
#   first fixes every RPC path. (Passive DDS subscribers work either way; only the
#   RPC serialize path is sensitive.) Diagnosed on this unit 2026-07-08.

# Per-researcher venv (override with ROBOTICS_CONNECT_ENV to use your own).
ENV_DIR="${ROBOTICS_CONNECT_ENV:-/home/unitree/robotics-connect-envs/armwaheed}"

# The CycloneDDS the Python binding was built against (colcon workspace build).
CDDS_PREFIX="${CDDS_PREFIX:-/home/unitree/cyclonedds_ws/install/cyclonedds}"

export CYCLONEDDS_HOME="${CDDS_PREFIX}"
export LD_LIBRARY_PATH="${CDDS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Put the venv first so `python3` is the armwaheed interpreter with the SDK.
if [ -d "${ENV_DIR}/bin" ]; then
    export PATH="${ENV_DIR}/bin:${PATH}"
else
    echo "setup_env.sh: WARNING - venv not found at ${ENV_DIR}" >&2
fi

# The Go2 host<->robot Cyclone config (unicast SPDP on the wired net). Only set it
# if the repo copy exists and the caller hasn't already chosen one.
if [ -z "${CYCLONEDDS_URI:-}" ]; then
    _rc_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
    if [ -f "${_rc_root}/unitree/go2/cyclonedds.xml" ]; then
        export CYCLONEDDS_URI="file://${_rc_root}/unitree/go2/cyclonedds.xml"
    fi
    unset _rc_root
fi
