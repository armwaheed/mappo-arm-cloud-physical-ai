#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# One peer-avoidance run, in the only geometry where avoidance is possible on this robot.
#
#   ./run-peer-supervised.sh scene          # no legs, checks the scene is set
#   ./run-peer-supervised.sh live  tag      # walks
#
# ─────────────────────────────────────────────────────────────────────────────
# WHERE THE PEER GOES, AND WHY IT IS NOT A PREFERENCE
#
# 1.75 m ahead, 0.40 m to the robot's LEFT of the straight line to the goal.
#
# That range is bounded on BOTH sides by measurements from 2026-08-25, and the window
# between them is narrow.
#
#   TOO FAR.  Full-frame detection of this peer is 0 of 315 frames beyond 2.7 m. Not
#             "unreliable" -- never. It is 80% at 1.5-1.9 m and 91% inside 1.1 m.
#
#   TOO NEAR. A 0.35 m peer at 0.45 m subtends asin(0.35/0.45) = 51 deg of half-angle,
#             so 102 deg of arc against an 85 deg field of view. It fills the camera.
#             `render_observation` scored the 2026-08-25 run at 0 of 91 driven ticks
#             with an open window toward the goal, and the robot -- given no direction
#             to steer -- drove into the peer and pushed it the length of the corridor.
#             No radius cap fixes that; the geometry forbids an opening.
#
# At 1.75 m the same peer subtends 23 deg, leaving 18 deg of open window to its left and
# 44 deg to its right. The policy has somewhere to go.
#
# The 0.40 m offset is what makes it a TEST rather than a drive-past. Peer radius 0.35
# plus robot radius 0.25 needs 0.60 m of centre-to-centre clearance to pass; at 0.40 m
# offset a robot going straight passes 0.20 m too close. It must deviate or it collides.
#
# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISED, NOT RAW.
#
# `mappo_drive` prints this on every raw start: "NO VETO. In the closed-loop simulation
# the raw policy collided and the supervised one did not. Empty arena only." On
# 2026-08-25 it was run in raw anyway and the raw policy collided, exactly as stated.
# The veto stays on here. If it fires and holds the robot, that is the safety system
# working and it is a result, not a failure of the run.
#
# ─────────────────────────────────────────────────────────────────────────────
# BEFORE EVERY RUN
#
#   * Hand-pose the D1 arm flat along the spine. It back-drives while walking and the
#     3.0 deg gate is ABSOLUTE, so creep accumulates: 1.8 -> 3.1 -> 5.0 deg over three
#     runs on 2026-08-25, one step per walk. It cannot be recentred in software on this
#     unit -- a commanded move produces 0.00 deg.
#   * Clear BOTH sides of the lane. The camera is an 85 deg forward cone with no lateral
#     sensing whatever, and this run is specifically trying to make the robot step
#     sideways.
#   * Check the goal marker is visible past the peer.
#
set -euo pipefail
MODE="${1:?usage: run-peer-supervised.sh scene|live [tag]}"
TAG="${2:-peer}"
TREE="${TREE:-/home/unitree/mappo-run}"
SECS="${SECS:-40}"

# Only exists on a robot with the stack installed; not resolvable at lint time.
# shellcheck disable=SC1091
source "${TREE}/robot-stack/unitree/go2/install/setup_env.sh"
export PYTHONPATH=/home/unitree/deps:/home/unitree/unitree_sdk2_python
cd "${TREE}/integration"

COMMON=(
  --package "${TREE}/policy"
  --calibration /home/unitree/go2_front_camera.json
  # 105 mm, not the 140 mm the sheet was generated at: the print scaled to ~75%.
  # Calibrated against the operator's tape rather than a ruler on the marker --
  # 45.3 px at a measured 3.0 m gives 1290.16 * 0.105 / 45.3 = 3.02 m, self-consistent.
  --marker-size 0.105
  # Full resolution for the marker. At the default 0.5 a 105 mm marker stops locking
  # past ~3.0 m, which aborted two runs as "goal never sighted".
  --goal-detect-scale 1.0
  # 224 rather than 300: measured 12/12 detections of this peer against 2/12 at 300,
  # and faster. Non-monotonic in input size, which is a marginal-detection smell --
  # do not read it as a rule.
  --input-size 224
  --classes aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog
            horse motorbike person pottedplant sheep sofa train tvmonitor
  # The peer's own dimensions. Passing width explicitly matters: without it the width
  # prior comes from a standing adult's aspect ratio and a vertically clipped box ranges
  # the peer at 0.09-0.14 m, inside the robot's own footprint.
  --obstacle-height 0.514 --obstacle-width 0.31
  --confidence 0.25
  --robot-radius 0.25 --obstacle-radius 0.20
  --no-latch-arm --arrive 0.30
  --policy-mode supervised          # THE VETO STAYS ON
  # Explicit rather than inherited. Off IS the default since issue #16, but this script
  # is the one somebody copies, and a runbook that shows the safe value survives a
  # future change of default that a silent omission would not.
  --heading-servo off               # no yaw servo; the robot crabs, and that is
                                    # the point -- issue #16
  --policy-gait-floor 0.35
  --policy-scale 5.0
  --max-seconds "${SECS}"
)

case "${MODE}" in
  scene) exec python3 mappo_drive.py --telemetry "/home/unitree/${TAG}-scene.jsonl" \
             --record "/home/unitree/${TAG}-scene.mp4" "${COMMON[@]}" ;;
  live)  exec python3 mappo_drive.py --live --telemetry "/home/unitree/${TAG}.jsonl" \
             --record "/home/unitree/${TAG}.mp4" "${COMMON[@]}" ;;
  *) echo "usage: run-peer-supervised.sh scene|live [tag]" >&2; exit 2 ;;
esac
