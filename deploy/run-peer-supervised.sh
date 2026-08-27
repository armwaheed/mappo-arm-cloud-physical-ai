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

# ─────────────────────────────────────────────────────────────────────────────
# THE DETECTOR PREPROCESSING IS NOT WRITTEN DOWN HERE ANY MORE, AND THAT IS THE FIX.
#
# THIS ROBOT RUNS THREE DIFFERENT DETECTORS DEPENDING ON WHICH SCRIPT STARTS IT:
#
#   this script                        --input-size 224   --confidence 0.25
#   run-smoke / run-berth / run-chair    (no flag) -> 300   --confidence 0.45
#   a bare visual_nav.py                 (default) -> 300              0.4
#
# Nothing reconciled those, and no run recorded which one produced it, so a number
# carried between two runs launched by different scripts was silently comparing two
# detectors. The checkpoint sweep then managed to score through a FOURTH configuration
# that no launcher runs at all -- 300 px from a scorer constant, 0.25 from THIS script's
# floor -- and ranked 94 checkpoints on it. Issue #129.
#
# So every configuration lives in ONE importable object now, each naming the launcher
# that produces it, and this script asks for its own. The same object is what
# `detector/score_crossday.py` scores through; that script has no default and REFUSES a
# configuration no launcher runs unless given a reason it then records. There is no
# longer a copy here that could drift.
#
# The other three launchers are on the robot, not in this repository, so nothing here
# can make them ask. They are declared in inference_profile.py by hand and checked
# against `dashboard/run-profile.example.json`, which is the one copy of run-smoke.sh's
# invocation this repository holds.
#
# ⚠️ Changing the profile changes what the robot does, and needs a live run to justify.
# See the warning on `GO2_PEER_SUPERVISED` in inference_profile.py for what 224 is
# measured to cost and to buy; the axis is non-monotonic and neither result generalises.
INFERENCE_ARGV="$(python3 \
    "${TREE}/robot-stack/unitree/go2/visual_nav/inference_profile.py" \
    --argv go2-peer-supervised)"
INFERENCE=()
while IFS= read -r flag; do INFERENCE+=("${flag}"); done <<< "${INFERENCE_ARGV}"
# `set -e` already aborts on a failed command substitution above. This catches the other
# half: a profile that parsed, emitted, and left out a flag this script is responsible for
# passing -- which would silently fall back to `visual_nav.py`'s own default of 300, i.e.
# straight back into #129. Each flag is checked on its own so the check does not also
# depend on the order they come out in.
for required in --input-size --confidence --classes; do
  case " ${INFERENCE[*]} " in
    *" ${required} "*) ;;
    *) echo "inference_profile.py emitted no ${required}: ${INFERENCE_ARGV}" >&2
       exit 3 ;;
  esac
done
printf 'detector preprocessing:'; printf ' %s' "${INFERENCE[@]}"; printf '\n'

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
  # --input-size, --confidence and --classes, from inference_profile.py above. Not
  # spelled out here: a literal in this file is a copy, and a copy is what broke.
  "${INFERENCE[@]}"
  # The peer's own dimensions. Passing width explicitly matters: without it the width
  # prior comes from a standing adult's aspect ratio and a vertically clipped box ranges
  # the peer at 0.09-0.14 m, inside the robot's own footprint.
  --obstacle-height 0.514 --obstacle-width 0.31
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
