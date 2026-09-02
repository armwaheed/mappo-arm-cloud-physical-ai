#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Friday demo, as one command: drive to the ArUco goal, avoid whatever is in the way,
# ask the room to clear in Chinese and then English when it cannot, and keep trying.
#
# WHAT THIS ENCODES, and why each part is not a default:
#
#   --camera-rectify   This camera is RECTILINEAR and the shared model is EQUIDISTANT.
#                      Without this every bearing is wrong by +8 deg at 40 deg off axis,
#                      which on 2026-08-31 collapsed a stationary box from 1.27 m to
#                      0.33 m across 63 deg of yaw. See WHITEPAPER Appendix A19.
#   --static-detect-*  Obstacles are ranged from their FLOOR CONTACT POINT, so nothing
#                      needs a size prior, a colour marker or a trained class. It is the
#                      only mechanism here that can handle "an attendee drops a bag in
#                      the lane".
#   the class list     EVERY VOC class except person. A Lite3 cannot be reliably detected
#                      as a Lite3 -- measured: zero response from MobileNet-SSD over 38
#                      frames -- so whatever the network decides a quadruped is (YOLO
#                      offered "motorcycle"), it is mapped as an obstacle anyway. People
#                      are excluded here because they are TRACKED instead, which is
#                      stronger: a landmark is routed around, a person is stopped for.
#   mission.py         Restarts the run when it ends without arriving, re-acquiring the
#                      goal each time, and speaks while it waits.
#
# ⚠️ AUDIO GOES THROUGH PULSEAUDIO, NOT THE RAW CARD. PulseAudio holds this codec
# exclusively, so `plughw:0,0` returns "Device or resource busy"; `pulse` is the route
# that works. And the account must be in the `audio` GROUP. The account running this must be in it, or the sound
# device cannot be opened, PulseAudio falls back to its auto_null sink, and aplay writes
# into a black hole and EXITS 0 -- every check passes and the robot is silent. Fix once
# with `sudo usermod -aG audio "$USER"` and start a new session. mission.py probes the
# device before each mission and says so if nothing will be audible.
#
# ⚠️ This platform reports NO motor temperatures. Retries are bounded and there is a real
# cooldown between them; that is the only thermal margin there is. Do not raise the caps
# without someone watching the robot. robot-stack/SAFETY.md governs all of this.
#
#   ./run-venue-demo.sh            # live
#   DRY=1 ./run-venue-demo.sh      # same configuration, no motion
set -euo pipefail

STAGE="${STAGE:-$HOME/mappo-lite3-stage}"
TAG="${TAG:-mappo-arm-cloud-physical-ai-lite3-20260901-v13}"
RELEASE="$STAGE/releases/$TAG"
RUN_ID="${RUN_ID:-venue-$(date -u +%Y%m%dT%H%M%SZ)}"

# WAS all nineteen VOC classes. Measured in the rehearsal room on 2026-09-02, that plus
# the 0.10 static score floor produced a robot that stood still asking the room to clear:
# bare carpet scored `pottedplant` 0.27 at 0.61 m and `chair` 0.25 at 0.42 m, both inside
# the planner's 1.20 m soft gap, so the veto never lifted. person_detector's own
# STATIC_CLASSES comment says why the wide list is wrong at a lowered floor -- it is there
# to discard "the whole-wall aeroplane and train slabs".
#
# This is that tuned set, minus pottedplant (the measured false positive above), plus
# motorbike for ONE reason: a Lite3 lying down detects as motorbike at 0.66, and a robot
# parked in the lane has to be avoided. 0.66 is far above the 0.25-0.27 phantom noise.
#
# An ARRAY, not a string: these are separate argv entries for an nargs="+" flag, so the
# word splitting is wanted. Quoting the string instead would pass them all as ONE class
# name, which argparse accepts and the detector then never matches.
OBSTACLE_CLASSES=(chair sofa diningtable tvmonitor motorbike)

LIVE=(--live --operator-ready)
if [ -n "${DRY:-}" ]; then LIVE=(); fi

cd "$RELEASE/robot-stack/deep_robotics/lite3/visual_nav"
# shellcheck disable=SC1091  # the venv is created on the robot, not in this repository
. "$STAGE/venv/bin/activate"
export PYTHONPATH="$STAGE/python"
export MAPPO_ROBOT_HOST=1

echo "[venue] release $TAG"
echo "[venue] run id  $RUN_ID"

exec python3 mission.py \
  --voice-dir "$STAGE/voice" \
  --voice-device "${VOICE_DEVICE:-pulse}" \
  --patience "${PATIENCE:-4}" \
  --cooldown "${COOLDOWN:-25}" \
  --max-attempts "${ATTEMPTS:-8}" \
  --max-total-seconds "${TOTAL:-900}" \
  -- python3 mappo_drive.py \
     --package "$RELEASE/policy" \
     --policy-mode supervised --policy-scale 4.0 \
     --execution-supervisor turn-drive \
     --policy-gait-floor 0.30 \
     --goal-detect-scale 1.0 \
     --heading-servo goal \
     --camera-source "${CAMERA:-rtsp://127.0.0.1:8554/test}" \
     --model-dir "$STAGE/models/mobilenet-ssd" \
     --camera-rectify "$STAGE/calibration/lite3_front_camera_rectify_20260901.json" \
     --calibration "$STAGE/calibration/lite3_front_camera_equidistant_20260901.json" \
     --marker-size 0.14 \
     --arrive "${ARRIVE:-0.5}" \
     --static-detect --static-detect-ground \
     --static-detect-classes "${OBSTACLE_CLASSES[@]}" \
     --static-detect-radius 0.35 \
     --static-detect-confidence "${STATIC_CONFIDENCE:-0.35}" \
     --static-detect-pitch-error-deg 1.5 \
     --static-detect-max-range-error 0.20 \
     --locomotion-transport axis \
     --axis-profile "$STAGE/calibration/lite3_axis_profile_LITE3-A-executable.json" \
     --state-bind 127.0.0.1 \
     "${LIVE[@]}" \
     --gait-floor 0.30 --actuator-gain 1.07 --robot-radius 0.40 \
     --max-vx 0.55 --max-vy 0 --max-wz 0.90 \
     --accept-no-motor-temperatures --max-seconds "${SECS:-90}" \
     --record "$STAGE/evidence/$RUN_ID.mp4" \
     --record-raw "$STAGE/evidence/$RUN_ID-raw.mp4" \
     --telemetry "$STAGE/evidence/$RUN_ID.jsonl"
