#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pull live frames from the Go2's frame server. READ-ONLY: HTTP GET and nothing else.
# No SSH, no motion, no lease. `robot-stack/SAFETY.md` does not apply because nothing
# here can move a leg -- but the robot is shared, so do not run this while somebody
# else owns the camera.
#
# The server is `dashboard/go2_frame_server.py`, running on the robot itself: the latest
# JPEG at / and a status document at /status. It caps at ~12 Hz, and this samples on
# `seq` changing rather than on a timer -- so a stalled pump produces FEWER frames
# rather than N identical copies of one, which would silently inflate any denominator
# computed over the result.
#
#   ./capture_live.sh [OUTDIR] [COUNT]
#
# The 149 frames behind this directory's README were captured this way on 2026-08-26
# and are NOT committed -- 27 MB of JPEG. `live_detections.json` beside this file is
# the network's output over them, which is what the README's numbers are computed from.
set -u

HOST="${HOST:-http://192.168.123.18:8801}"
OUT="${1:-frames}"
WANT="${2:-150}"

mkdir -p "$OUT"
last=-1
n=0
for _ in $(seq 1 900); do
  seq_now=$(curl -s --max-time 5 "$HOST/status" |
            sed -n 's/.*"seq": *\([0-9]*\).*/\1/p')
  if [ -n "$seq_now" ] && [ "$seq_now" != "$last" ]; then
    target=$(printf "%s/f%05d_seq%s.jpg" "$OUT" "$n" "$seq_now")
    if curl -s --max-time 8 "$HOST/" -o "$target" && [ -s "$target" ]; then
      last=$seq_now
      n=$((n + 1))
    fi
  fi
  [ "$n" -ge "$WANT" ] && break
  sleep 1
done
echo "captured $n frames into $OUT"

# A truncated JPEG is a capture artefact, not a detector input. One of the 150 frames
# behind this README ended mid-scan and was dropped, which is why the denominator is 149.
#
# `tr -d ' \n'` is load-bearing and is not tidying. `od -An -tx1` pads its columns with
# TWO spaces, so the obvious `grep -q 'ff d9'` matches NOTHING and reports every frame --
# including every intact one -- as truncated. A check that always fires is exactly as
# useless as one that never does, and this one was written the wrong way first.
bad=0
for f in "$OUT"/*.jpg; do
  end=$(tail -c2 "$f" | od -An -tx1 | tr -d ' \n')
  [ "$end" = "ffd9" ] || { echo "truncated: $f"; bad=$((bad + 1)); }
done
echo "truncated frames: $bad"
