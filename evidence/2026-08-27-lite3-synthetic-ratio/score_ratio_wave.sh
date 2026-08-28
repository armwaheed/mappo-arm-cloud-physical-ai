#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Score every checkpoint of every arm at every deployed preprocessing, plus the incumbent.
#
# ⚠️ THE INSTRUMENT IS WAVE 7'S, DELIBERATELY UNCHANGED. This runs
# `evidence/2026-08-27-lite3-training-set/score_checkpoints.py` -- the same file, not a
# copy and not a fork -- because wave 8's whole argument is a comparison against wave 7's
# table, and swapping the scorer between the two waves would put an instrument change
# inside the comparison. `audit.py` in this directory checks that scorer's hand-copied
# PROFILES table against `inference_profile.py` itself, so the copy is a CHECKED copy
# rather than a trusted one.
#
# ⚠️ A FRESH `cv2.dnn.Net` PER (CHECKPOINT, PROFILE). That is the scorer's own rule and the
# reason the loop below is over profiles on the OUTSIDE and never reuses a process across
# two squares: `cv2.dnn.Net` retains state across an input-size change, and this project
# measured 300-after-224 at 42/60 where a fresh net on the same weights measured 41/60.
#
# ⚠️ THE TWO COLUMNS ARE NOT THE SAME KIND OF NUMBER. `lite3` is SAME-SESSION -- a held-out
# time block of the same six tripod shots, one room, one morning, 0.0-1.0 px of camera
# motion. `person` is CROSS-DAY: the 2026-08-20 Go2 manifest, another day and another
# building. Only the second has a day boundary in it, and it is the gate.
set -uf
ROOT="$HOME/lite3-ratio-20260827"
W7="$HOME/lite3ds-20260827"                  # wave 7's dataset and scorer. READ ONLY.
PY="$HOME/cvenv/bin/python"
OUT="$ROOT/scored"; mkdir -p "$OUT"
PROTO="$HOME/ssdft/base/mnssd22.prototxt"
PERSON="$HOME/ssdft/code/peer_crossday_20260820.json"
FRAMES="$HOME/ssdft/eval/frames"
PROFILES="go2-peer-supervised go2-navigator-default go2-run-smoke"
RUNS="r1x1_224 r1x3_224 r1x9_224 r1x1_300 r1x9_300"

score() {   # score <models-glob> <profile> <out>
  "$PY" "$W7/score_checkpoints.py" --proto "$PROTO" --models "$1" \
    --lite3-manifest "$W7/lite3_eval_20260827.json" --lite3-frames "$W7/images" \
    --person-manifest "$PERSON" --person-frames "$FRAMES" \
    --preprocessing "$2" --out "$3"
}

for profile in $PROFILES; do
  # The incumbent is re-measured here rather than cited from wave 7, so this directory's
  # gate floor comes out of this directory's own run. It is deterministic and it agrees.
  score "$HOME/ssdft/base/mnssd22.caffemodel" "$profile" \
        "$OUT/incumbent_$profile.json" > "$OUT/incumbent_$profile.log" 2>&1 &
  for run in $RUNS; do
    score "$ROOT/runs/$run/epoch*.caffemodel" "$profile" \
          "$OUT/scored_${run}_$profile.json" > "$OUT/scored_${run}_$profile.log" 2>&1 &
  done
  wait
  echo "scored every arm at $profile"
done
echo LITE3_RATIO_SCORING_DONE
