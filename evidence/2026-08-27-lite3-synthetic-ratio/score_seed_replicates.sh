#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Score the two seed replicates of r1x1_224, with the same instrument as everything else.
# Separate from score_ratio_wave.sh rather than an edit to it: that script's bytes are the
# record of how the eighteen files beside it were produced, and editing it after the fact
# would make the committed copy a description of a run that never happened.
#
# The question is a COUNT, not an average: how many of three seeds put a checkpoint at or
# above the incumbent's 25 people at 224 px / 0.25, and what is the best lite3 each reaches
# under that floor. Seed 0 did, by exactly +0, at 4 of its 144 epochs.
set -uf
ROOT="$HOME/lite3-ratio-20260827"
W7="$HOME/lite3ds-20260827"
PY="$HOME/cvenv/bin/python"
OUT="$ROOT/scored"; mkdir -p "$OUT"

for profile in go2-peer-supervised go2-navigator-default go2-run-smoke; do
  for run in r1x1_224_s1 r1x1_224_s2; do
    "$PY" "$W7/score_checkpoints.py" --proto "$HOME/ssdft/base/mnssd22.prototxt" \
      --models "$ROOT/runs/$run/epoch*.caffemodel" \
      --lite3-manifest "$W7/lite3_eval_20260827.json" --lite3-frames "$W7/images" \
      --person-manifest "$HOME/ssdft/code/peer_crossday_20260820.json" \
      --person-frames "$HOME/ssdft/eval/frames" \
      --preprocessing "$profile" --out "$OUT/scored_${run}_$profile.json" \
      > "$OUT/scored_${run}_$profile.log" 2>&1 &
  done
  wait
  echo "scored both replicates at $profile"
done
echo LITE3_SEED_SCORING_DONE
