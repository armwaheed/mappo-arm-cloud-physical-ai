#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ⛔ THE OTHER END OF THE RATIO SWEEP, REPLICATED, BECAUSE THE FIRST REPLICATE MOVED MORE
# THAN THE CONDITION DID.
#
# Three seeds of `r1x1_224` gave median person retention of 12.5 / 6.0 / 16.0 at 224 px /
# 0.25, and 1 / 0 / 108 gate-clearing epochs at 300 px / 0.45. The seed moves the gate
# result by more than any condition in this wave moves it. That does not by itself threaten
# the RATIO finding -- 1:9's median is 3.5 and every 1:1 seed is above it -- but the
# comparison was n=3 against n=1, and the worst 1:1 seed is only 2.5 people clear of 1:9,
# which is inside this project's own +/-1-3 band.
#
# So the 1:9 arm gets the same three seeds, and the headline comparison becomes n=3 against
# n=3 at the square production launches. 40 epochs, not 144, because that is what 1:9 needs
# to reach ~5,200 steps -- the whole point of the epoch counts.
#
set -uf
ROOT="$HOME/lite3-ratio-20260827"
cd "$ROOT/tree/detector" || exit 1
export PYTHONPATH="$HOME/ssdft/pylibs"
PY="$HOME/cvenv/bin/python"
D="$HOME/lite3ds-20260827"; B="$HOME/ssdft/base"; R="$ROOT/runs"

run() {   # run <seed>
  local seed=$1
  "$PY" finetune_ssd.py \
    --proto "$B/mnssd22.prototxt" --model "$B/mnssd22.caffemodel" \
    --labels "$ROOT/lite3_train_aug_20260827.json" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 40 --input-size 224 --seed "$seed" \
    --batch-size 24 --workers 8 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    --out-dir "$R/r1x9_224_s$seed" > "$R/r1x9_224_s$seed.log" 2>&1
}

run 1 &
run 2 &
wait
echo LITE3_SEED1X9_REPLICATES_DONE
