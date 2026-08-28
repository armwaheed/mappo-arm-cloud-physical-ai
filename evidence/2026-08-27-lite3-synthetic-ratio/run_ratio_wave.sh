#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lite3 wave 8: does the real:synthetic RATIO decide person retention, and does training at
# the square the peer launcher opens at change anything?
#
# WHAT WAVE 7 LEFT. `evidence/2026-08-27-lite3-training-set` scored three runs at three
# configurations and its ablation is monotone in both directions at every resolution: each
# augmentation step adds `lite3` hits and removes people. `a`, the real-only control, is the
# only arm at either resolution that keeps people. Resolution is NOT the variable that does
# it -- the people collapse at 300 as well (29 -> 18 -> 0). What separates `a` from `b` is
# 283 real positives against 2,542 synthetic: 1 : 9.0.
#
# WAVE-6 FLAGS ARE OFF IN EVERY ARM. --motion-blur/--sensor-noise/--composite cost the most
# people in every prior run, and adding them here would confound the ratio. They are also
# not in the committed trainer -- see the README's note on the Spark's divergent copy.
#
# ⚠️ TWO VARIABLES, AT THE FOUR CORNERS AND ONE MIDPOINT.
#
#   arm            ratio  px   epochs  positives  steps/epoch  total steps
#   r1x1_224        1:1   224    144       566        36          5,184
#   r1x3_224        1:3   224     87     1,132        60          5,220
#   r1x9_224        1:9   224     40     2,825       130          5,200   <- wave 7's set
#   r1x1_300        1:1   300    144       566        36          5,184
#   r1x9_300        1:9   300     40     2,825       130          5,200   <- wave 7's `b`
#
# ⛔ THE EPOCH COUNTS ARE NOT A TYPO AND NOT A KNOB. Wave 7 ran 283-positive and
# 2,825-positive arms for the same 40 epochs, so its `a` took a tenth of the gradient steps
# its `b` did and "real only" and "fewer steps" moved together. Here every arm takes ~5,200
# steps (within 0.7%), and CosineAnnealingLR anneals over each arm's own epoch count, so the
# learning rate as a function of STEP is the same curve in all five. Composition of the
# training set is then the only thing that differs. Each arm's own epoch 40 is still on
# disk, so the equal-EPOCH reading is available too -- it is just not the headline.
#
# `r1x9_300` is a contemporaneous control and not a citation of wave 7's `b`. It runs the
# COMMITTED detector/finetune_ssd.py; the Spark's copy carries three augmentation operators
# that were never committed, and those operators call rng.random() even at probability 0,
# so `b` is not byte-reproducible under this code. Re-running it now, beside the arms it is
# a control for, is the only way its number means anything here.
set -uf
ROOT="$HOME/lite3-ratio-20260827"
cd "$ROOT/tree/detector" || exit 1
export PYTHONPATH="$HOME/ssdft/pylibs"
PY="$HOME/cvenv/bin/python"
D="$HOME/lite3ds-20260827"          # wave 7's dataset. READ ONLY -- nothing here writes to it.
B="$HOME/ssdft/base"
R="$ROOT/runs"
mkdir -p "$R"

run() {   # run <name> <labels> <epochs> <input-size>
  local name=$1 labels=$2 epochs=$3 size=$4
  "$PY" finetune_ssd.py \
    --proto "$B/mnssd22.prototxt" --model "$B/mnssd22.caffemodel" \
    --labels "$ROOT/$labels" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs "$epochs" --input-size "$size" \
    --batch-size 24 --workers 4 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    --out-dir "$R/$name" > "$R/$name.log" 2>&1
}

# --workers 4, not wave 7's 6: five concurrent runs on a 20-core host would otherwise ask
# for 30 loader processes. Worker count does not change what is trained -- the per-sample
# augmentation RNG is seeded from (seed, index), not from the worker.
run r1x1_224 lite3_train_r1x1_20260827.json 144 224 &
run r1x3_224 lite3_train_r1x3_20260827.json  87 224 &
run r1x9_224 lite3_train_aug_20260827.json   40 224 &
run r1x1_300 lite3_train_r1x1_20260827.json 144 300 &
run r1x9_300 lite3_train_aug_20260827.json   40 300 &
wait
echo LITE3_RATIO_WAVE_DONE
