#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ⛔ THIS EXISTS TO FALSIFY THIS WAVE'S OWN HEADLINE.
#
# `r1x1_224` produced the first checkpoint in four waves to clear the person gate at the
# square production launches: epoch 037, 15/36 lite3, **25 of 284 people against the
# incumbent's 25**. It clears by exactly zero. And it is rare inside its own run --
# 4 of 144 epochs reach 25 people, 2 reach 26, and 1 reaches 27, at which point the best
# lite3 among survivors falls from 15/36 to 11/36 to 1/36.
#
# Two different things are being called "noise" and only one of them applies here.
#
#   the SCORE of a checkpoint   is deterministic. cv2 runs a fixed graph over fixed frames;
#                               re-scoring epoch 037 returns 25 every time, forever.
#   the RUN that produced it    is not. This project's +/-1-3 person band is a statement
#                               about re-running training, and a gate cleared by +0 is
#                               inside it.
#
# So the question the tables cannot answer is: does the RECIPE clear the gate, or did one
# epoch of one seed? Two more seeds of the same arm answer it. Nothing else changes -- same
# manifest, same square, same 144 epochs, same hyperparameters, same committed trainer.
#
# Read the result as a count, not an average: N of 3 seeds produced a checkpoint at or above
# 25 people, and here is the best lite3 each of them reached under the gate. If only the
# original seed does, the headline is a lucky epoch and this file is why we know.
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
    --labels "$ROOT/lite3_train_r1x1_20260827.json" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 144 --input-size 224 --seed "$seed" \
    --batch-size 24 --workers 8 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    --out-dir "$R/r1x1_224_s$seed" > "$R/r1x1_224_s$seed.log" 2>&1
}

run 1 &
run 2 &
wait
echo LITE3_SEED_REPLICATES_DONE
