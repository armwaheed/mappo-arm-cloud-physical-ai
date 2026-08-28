#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ⛔ THE 300 px TWIN OF run_seed_replicates.sh, AND IT EXISTS BECAUSE THAT ONE WORKED.
#
# Replicating `r1x1_224` falsified this wave's 224 px result outright: of three seeds of one
# arm, seed 0 put 4 of 144 checkpoints at or above the incumbent's 25 people and reached
# 15/36 lite3 under the gate, seed 2 reached 12/36 -- and **seed 1 produced nothing but
# epoch 001**, i.e. the base weights. A pass that one seed in three does not reproduce is a
# property of the seed, not of the recipe.
#
# `r1x1_300` is the arm that made this wave's headline: 17/36 lite3 at 25 of 284 people, at
# 26 of its 144 epochs, surviving a person floor three above the incumbent. Every one of
# those numbers is n=1, and the only arm that has been replicated so far turned out to be
# seed-dependent. So the headline gets the same test that killed the other claim, and it
# gets it before it is published rather than after someone tries to deploy it.
#
# A separate file rather than an argument to run_seed_replicates.sh: that script's bytes are
# the record of how the 224 pair was produced, and editing it after the fact would make the
# committed copy a description of a run that never happened.
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
    --labels "$ROOT/lite3_train_r1x1_20260827.json" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 144 --input-size 300 --seed "$seed" \
    --batch-size 24 --workers 8 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    --out-dir "$R/r1x1_300_s$seed" > "$R/r1x1_300_s$seed.log" 2>&1
}

run 1 &
run 2 &
wait
echo LITE3_SEED300_REPLICATES_DONE
