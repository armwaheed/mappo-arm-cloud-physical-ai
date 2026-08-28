#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ⛔ IS THIS TRAINER REPRODUCIBLE AT A FIXED SEED? NO, AND THAT IS THIS WAVE'S MAIN FINDING.
#
# Replicating one arm gave median person retention of 10.0 / 15.5 / 21.0 at 300 px / 0.45
# and 1 / 0 / 108 gate-clearing epochs -- a spread wider than any difference this wave was
# built to measure. The obvious explanation is the seed. This checks it, and the seed is not
# the explanation: the runs below differ in NOTHING, `--seed 0` included.
#
# Run N byte-identical one-epoch trainings and compare (a) the losses and (b) the md5 of the
# exported weights. `matched/batch` is printed too, because it is the control: it is a
# property of the DATA pipeline -- which images, in which order, with which augmentation --
# and if it is identical across runs while the weights are not, the nondeterminism is in the
# GPU compute and not in the sampler or the augmentation RNG.
#
# One epoch, not 144, because the question is whether two identical runs diverge at all. A
# difference at epoch 1 compounds; that is what the 144-epoch replicates then show.
#
# ⚠️ WHAT THIS MEANS FOR EVERY EARLIER WAVE. Wave 7 records that "these operators call
# rng.random() even at probability 0, so no earlier run is byte-reproducible under this
# code", which attributes irreproducibility to the RNG stream and implies that fixing the
# stream would fix it. It would not. Same code, same seed, same machine, same data order,
# different weights. Every wave in this project has compared n=1 arms.
# ⚠️ REDIRECT THIS TO A TEMPORARY FILE AND `mv` IT INTO PLACE. Two copies of this script
# writing to one path with `>` do not conflict and do not error: each holds its own offset,
# and the shorter one leaves a block of NUL bytes where the other's output was. The file
# then has a plausible size and a plausible head, and fails only at `json.load`. That
# happened here, when a foreground run was killed by a client timeout and a second copy was
# launched over the top of it.
set -uf
ROOT="$HOME/lite3-ratio-20260827"
cd "$ROOT/tree/detector" || exit 1
export PYTHONPATH="$HOME/ssdft/pylibs"
PY="$HOME/cvenv/bin/python"
D="$HOME/lite3ds-20260827"; B="$HOME/ssdft/base"
RUNS=${1:-3}

echo "{"
echo " \"what\": \"N byte-identical one-epoch runs of the same command, --seed 0 in all\","
echo " \"runs\": ["
for i in $(seq 1 "$RUNS"); do
  out=$(mktemp -d)
  line=$("$PY" finetune_ssd.py \
    --proto "$B/mnssd22.prototxt" --model "$B/mnssd22.caffemodel" \
    --labels "$ROOT/lite3_train_r1x1_20260827.json" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 1 --input-size 224 --seed 0 --batch-size 24 --workers 4 \
    --learning-rate 1e-3 --freeze-through= --backbone-lr-scale 0.5 \
    --pseudo-labels 0.3 --distil 0.1 --out-dir "$out" 2>&1 | grep "^epoch")
  conf=$(echo "$line" | awk '{print $4}')
  loc=$(echo "$line" | awk '{print $6}')
  matched=$(echo "$line" | awk '{print $10}')
  md5=$(md5sum "$out/epoch001.caffemodel" | awk '{print $1}')
  comma=","; [ "$i" = "$RUNS" ] && comma=""
  echo "  {\"run\": $i, \"conf\": $conf, \"loc\": $loc, \"matched_per_batch\": $matched, \"weights_md5\": \"$md5\"}$comma"
  rm -rf "$out"
done
echo " ]"
echo "}"
