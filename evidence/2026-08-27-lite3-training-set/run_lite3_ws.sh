#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lite3 weak-supervision wave. Three runs, one variable moved at a time.
#
# THE RECIPE IS NOT INVENTED. `k_full_pseudo03` is the best detector this project has
# produced -- 89% recall at 12% false alarms against the shipped weights' 68%/56%, keeping
# 17 of 22 people, at epoch 22. Its hyperparameters are the constant here:
#
#     --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1
#
# and note that k ran with NO augmentation flags. That matters, because wave 6's own
# ablation reverses wave 5's sign: `t_pseudo03_aug` is k with --motion-blur/--sensor-noise/
# --composite added and it LOSES 4-13 people at every matched epoch. Person retention is
# the gate every candidate has failed, so the flags are tested here rather than assumed.
#
# WHAT EACH RUN ISOLATES.
#   a  283 real positives only .............. the contemporaneous paired control
#   b  a + 2,542 offline synthetic (1:9) ..... does the synthetic half help?
#   c  b, plus the wave-6 augmentation flags . does the online recipe help on TOP of that?
#
# The positives are 131 owlv2+sam2 boxes over the 456 DISTINCT VIEWS in 5,854 frames, plus
# a +/-1-frame ride-along measured at 0.954 median IoU. 44 of 44 hand-checked boxes are on
# their subject. None of that changes the ceiling: one room, one morning, 13 minutes.
#
# `a` is a real control and not a citation: wave 5 established that these operators call
# rng.random() even at probability 0, so no earlier run is byte-reproducible under this
# code and a cited number would not be a control.
#
# ⛔ DO NOT run ~/sweep_all.py. Its on-disk version scores ONE run over 40 epochs and
# overwrites ~/sweep_all.json unconditionally, destroying the 64-row archive that produced
# every number in the wave-5 and wave-6 headers. The version that built that archive no
# longer exists.
set -uf
cd "$HOME/ssdft/code" || exit 1
export PYTHONPATH="$HOME/ssdft/pylibs"
PY="$HOME/cvenv/bin/python"
D="$HOME/lite3ds-20260827"; B="$HOME/ssdft/base"; R="$HOME/lite3ds-20260827/runs"
mkdir -p "$R"

run() {   # run <name> <labels> <extra-flags...>
  local name=$1 labels=$2; shift 2
  # shellcheck disable=SC2086
  "$PY" finetune_ssd.py \
    --proto "$B/mnssd22.prototxt" --model "$B/mnssd22.caffemodel" \
    --labels "$D/$labels" --images "$D/images" \
    --negatives-glob "$D/images/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 40 --batch-size 24 --workers 6 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    "$@" --out-dir "$R/$name" > "$R/$name.log" 2>&1
}

run a_ws_real       lite3_train_20260827.json &
run b_ws_synth      lite3_train_aug_20260827.json &
run c_ws_synth_aug  lite3_train_aug_20260827.json  \
      --motion-blur 0.5 --sensor-noise 0.5 --composite 0.3 &
wait
echo LITE3_WS_DONE
