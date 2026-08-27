#!/bin/bash
# Lite3 wave 1. The augmentation recipe is NOT invented here: it is the wave-5/6 Go2 set,
# read off ~/ssdft/run_wave6.sh -- --motion-blur 0.5 --sensor-noise 0.5 --composite 0.3,
# on top of SSD's stock photometric/expand/crop/flip which waves 1-4 already ran.
#
# `ctl` is a CONTEMPORANEOUS PAIRED CONTROL, not a citation of a Go2 run: the wave-5/6
# operators call rng.random() even at probability 0, so a run with them off is the only
# thing that isolates augmentation on THIS corpus.
#
# ⚠️ Both runs train on 168 frames that are ONE camera pose of ONE peer pose. Whatever
# they score on this clip is same-session and is an upper bound, not a generalisation.
set -uf
cd "$HOME/lite3-detector-20260827" || exit 1
export PYTHONPATH="$HOME/ssdft/pylibs"
PY="$HOME/cvenv/bin/python"
D="$HOME/lite3-detector-20260827"
N="$HOME/go2-peer-dataset-20260824"
R="$D/runs"; mkdir -p "$R"

run() {   # run <name> <aug: 0|1>
  local name=$1 aug=$2 extra=""
  [ "$aug" = "1" ] && extra="--motion-blur 0.5 --sensor-noise 0.5 --composite 0.3"
  # shellcheck disable=SC2086
  "$PY" code/finetune_ssd.py \
    --proto "$D/base_lite3_22.prototxt" --model "$D/base_lite3_22.caffemodel" \
    --labels "$D/lite3_shanghai_20260827.json" --images "$D/dataset" \
    --negatives-glob "$N/neg_*.jpg" --caffe-pb2 "$HOME/ssdft/pb" \
    --epochs 40 --batch-size 24 --workers 6 --learning-rate 1e-3 \
    --freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1 \
    $extra --out-dir "$R/$name" > "$R/$name.log" 2>&1
}

run a_lite3_aug 1 &
run b_lite3_ctl 0 &
wait
echo LITE3_WAVE1_DONE
