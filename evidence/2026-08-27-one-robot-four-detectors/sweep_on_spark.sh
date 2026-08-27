#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Every candidate checkpoint, the incumbent, and the 22-class starting point, scored on the
# 2026-08-20 cross-day day at EVERY configuration a launcher of this robot produces -- and
# at the one the 2026-08-26 checkpoint sweep used, which no launcher produces.
#
#   go2-peer-supervised     224 px  0.25   deploy/run-peer-supervised.sh
#   go2-run-smoke           300 px  0.45   run-smoke/berth/chair, i.e. the 89-run corpus
#   go2-navigator-default   300 px  0.40   a bare visual_nav.py
#   mobilenet-ssd-trained   300 px  0.25   run by nothing; what the sweep scored
#
# The three at 300 px come from ONE forward pass per checkpoint, so a difference between
# those rows is the score floor and cannot be an inference difference. The 224 pass gets
# its own freshly loaded Net -- `cv2.dnn.Net` keeps state across a change of input size,
# see the comment on `score_model` in detector/score_crossday.py.
#
# THIS IS WHERE IT HAS TO RUN, AND WHY. The 800 candidate .caffemodel files are on the
# training host and are not vendored anywhere a clone can reach; issue #129's own evidence
# records four routes to them, all dead. Nothing here touches a robot.
#
#   scp -r detector robot-stack <host>:i129/ && scp sweep_on_spark.sh <host>:i129/
#   ssh <host> 'bash i129/sweep_on_spark.sh'
#
# Override any of these; the defaults are the paths used on 2026-08-26.
#
#   TREE        where detector/ and robot-stack/ were copied to on the host
#   RUNS        directory of <run>/epoch<NN>.caffemodel
#   PROTO       the 22-class prototxt every candidate was trained against
#   INCUMBENT   directory holding MobileNetSSD_deploy.{prototxt,caffemodel}
#   FRAMES      284 JPEGs, <clip>_<index>.jpg, decoded at the quality the manifest names
#   OUT         where the three JSON files go
#   PY          an interpreter with OpenCV 4.x (OpenCV 5 removed readNetFromCaffe)
#
# ~55 min on a shared GB10, measured 2026-08-26: 800 checkpoints x two forward passes over
# 284 1080p frames, decoded once.
#
# ⚠️ A 40-frame warm benchmark on the same machine said 2.3x faster than this. Size a run
# off the real corpus, not off a benchmark whose working set fits in cache.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="${TREE:-${HOME}/i129}"
RUNS="${RUNS:-${HOME}/ssdft/runs}"
PROTO="${PROTO:-${HOME}/ssdft/base/mnssd22.prototxt}"
INCUMBENT="${INCUMBENT:-${HOME}/go2-peer-dataset-20260824/artifacts/models_robot}"
FRAMES="${FRAMES:-${HOME}/ssdft/eval/frames}"
OUT="${OUT:-${HERE}/sweep}"
PY="${PY:-${HOME}/cvenv/bin/python}"
mkdir -p "${OUT}"

# The reason is not decoration. `score_crossday.py` refuses a profile no launcher runs
# without one, and writes whatever is given here into the results file beside the numbers
# it qualifies -- so the sweep's 300 px / 0.25 row can never be quoted as a run.
WHY='300 px at 0.25 is the pair the 2026-08-26 checkpoint sweep scored through and no launcher runs; it is here so the published tables can be lined up against configurations that are real'

PROFILES=(
  --preprocessing go2-peer-supervised
  --preprocessing go2-run-smoke
  --preprocessing go2-navigator-default
  --preprocessing mobilenet-ssd-trained
  --allow-preprocessing-mismatch "${WHY}"
)

cd "${TREE}/detector"

echo "== the incumbent, at all four =="
"${PY}" score_crossday.py --frames-dir "${FRAMES}" \
    --proto "${INCUMBENT}/MobileNetSSD_deploy.prototxt" \
    --model "${INCUMBENT}/MobileNetSSD_deploy.caffemodel" \
    "${PROFILES[@]}" --out "${OUT}/incumbent.json"

# The starting point of every candidate, so that "what was not scored" is empty rather than
# one file. `mnssd22.caffemodel` is the incumbent's weights grown to 22 classes by
# detector/add_class.py with an untrained `go2wheel` head -- epoch 0 of all twenty runs. It
# is not a candidate and cannot be one, but leaving it out would leave a .caffemodel on this
# host that no table accounts for.
echo "== the 22-class starting point, at all four =="
"${PY}" score_crossday.py --frames-dir "${FRAMES}" --proto "${PROTO}" \
    --model "${PROTO%.prototxt}.caffemodel" \
    "${PROFILES[@]}" --out "${OUT}/base.json"

echo "== every candidate, at all four =="
"${PY}" score_crossday.py --frames-dir "${FRAMES}" --proto "${PROTO}" \
    --models "${RUNS}/*/epoch*.caffemodel" \
    --inventory-glob "${RUNS}/*/*.caffemodel" \
    "${PROFILES[@]}" --out "${OUT}/candidates.json"

echo "wrote ${OUT}/{incumbent,base,candidates}.json"
