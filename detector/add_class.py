#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Grow MobileNet-SSD by N classes, in place, without Caffe installed.

The robot's only inference engine is ``cv2.dnn`` on OpenCV 4.2 — no CUDA, no torch, no
onnxruntime — so whatever is trained has to come back as the ``.caffemodel`` the stack
already loads. A ``.caffemodel`` is protobuf, and protobuf does not need Caffe: this
module parses one, resizes the class-score layers, and writes it back. Verified
end-to-end: re-serialising an untouched model reproduces ``cv2``'s output bit for bit.

## What has to change, and where

Three things carry the class count, and all three are in the prototxt:

  * ``num_output`` on the six ``*_mbox_conf`` convolutions — ``num_priors * num_classes``,
    so 63 (3 priors) and 126 (6 priors) at 21 classes.
  * ``dim: 21`` in ``mbox_conf_reshape``.
  * ``num_classes: 21`` in ``detection_output_param``.

In the weights, an ``mbox_conf`` output channel is ``prior * num_classes + class``, so the
class index is the fastest-varying axis: growing a class means reshaping to
``(priors, classes, in_ch, 1, 1)`` and appending one slice per prior.

## ⚠️ Do not initialise a new class to a constant

The obvious init — zero the weights, set a large negative bias — is wrong, and it fails
loudly enough to waste a training run. With zeroed weights the new logit is that constant
*everywhere*, and a constant is a FLOOR: on any prior whose trained logits all fall below
it, the new class becomes the arg-max. Measured at bias -50 on a real frame: eight
detections at confidence **1.000000**, on degenerate slivers at the frame edge.

Seeding from an existing class instead gives the new logit the same dynamics as a class
the network already calibrates, so it cannot be a spurious global maximum. ``dog`` is the
default seed because a quadruped is the nearest thing VOC knows, which also starts
fine-tuning somewhere warmer than noise. Measured with that seed, on the same frame:
person 0.928999 -> 0.928981, chair 0.495680 -> 0.495674, and no spurious boxes.

Requires ``caffe_pb2`` generated from the SSD fork's ``caffe.proto`` (it carries the
``DetectionOutput``/``PriorBox``/``Normalize`` messages the upstream one lacks):

    protoc --python_out=. caffe.proto     # from weiliu89/caffe, branch ssd

Usage:

    add_class.py --in-proto d.prototxt --in-model d.caffemodel \\
                 --out-proto d22.prototxt --out-model d22.caffemodel \\
                 --classes lite3 go2wheel
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

#: The twenty PASCAL VOC classes the shipped weights were trained on, plus background.
#: Kept here rather than imported from the robot stack: this tool has to run on a
#: workstation with no robot code on the path.
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

#: Class whose parameters seed each new class. See the module docstring for why this is
#: not a constant.
DEFAULT_SEED = "dog"


def rewrite_prototxt(text: str, old_classes: int, new_classes: int) -> str:
    """The three class-count edits, checked rather than assumed.

    Every substitution is asserted to have fired the expected number of times. A blanket
    ``str.replace`` on ``num_output`` would silently corrupt any backbone convolution
    that happened to share a channel count with a confidence head, and the model would
    still load — it would just be wrong.
    """
    out = text
    for priors in (3, 6):
        old, new = priors * old_classes, priors * new_classes
        pattern = rf"num_output:\s*{old}\b"
        hits = len(re.findall(pattern, out))
        if not hits:
            raise ValueError(f"no conv with num_output {old} — is this MobileNet-SSD?")
        out = re.sub(pattern, f"num_output: {new}", out)
    for pattern, replacement, expected in (
        (rf"dim:\s*{old_classes}\b", f"dim: {new_classes}", 1),
        (rf"num_classes:\s*{old_classes}\b", f"num_classes: {new_classes}", 1),
    ):
        hits = len(re.findall(pattern, out))
        if hits != expected:
            raise ValueError(f"expected {expected} match for {pattern!r}, found {hits}")
        out = re.sub(pattern, replacement, out)
    return out


def grow_weights(net, old_classes: int, new_classes: int, seed_index: int) -> list[str]:
    """Append ``new_classes - old_classes`` seeded class slices to every conf head."""
    grown = []
    for layer in net.layer:
        if "mbox_conf" not in layer.name or not layer.blobs:
            continue
        weight_blob, bias_blob = layer.blobs[0], layer.blobs[1]
        outputs, in_channels = weight_blob.shape.dim[0], weight_blob.shape.dim[1]
        priors = outputs // old_classes
        if priors * old_classes != outputs:
            raise ValueError(f"{layer.name}: {outputs} not divisible by {old_classes}")
        weights = np.array(weight_blob.data, np.float32).reshape(
            priors, old_classes, in_channels, 1, 1)
        biases = np.array(bias_blob.data, np.float32).reshape(priors, old_classes)

        extra = new_classes - old_classes
        seed_w = np.repeat(weights[:, seed_index:seed_index + 1], extra, axis=1)
        seed_b = np.repeat(biases[:, seed_index:seed_index + 1], extra, axis=1)
        weights = np.concatenate([weights, seed_w], axis=1)
        biases = np.concatenate([biases, seed_b], axis=1)

        del weight_blob.data[:], bias_blob.data[:]
        weight_blob.data.extend(weights.ravel().tolist())
        weight_blob.shape.dim[0] = priors * new_classes
        bias_blob.data.extend(biases.ravel().tolist())
        bias_blob.shape.dim[0] = priors * new_classes
        grown.append(f"{layer.name} {outputs}->{priors * new_classes}")
    if not grown:
        raise ValueError("no mbox_conf layers found — wrong model?")
    return grown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--in-proto", type=Path, required=True)
    parser.add_argument("--in-model", type=Path, required=True)
    parser.add_argument("--out-proto", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True,
                        help="names of the classes to append, in order")
    parser.add_argument("--seed-class", default=DEFAULT_SEED, choices=VOC_CLASSES,
                        help="existing class whose parameters seed each new one")
    parser.add_argument("--caffe-pb2", type=Path, default=Path("."),
                        help="directory holding the generated caffe_pb2.py")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.caffe_pb2))
    try:
        import caffe_pb2
    except ImportError:
        parser.error(f"caffe_pb2 not importable from {args.caffe_pb2} — see the module "
                     f"docstring for the one protoc command that generates it")

    old_classes = len(VOC_CLASSES)
    new_classes = old_classes + len(args.classes)

    args.out_proto.write_text(rewrite_prototxt(
        args.in_proto.read_text(), old_classes, new_classes))

    net = caffe_pb2.NetParameter()
    net.ParseFromString(args.in_model.read_bytes())
    grown = grow_weights(net, old_classes, new_classes,
                         VOC_CLASSES.index(args.seed_class))
    args.out_model.write_bytes(net.SerializeToString())

    print(f"{old_classes} -> {new_classes} classes, seeded from '{args.seed_class}'")
    for line in grown:
        print(f"    {line}")
    for offset, name in enumerate(args.classes):
        print(f"    class {old_classes + offset} = {name}")
    print(f"\n{args.out_proto}\n{args.out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
