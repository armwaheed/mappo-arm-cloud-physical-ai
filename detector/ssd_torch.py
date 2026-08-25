#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A PyTorch mirror of the MobileNet-SSD the robot actually runs, built from its prototxt.

Why a mirror rather than a fresh model: the Jetson's only inference engine is ``cv2.dnn`` on
OpenCV 4.2, so whatever is trained has to come back as the ``.caffemodel`` the stack already
loads. Training a differently-shaped network and hoping the weights transfer is not a plan.
This reads the deployed prototxt, builds the same graph in torch, and loads the same
weights, so the thing being trained IS the thing being deployed.

WHY THIS IS FEASIBLE AT ALL: the deployed model has its BatchNorm FOLDED into the
convolutions — 127 layers against the 194 of the published chuanqi305 release, and no
BatchNorm or Scale layers anywhere. What remains is 47 convolutions (depthwise ones
expressed with ``group``), 35 ReLUs, and the detection head. That is a graph a few hundred
lines can mirror honestly; an unfolded model would need the BN statistics handled too.

⚠️ THE MIRROR IS WORTHLESS UNLESS IT REPRODUCES cv2's OUTPUT, and a shape check will not
tell you that. :func:`verify_against_cv2` asserts the feature maps and the raw head outputs
match the deployed network to a tolerance, on real input. Building a mirror that is subtly
wrong — a missing pad, a transposed group, the wrong ReLU — produces a network that trains
happily and deploys to something else. That failure has already happened once in this
directory, when features were read from the pre-BatchNorm blob rather than the ReLU, so the
check is not hypothetical.

Only the graph up to the six ``*_mbox_loc`` / ``*_mbox_conf`` heads is mirrored. PriorBox,
Permute, Flatten, Concat, Reshape, Softmax and DetectionOutput are inference-side assembly:
priors depend only on the input size and can be read once from ``cv2``, and the rest is
reshaping that the loss does for itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

#: Square network input, matching what the robot runs.
INPUT_SIZE = 300

#: Preprocessing baked into the published weights: ``(pixel - 127.5) / 127.5``.
SSD_SCALE, SSD_MEAN = 1.0 / 127.5, 127.5

#: Feature maps that feed the detection heads, in the order the heads are concatenated.
#: Read off the prototxt rather than assumed; :func:`build` checks each one exists.
HEAD_SOURCES = ("conv11", "conv13", "conv14_2", "conv15_2", "conv16_2", "conv17_2")


@dataclass
class LayerSpec:
    """One Caffe layer, reduced to what a torch mirror needs."""

    name: str
    kind: str
    bottom: list = field(default_factory=list)
    top: list = field(default_factory=list)
    num_output: int = 0
    kernel: int = 1
    stride: int = 1
    pad: int = 0
    groups: int = 1


def parse_prototxt(text: str) -> list:
    """Every Convolution and ReLU in the prototxt, in declaration order.

    Deliberately a small hand parser rather than protobuf text-format: the point is to fail
    loudly on anything unexpected. A tolerant parser that silently skips a layer type it
    does not know would build a mirror missing a layer, and the verification would then be
    reporting on a graph nobody described.
    """
    specs = []
    for block in re.findall(r"layer\s*\{(.*?)\n\}", text, re.S):
        kind = re.search(r'type:\s*"([^"]+)"', block)
        name = re.search(r'name:\s*"([^"]+)"', block)
        if not kind or not name:
            continue
        if kind.group(1) not in ("Convolution", "ReLU"):
            continue

        def one(pattern, default=None, _block=block):
            found = re.search(pattern, _block)
            return int(found.group(1)) if found else default

        specs.append(LayerSpec(
            name=name.group(1), kind=kind.group(1),
            bottom=re.findall(r'bottom:\s*"([^"]+)"', block),
            top=re.findall(r'top:\s*"([^"]+)"', block),
            num_output=one(r"num_output:\s*(\d+)", 0),
            kernel=one(r"kernel_size:\s*(\d+)", 1),
            stride=one(r"stride:\s*(\d+)", 1),
            pad=one(r"pad:\s*(\d+)", 0),
            groups=one(r"group:\s*(\d+)", 1)))
    return specs


class CaffeMirror(nn.Module):
    """The deployed graph, in torch, keyed by Caffe blob name.

    Blob-keyed rather than a ``Sequential`` because Caffe layers write IN PLACE: ``conv11``,
    ``conv11/bn``, ``conv11/scale`` and ``conv11/relu`` all name the blob ``conv11``, and the
    head reads that blob AFTER the ReLU. A sequential mirror has no way to express "the
    value of this name at this point", which is exactly the distinction that broke the
    frozen-feature extraction before.
    """

    def __init__(self, specs: list) -> None:
        super().__init__()
        self.specs = specs
        self.convs = nn.ModuleDict()
        for spec in specs:
            if spec.kind != "Convolution":
                continue
            in_channels = self._infer_in_channels(spec, specs)
            self.convs[spec.name.replace("/", "__")] = nn.Conv2d(
                in_channels, spec.num_output, spec.kernel, spec.stride, spec.pad,
                groups=spec.groups, bias=True)

    @staticmethod
    def _infer_in_channels(spec: LayerSpec, specs: list) -> int:
        """Channels feeding ``spec``, from whichever layer last wrote its bottom blob."""
        if spec.bottom and spec.bottom[0] == "data":
            return 3
        for earlier in reversed(specs[:specs.index(spec)]):
            if earlier.kind == "Convolution" and spec.bottom[0] in earlier.top:
                return earlier.num_output
        raise ValueError(f"cannot infer input channels for {spec.name}")

    def forward(self, x):
        """Returns ``{blob_name: tensor}`` with every blob at its FINAL value."""
        blobs = {"data": x}
        for spec in self.specs:
            key = spec.name.replace("/", "__")
            if spec.kind == "Convolution":
                blobs[spec.top[0]] = self.convs[key](blobs[spec.bottom[0]])
            else:                                   # ReLU, always in place in this model
                blobs[spec.top[0]] = torch.relu(blobs[spec.bottom[0]])
        return blobs


def load_caffemodel(model: CaffeMirror, net_param) -> int:
    """Copy weights from a parsed NetParameter into the mirror. Returns layers loaded."""
    loaded = 0
    by_name = {layer.name: layer for layer in net_param.layer}
    for key, conv in model.convs.items():
        name = key.replace("__", "/")
        layer = by_name.get(name)
        if layer is None or not layer.blobs:
            raise ValueError(f"{name}: no weights in the caffemodel")
        weight = np.array(layer.blobs[0].data, np.float32).reshape(
            list(layer.blobs[0].shape.dim))
        if tuple(weight.shape) != tuple(conv.weight.shape):
            raise ValueError(f"{name}: caffemodel {weight.shape} vs mirror "
                             f"{tuple(conv.weight.shape)}")
        conv.weight.data = torch.from_numpy(weight)
        conv.bias.data = torch.from_numpy(np.array(layer.blobs[1].data, np.float32))
        loaded += 1
    return loaded


def verify_against_cv2(model: CaffeMirror, proto: str, weights: str,
                       tolerance: float = 2e-3) -> dict:
    """Assert the mirror reproduces the DEPLOYED network on real input.

    Checks the six head-source feature maps AND the raw ``*_mbox_conf`` / ``*_mbox_loc``
    outputs, on random input rather than zeros — a zero image passes through a surprising
    number of wrong graphs unchanged.

    Raises on mismatch. A mirror that is quietly wrong trains to something the robot will
    never run, and no shape assertion catches it.
    """
    import cv2
    net = cv2.dnn.readNetFromCaffe(proto, weights)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(image, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN)

    wanted = [f"{s}/relu" for s in HEAD_SOURCES]
    wanted += [f"{s}_mbox_conf" for s in HEAD_SOURCES]
    wanted += [f"{s}_mbox_loc" for s in HEAD_SOURCES]
    net.setInput(blob)
    theirs = dict(zip(wanted, net.forward(wanted), strict=True))

    model.eval()
    with torch.no_grad():
        blobs = model(torch.from_numpy(blob))

    worst = {}
    for name, reference in theirs.items():
        # cv2 addresses the LAYER; the mirror keys by blob. `conv11/relu` the layer writes
        # blob `conv11`, so the mirror's `conv11` is the post-ReLU value cv2 returns.
        key = name[:-5] if name.endswith("/relu") else name
        mine = blobs[key].numpy()
        error = float(np.abs(mine - reference).max())
        worst[name] = error
        if error > tolerance:
            raise ValueError(
                f"MIRROR DOES NOT MATCH THE DEPLOYED NETWORK at {name}: max abs error "
                f"{error:.5f} > {tolerance}. Training this would produce weights for a "
                f"graph the robot does not run.")
    return worst
