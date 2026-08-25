#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune the deployed MobileNet-SSD end to end, backbone unfrozen, and export it back.

``train_new_class.py`` fits a linear probe on frozen features and
``FROZEN-FEATURE-CEILING.md`` records that it does not work: 53% recall at 38% false
positives cross-day, no usable operating point at any threshold. The stated next step was
"unfreeze the backbone: real SSD training — multibox loss, prior matching, hard negative
mining, augmentation — then export back through ``add_class.py``". This is that.

## What is borrowed rather than re-derived, and why it matters

**The priors come from ``cv2``**, out of the deployed network's own ``mbox_priorbox`` blob.
Re-deriving PriorBox from the layer parameters would produce 1917 boxes that are *probably*
the same, and a half-cell offset error there is invisible in the loss and fatal at
inference — the loc head would learn to regress from boxes ``DetectionOutput`` does not use.

**The head layout is checked, not assumed.** ``mbox_loc`` and ``mbox_conf`` are ordinary
convolutions whose channels are ``prior * 4 + coord`` and ``prior * classes + class``, and
Caffe flattens them cell-major after a permute. :func:`verify_head_assembly` asserts that
this module's re-assembly reproduces ``cv2``'s own ``mbox_loc`` and ``mbox_conf`` blobs on
real input, so the claim about the ordering is measured rather than believed. Measured on
the robot's own weights grown to 22 classes: 2.9e-05 on loc, 3.7e-04 on conf.

**The encoding is the prototxt's.** ``detection_output_param`` says ``code_type:
CENTER_SIZE`` and PriorBox carries ``variance: 0.1, 0.1, 0.2, 0.2``, so
:func:`encode_boxes` divides the centre offsets by 0.1 and the log size ratios by 0.2.
Training against any other parameterisation gives a loc head that ``DetectionOutput``
decodes into boxes somewhere else entirely, and the conf head would still look fine.

## Three departures from stock SSD, each forced by this corpus

**Negatives-only frames get a floor of hard negatives.** Stock SSD mines ``3 * num_pos``
negatives per image, so an image with no object contributes NOTHING to the loss. 705 of the
2,048 training frames here are deliberately peer-free corridor, and they are the most
valuable frames in the set — the frozen-head work established that this network's failure
mode is firing on empty corridor, not missing the robot. ``--neg-floor`` mines that many
hardest negatives from an image with no positives.

**The old classes are held in place by distillation, not by construction.** The frozen head
could promise ``person`` was untouched because it never wrote outside one class's channels.
Unfreezing the backbone moves every class at once, and ``person`` is on this robot's stop
path. So a frozen copy of the starting network runs on the same augmented batch and its
logits for classes 0..20 and its loc predictions are an L2 target. This costs one extra
forward pass and is the only thing standing between a fine-tune and a robot that has
stopped seeing people.

**Early layers are frozen by default.** 1,343 labelled frames from one corridor against
5.7M parameters overfits harder unfrozen, not less. ``conv0``..``conv5`` are generic edge
and texture filters that 1,343 near-duplicate frames cannot improve on, and freezing them
also removes the largest activations from the backward pass.

## What this does NOT do

It does not touch the prototxt beyond what ``add_class.py`` already changes, so the exported
model loads in the OpenCV 4.2 ``cv2.dnn`` on the Jetson exactly as the current one does.
It does not evaluate itself: ``eval_detector.py`` scores a written ``.caffemodel`` through
the real ``DetectionOutput``, because a torch-side decode that agreed with itself would
prove nothing about the thing the robot runs.

Usage:

    finetune_ssd.py --proto mnssd22.prototxt --model mnssd22.caffemodel \\
                    --labels peer_go2wheel_20260824.json --images .../peercap \\
                    --negatives-glob '.../peercap/neg_*.jpg' \\
                    --out-dir runs/unfrozen --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ssd_torch import (
    HEAD_SOURCES,
    INPUT_SIZE,
    SSD_MEAN,
    SSD_SCALE,
    CaffeMirror,
    load_caffemodel,
    parse_prototxt,
)

#: IoU at which a prior is "responsible" for a box. SSD's own threshold; the same 0.5 the
#: frozen-head trainer used, so the two are comparable.
MATCH_IOU = 0.5

#: Negatives per positive in hard mining. SSD's ratio, unchanged.
NEGATIVE_RATIO = 3

#: Weight on the localisation term relative to confidence. SSD uses 1.0.
LOC_WEIGHT = 1.0

#: A teacher pseudo-label overlapping the labelled new-class box by more than this is
#: dropped. The shipped network reads this robot as ``dog`` and as ``chair``, and keeping
#: those would train two contradictory labels onto one object.
OVERLAP_LIMIT = 0.3


# --------------------------------------------------------------------------- priors/heads


def read_priors(proto: Path, model: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(priors, variances)`` straight out of the deployed network's ``mbox_priorbox``.

    Returned as ``(N, 4)`` each: priors are normalised ``xmin, ymin, xmax, ymax`` and are
    NOT clipped to the image (the prototxt says ``clip: false``, and row 0 really does start
    at -0.0737). Variances are the per-coordinate divisors ``DetectionOutput`` multiplies
    back in when it decodes, so encoding has to divide by them.
    """
    import cv2

    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    blank = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    net.setInput(cv2.dnn.blobFromImage(blank, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN))
    prior_blob = net.forward(["mbox_priorbox"])[0]
    return (prior_blob[0, 0].reshape(-1, 4).astype(np.float32),
            prior_blob[0, 1].reshape(-1, 4).astype(np.float32))


def assemble_heads(blobs: dict, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(loc, conf)`` as ``(N, priors, 4)`` and ``(N, priors, classes)``.

    Caffe permutes each head to ``(N, H, W, C)`` and flattens, so the prior index runs
    cell-major (row-major over the feature map) then slot, exactly matching the order
    PriorBox emits. :func:`verify_head_assembly` is what makes that a fact.
    """
    locs, confs = [], []
    for source in HEAD_SOURCES:
        loc = blobs[f"{source}_mbox_loc"]
        conf = blobs[f"{source}_mbox_conf"]
        batch = loc.shape[0]
        locs.append(loc.permute(0, 2, 3, 1).reshape(batch, -1, 4))
        confs.append(conf.permute(0, 2, 3, 1).reshape(batch, -1, num_classes))
    return torch.cat(locs, 1), torch.cat(confs, 1)


def verify_head_assembly(model: CaffeMirror, proto: Path, model_path: Path,
                         num_classes: int, tolerance: float = 2e-3) -> dict:
    """Assert :func:`assemble_heads` reproduces ``cv2``'s own ``mbox_loc``/``mbox_conf``.

    The whole loss is written against one claim about channel and cell ordering. If that
    claim is wrong the loss still decreases — it just trains the wrong prior for every box,
    and the failure surfaces as a detector whose boxes are in the wrong place, which is
    indistinguishable from "it did not learn". So the claim is checked against the network
    that will run, on random input, every time training starts.
    """
    import cv2

    net = cv2.dnn.readNetFromCaffe(str(proto), str(model_path))
    image = np.random.default_rng(1).integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(image, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN)
    net.setInput(blob)
    reference_loc, reference_conf = net.forward(["mbox_loc", "mbox_conf"])

    model.eval()
    with torch.no_grad():
        loc, conf = assemble_heads(model(torch.from_numpy(blob)), num_classes)
    worst = {
        "mbox_loc": float(np.abs(loc.numpy().reshape(1, -1) - reference_loc).max()),
        "mbox_conf": float(np.abs(conf.numpy().reshape(1, -1) - reference_conf).max()),
    }
    for name, error in worst.items():
        if error > tolerance:
            raise ValueError(
                f"HEAD ASSEMBLY DISAGREES WITH THE DEPLOYED NETWORK at {name}: {error:.5f} "
                f"> {tolerance}. The loss would train the wrong prior for every box.")
    return worst


# ------------------------------------------------------------------------------ matching


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(len(a), len(b))`` IoU between two sets of ``xmin, ymin, xmax, ymax`` boxes."""
    top_left = torch.max(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.min(a[:, None, 2:], b[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return overlap / (area_a[:, None] + area_b[None, :] - overlap).clamp(min=1e-9)


def encode_boxes(matched: torch.Tensor, priors: torch.Tensor,
                 variances: torch.Tensor) -> torch.Tensor:
    """CENTER_SIZE encoding of ``matched`` against ``priors``, divided by the variances.

    This is the exact inverse of what ``DetectionOutput`` does when it decodes, which is
    the only reason the loc head's output means anything at inference. Getting the variance
    division backwards (or omitting it) trains a head whose predictions are 10x and 5x off
    in the two halves — a mistake that looks like a well-behaved smooth-L1 curve.
    """
    prior_centre = (priors[:, :2] + priors[:, 2:]) / 2
    prior_size = (priors[:, 2:] - priors[:, :2]).clamp(min=1e-9)
    box_centre = (matched[:, :2] + matched[:, 2:]) / 2
    box_size = (matched[:, 2:] - matched[:, :2]).clamp(min=1e-9)
    centre = (box_centre - prior_centre) / prior_size / variances[:, :2]
    size = torch.log(box_size / prior_size) / variances[:, 2:]
    return torch.cat([centre, size], 1)


def match_priors(truths: torch.Tensor, priors: torch.Tensor,
                 threshold: float = MATCH_IOU) -> tuple[torch.Tensor, torch.Tensor]:
    """``(box index per prior, matched mask)`` for one image.

    Two rules, both SSD's. Every prior whose IoU with a box clears ``threshold`` is matched
    to its best box; and every box additionally FORCES its single best prior, whatever the
    overlap. The forced match is not decoration: a box smaller or oddly-shaped enough to
    clear 0.5 against nothing at all would otherwise contribute no positive and no loss,
    and 39% of this corpus's boxes are truncated at a frame edge.
    """
    if len(truths) == 0:
        empty = torch.zeros(len(priors), dtype=torch.long, device=priors.device)
        return empty, torch.zeros(len(priors), dtype=torch.bool, device=priors.device)
    overlaps = box_iou(truths, priors)
    best_truth_overlap, best_truth_index = overlaps.max(0)
    _, best_prior_index = overlaps.max(1)
    best_truth_overlap.index_fill_(0, best_prior_index, 2.0)
    for truth in range(len(truths)):
        best_truth_index[best_prior_index[truth]] = truth
    return best_truth_index, best_truth_overlap >= threshold


class MultiBoxLoss(nn.Module):
    """SSD's loss: smooth-L1 on matched loc, softmax CE on conf, hard negatives at 3:1.

    ``neg_floor`` is the one departure. Stock SSD takes ``3 * num_pos`` negatives per image,
    which is zero for an image containing no object — and a third of this corpus is exactly
    that, on purpose. Those frames carry the corridor's confusing corners, and the measured
    failure of the previous approach was firing on them.
    """

    def __init__(self, priors: torch.Tensor, variances: torch.Tensor, num_classes: int,
                 neg_floor: int = 32) -> None:
        super().__init__()
        self.register_buffer("priors", priors)
        self.register_buffer("variances", variances)
        self.num_classes = num_classes
        self.neg_floor = neg_floor

    def forward(self, loc: torch.Tensor, conf: torch.Tensor,
                targets: list) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch, num_priors = conf.shape[:2]
        device = conf.device
        loc_target = torch.zeros(batch, num_priors, 4, device=device)
        conf_target = torch.zeros(batch, num_priors, dtype=torch.long, device=device)
        positive = torch.zeros(batch, num_priors, dtype=torch.bool, device=device)
        for index, (boxes, labels) in enumerate(targets):
            boxes = boxes.to(device)
            index_per_prior, matched = match_priors(boxes, self.priors)
            if len(boxes):
                loc_target[index] = encode_boxes(boxes[index_per_prior], self.priors,
                                                 self.variances)
                conf_target[index] = labels.to(device)[index_per_prior] * matched
            positive[index] = matched

        num_positive = int(positive.sum())
        loc_loss = functional.smooth_l1_loss(loc[positive], loc_target[positive],
                                             reduction="sum") if num_positive else loc.sum() * 0

        # Hard negative mining, per image, on the loss the CURRENT model assigns. Mining
        # once up front against the starting model was measured to be worse than not mining
        # at all -- the head stops resembling its seed on the first step, so the mined set
        # is chosen for a model that no longer exists.
        flat = conf.reshape(-1, self.num_classes)
        all_loss = functional.cross_entropy(flat, conf_target.reshape(-1), reduction="none")
        all_loss = all_loss.reshape(batch, num_priors)
        negative_loss = all_loss.masked_fill(positive, -1.0)
        order = negative_loss.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        wanted = (positive.sum(1) * NEGATIVE_RATIO).clamp(min=self.neg_floor)
        wanted = torch.min(wanted, torch.full_like(wanted, num_priors) - positive.sum(1))
        negative = rank < wanted[:, None]

        selected = positive | negative
        conf_loss = all_loss[selected].sum()
        normaliser = max(num_positive, 1)
        return conf_loss / normaliser, LOC_WEIGHT * loc_loss / normaliser, num_positive


# -------------------------------------------------------------------------- augmentation


@dataclass
class AugmentConfig:
    """Augmentation strengths. Defaults are SSD's, minus the channel swap.

    No random channel swap: the object is achromatic and the corridor's colour is one of
    the few cues separating a grey robot from a grey floor, so permuting BGR manufactures
    illuminants that cannot occur and throws away a real signal. Everything else is stock.
    """

    brightness: float = 32.0
    contrast: tuple = (0.5, 1.5)
    saturation: tuple = (0.5, 1.5)
    hue: float = 18.0
    expand_max: float = 3.0
    expand_probability: float = 0.5
    flip_probability: float = 0.5
    crop_attempts: int = 50


def photometric(image: np.ndarray, config: AugmentConfig, rng: random.Random) -> np.ndarray:
    """Brightness, contrast, saturation and hue jitter, in SSD's order."""
    import cv2

    out = image.astype(np.float32)
    if rng.random() < 0.5:
        out += rng.uniform(-config.brightness, config.brightness)
    if rng.random() < 0.5:
        out *= rng.uniform(*config.contrast)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(*config.saturation)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-config.hue, config.hue)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def expand(image: np.ndarray, boxes: np.ndarray, config: AugmentConfig,
           rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Zoom out: paste the frame into a larger mean-filled canvas.

    This is the augmentation that matters most here. The corpus has the peer at nine staged
    distances and nothing further away than about 4 m, while the deployment case is a peer
    arriving down a corridor. Expand is the only operator that manufactures a SMALLER
    apparent robot, and small-and-far is where the frozen head failed.
    """
    if rng.random() > config.expand_probability:
        return image, boxes
    height, width = image.shape[:2]
    ratio = rng.uniform(1.0, config.expand_max)
    new_height, new_width = int(height * ratio), int(width * ratio)
    left = rng.randint(0, new_width - width)
    top = rng.randint(0, new_height - height)
    canvas = np.full((new_height, new_width, 3), SSD_MEAN, np.uint8)
    canvas[top:top + height, left:left + width] = image
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width + left) / new_width
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height + top) / new_height
    return canvas, boxes


def random_crop(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                config: AugmentConfig, rng: random.Random) -> tuple:
    """SSD's IoU-constrained sample crop; returns the original if no sample qualifies.

    A box is kept when its CENTRE falls inside the crop, and is then clipped to it — so a
    crop can truncate the robot, which is what half this corpus already looks like and what
    a peer walking past the camera does.
    """
    height, width = image.shape[:2]
    modes = (None, 0.1, 0.3, 0.5, 0.7, 0.9, "any")
    for _ in range(config.crop_attempts):
        mode = rng.choice(modes)
        if mode is None:
            return image, boxes, labels
        scale = rng.uniform(0.3, 1.0)
        aspect = rng.uniform(0.5, 2.0)
        crop_width = int(width * scale * math.sqrt(aspect))
        crop_height = int(height * scale / math.sqrt(aspect))
        if not (0 < crop_width <= width and 0 < crop_height <= height):
            continue
        left = rng.randint(0, width - crop_width)
        top = rng.randint(0, height - crop_height)
        window = np.array([left / width, top / height,
                           (left + crop_width) / width, (top + crop_height) / height], np.float32)
        kept = np.zeros(len(boxes), bool)
        if len(boxes):
            overlap = box_iou(torch.from_numpy(boxes), torch.from_numpy(window[None]))[:, 0]
            if mode != "any" and float(overlap.max()) < mode:
                continue
            centres = (boxes[:, :2] + boxes[:, 2:]) / 2
            kept = ((centres > window[:2]) & (centres < window[2:])).all(1)
            if not kept.any():
                continue
        cropped = image[top:top + crop_height, left:left + crop_width]
        if not len(boxes):
            return cropped, boxes, labels
        out = boxes[kept].copy()
        out[:, [0, 2]] = (np.clip(out[:, [0, 2]], window[0], window[2]) - window[0]) / (
            window[2] - window[0])
        out[:, [1, 3]] = (np.clip(out[:, [1, 3]], window[1], window[3]) - window[1]) / (
            window[3] - window[1])
        return cropped, out, labels[kept]
    return image, boxes, labels


def horizontal_flip(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the frame AND the boxes. Flipping only the image trains against noise."""
    flipped = image[:, ::-1].copy()
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
    return flipped, boxes


# ------------------------------------------------------------------------------- dataset


class PeerFrames(Dataset):
    """Labelled peer frames plus peer-free frames, augmented into 300x300 network input.

    Boxes are carried normalised, so every geometric operator is a coordinate change and
    nothing has to know the source resolution. The frames are 1920x1080 and the network is
    a square 300x300: ``cv2.dnn.blobFromImage`` warps rather than letterboxes, so training
    warps too. Preserving aspect here would train on a geometry the robot never sees.
    """

    def __init__(self, records: list, augment: bool, config: AugmentConfig,
                 new_class: int, seed: int = 0, pseudo: dict | None = None) -> None:
        self.records = records
        self.augment = augment
        self.config = config
        self.new_class = new_class
        self.seed = seed
        self.pseudo = pseudo or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import cv2

        path, box = self.records[index]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        if box is None:
            boxes = np.zeros((0, 4), np.float32)
        else:
            boxes = np.array([[box[0] / width, box[1] / height,
                               box[2] / width, box[3] / height]], np.float32)
        labels = np.full(len(boxes), self.new_class, np.int64)
        extra = self.pseudo.get(str(path)) or []
        if extra:
            boxes = np.concatenate([boxes, np.array(extra, np.float32)[:, :4]])
            labels = np.concatenate([labels, np.array([int(e[4]) for e in extra], np.int64)])

        if self.augment:
            rng = random.Random((self.seed * 1_000_003) ^ (index * 2_654_435_761))
            image = photometric(image, self.config, rng)
            image, boxes = expand(image, boxes, self.config, rng)
            image, boxes, labels = random_crop(image, boxes, labels, self.config, rng)
            if rng.random() < self.config.flip_probability:
                image, boxes = horizontal_flip(image, boxes)

        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        blob = (image.astype(np.float32) - SSD_MEAN) * SSD_SCALE
        # Degenerate boxes are possible after a crop clips a truncated box to a sliver.
        # Dropping them here rather than in the loss keeps the loss free of guards.
        if len(boxes):
            keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
            boxes, labels = boxes[keep], labels[keep]
        return (torch.from_numpy(blob.transpose(2, 0, 1).copy()),
                torch.from_numpy(boxes), torch.from_numpy(labels))


def teacher_labels(proto: Path, weights: Path, paths: list, threshold: float,
                   cache: Path) -> dict:
    """Old-class boxes the STARTING network still finds, keyed by image path.

    ⚠️ WITHOUT THIS, FINE-TUNING TEACHES THE NETWORK THAT PEOPLE ARE BACKGROUND, and the
    measurement is not subtle. This corpus is labelled for one class, so every other object
    in it -- and the human operator is in a large fraction of the frames, by the labeller's
    own account -- carries no box. Hard negative mining then picks precisely the priors the
    model is most confident about, which are the operator's, and trains them to class 0.
    Measured on the first run that did not do this: on the fifteen held-out frames where the
    shipped model detects a person at 0.819 mean, the fine-tuned model scored 0.179 and fell
    below the detection threshold on FOURTEEN of them. The new class was the best it had
    ever been on the same run. A detector that finds a peer robot and stops seeing people is
    strictly worse than no detector, because ``person`` is what the stop path runs on.

    Distillation alone does not fix it. That term pins the old classes' LOGITS; the
    cross-entropy is free to raise class 0 above them, and softmax does the rest.

    Boxes overlapping the labelled new-class box by more than ``OVERLAP_LIMIT`` are dropped:
    the network reads this robot as ``dog`` and as ``chair`` often enough that keeping them
    would put two contradictory labels on the same object.

    Cached, because it is one deterministic pass over 2,048 frames and re-running it per
    experiment is the difference between a sweep and an afternoon.
    """
    import cv2

    if cache.exists():
        return json.loads(cache.read_text())
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    out = {}
    for path, box in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE),
                                           SSD_MEAN))
        rows = net.forward()[0, 0]
        kept = []
        truth = None if box is None else np.array(
            [box[0] / width, box[1] / height, box[2] / width, box[3] / height], np.float32)
        for row in rows:
            label, score = int(row[1]), float(row[2])
            if label == 0 or score < threshold:
                continue
            found = np.clip(row[3:7].astype(np.float32), 0.0, 1.0)
            if found[2] - found[0] < 1e-3 or found[3] - found[1] < 1e-3:
                continue
            if truth is not None and float(box_iou(torch.from_numpy(found[None]),
                                                   torch.from_numpy(truth[None]))) > OVERLAP_LIMIT:
                continue
            kept.append([*found.tolist(), label])
        out[str(path)] = kept
    cache.write_text(json.dumps(out))
    return out


def collate(batch: list) -> tuple:
    images = torch.stack([item[0] for item in batch])
    return images, [(item[1], item[2]) for item in batch]


def load_records(labels_path: Path, images_dir: Path, negatives: list,
                 peer01_keep: int, seed: int) -> tuple[list, dict]:
    """``[(path, box or None)]`` plus a count breakdown.

    ``peer01`` is subsampled because ``labels/LABELLING.md`` measured it to be 640 frames of
    ONE picture -- identical box on every frame, anchor-NCC shift of 0 px over the whole
    segment, 34% of the corpus and about one observation. Left whole it outweighs the eight
    other viewpoints put together. The same 80 the frozen-head run kept, so the two are
    trained on the same 1,343 frames.
    """
    data = json.loads(labels_path.read_text())
    rng = random.Random(seed)
    peer01 = [r for r in data["records"] if r["image"].startswith("peer01")]
    others = [r for r in data["records"] if not r["image"].startswith("peer01")]
    if 0 < peer01_keep < len(peer01):
        peer01 = rng.sample(peer01, peer01_keep)
    records = [(images_dir / r["image"], r["box"]) for r in others + peer01]
    records += [(path, None) for path in negatives]
    return records, {"positive": len(others) + len(peer01), "negative": len(negatives),
                     "peer01_kept": len(peer01)}


# -------------------------------------------------------------------------------- export


def export_caffemodel(model: CaffeMirror, template: Path, out_model: Path,
                      caffe_pb2_dir: Path) -> None:
    """Write the trained convolutions back into a ``.caffemodel``.

    Same protobuf path ``add_class.py`` proved: parse, replace the blob payloads, reserialise.
    A fine-tune that cannot be exported is not a fine-tune, so this is not an afterthought --
    it is checked immediately afterwards by re-running ``verify_against_cv2`` on the file
    that was written, which is the only evidence that the robot would run what was trained.
    """
    sys.path.insert(0, str(caffe_pb2_dir))
    import caffe_pb2

    net = caffe_pb2.NetParameter()
    net.ParseFromString(template.read_bytes())
    by_name = {layer.name: layer for layer in net.layer}
    for key, conv in model.convs.items():
        layer = by_name[key.replace("__", "/")]
        weight = conv.weight.detach().cpu().numpy().astype(np.float32)
        bias = conv.bias.detach().cpu().numpy().astype(np.float32)
        if list(layer.blobs[0].shape.dim) != list(weight.shape):
            raise ValueError(f"{layer.name}: template {list(layer.blobs[0].shape.dim)} vs "
                             f"trained {list(weight.shape)}")
        del layer.blobs[0].data[:], layer.blobs[1].data[:]
        layer.blobs[0].data.extend(weight.ravel().tolist())
        layer.blobs[1].data.extend(bias.ravel().tolist())
    out_model.write_bytes(net.SerializeToString())


# ------------------------------------------------------------------------------ training


def freeze_prefixes(model: CaffeMirror, through: str) -> list:
    """Freeze every convolution up to and including ``through``. Returns the frozen names.

    Layers are frozen by DECLARATION ORDER in the prototxt, not by name matching, because
    the depthwise layers are called ``conv3/dw`` and a name test would either miss them or
    catch ``conv13``. An empty ``through`` freezes nothing.
    """
    frozen = []
    if not through:
        return frozen
    names = [spec.name for spec in model.specs if spec.kind == "Convolution"]
    if through not in names:
        raise ValueError(f"--freeze-through {through!r} is not a convolution in this model")
    for name in names[:names.index(through) + 1]:
        conv = model.convs[name.replace("/", "__")]
        conv.weight.requires_grad_(False)
        conv.bias.requires_grad_(False)
        frozen.append(name)
    return frozen


def build_model(proto: Path, weights: Path, caffe_pb2_dir: Path) -> CaffeMirror:
    sys.path.insert(0, str(caffe_pb2_dir))
    import caffe_pb2

    model = CaffeMirror(parse_prototxt(proto.read_text()))
    parameters = caffe_pb2.NetParameter()
    parameters.ParseFromString(weights.read_bytes())
    load_caffemodel(model, parameters)
    return model


def distillation_loss(student_loc: torch.Tensor, student_conf: torch.Tensor,
                      teacher_loc: torch.Tensor, teacher_conf: torch.Tensor,
                      old_classes: int, distil_background: bool = False) -> torch.Tensor:
    """L2 pull towards the STARTING network's predictions for the twenty existing classes.

    Only the old classes are constrained; the new class has no teacher and must be free to
    move. The loc head is constrained too, and that is the half people forget:
    ``share_location: true`` means one box regression serves every class, so a loc head that
    drifts to fit one new object moves ``person``'s boxes with it while ``person``'s SCORE
    looks untouched.

    ⚠️ BACKGROUND IS EXCLUDED, and the first wave of runs is why. With class 0 in the
    target, distillation pins the background logit at the value a network that had never
    seen this object assigned it -- which is high on every prior, since none of them held an
    object it knew. The new class then has to beat a logit the loss is actively holding up,
    and it does not: at ``--distil 1.0`` with background included, the ``conv5``-frozen run
    reached 15% frame recall at epoch 16 while the same recipe with the trunk frozen (where
    the distillation term was 60x smaller and so barely acted) reached 46%.

    Background does not need a teacher anyway. It is the ONE old class the training data
    supervises directly: every mined hard negative is a background label. Classes 1..20 are
    the ones with no labels in this corpus, and they are exactly what this term is for.
    """
    first = 0 if distil_background else 1
    conf_term = functional.mse_loss(student_conf[..., first:old_classes],
                                    teacher_conf[..., first:old_classes])
    return conf_term + functional.mse_loss(student_loc, teacher_loc)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--proto", type=Path, required=True,
                        help="prototxt ALREADY grown to the new class count by add_class.py")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--negatives-glob", default="",
                        help="glob of TRAINING frames known to hold no instance of the class")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="multiplier on the backbone's learning rate. The heads have a "
                             "new class to fit; the backbone has 1,343 frames of one "
                             "corridor and every reason to memorise it")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--freeze-through", default="conv5",
                        help="freeze convolutions up to and including this one; '' for none")
    parser.add_argument("--distil", type=float, default=1.0,
                        help="weight on the old-class distillation term. 0 disables it and "
                             "hands the robot a detector whose person class has moved")
    parser.add_argument("--pseudo-labels", type=float, default=0.0,
                        help="score above which the STARTING network's own detections are "
                             "carried as extra ground truth. 0 disables it, and the "
                             "measured cost of disabling it is the person class")
    parser.add_argument("--distil-background", action="store_true",
                        help="include class 0 in the distillation target. Measured to "
                             "suppress the new class; see distillation_loss")
    parser.add_argument("--neg-floor", type=int, default=32,
                        help="hard negatives mined from a frame containing no object")
    parser.add_argument("--peer01-keep", type=int, default=80,
                        help="frames kept from the peer01 segment; see load_records")
    parser.add_argument("--num-classes", type=int, default=22)
    parser.add_argument("--old-classes", type=int, default=21)
    parser.add_argument("--expand-max", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--caffe-pb2", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.proto, args.model, args.caffe_pb2)
    checks = verify_head_assembly(model, args.proto, args.model, args.num_classes)
    print(f"head assembly matches cv2: {checks}")

    teacher = build_model(args.proto, args.model, args.caffe_pb2).to(args.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    priors_np, variances_np = read_priors(args.proto, args.model)
    print(f"{len(priors_np)} priors from the deployed network, variances "
          f"{variances_np[0].tolist()}")
    criterion = MultiBoxLoss(torch.from_numpy(priors_np), torch.from_numpy(variances_np),
                             args.num_classes, args.neg_floor).to(args.device)

    negatives = [Path(p) for p in sorted(glob(args.negatives_glob))]
    records, counts = load_records(args.labels, args.images, negatives,
                                   args.peer01_keep, args.seed)
    print(f"{counts['positive']} labelled + {counts['negative']} peer-free frames "
          f"({counts['peer01_kept']} kept from peer01)")

    pseudo = None
    if args.pseudo_labels > 0:
        pseudo = teacher_labels(args.proto, args.model, records, args.pseudo_labels,
                                args.out_dir / "pseudo_labels.json")
        total = sum(len(v) for v in pseudo.values())
        print(f"{total} teacher boxes over {sum(1 for v in pseudo.values() if v)} frames "
              f"carried as old-class ground truth at >= {args.pseudo_labels}")

    config = AugmentConfig(expand_max=args.expand_max)
    loader = DataLoader(PeerFrames(records, True, config, args.num_classes - 1, args.seed,
                                   pseudo),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)

    model = model.to(args.device)
    frozen = freeze_prefixes(model, args.freeze_through)
    print(f"frozen: {len(frozen)} convolutions up to {args.freeze_through or 'none'}")

    head_names = {f"{s}_mbox_{k}".replace("/", "__") for s in HEAD_SOURCES
                  for k in ("loc", "conf")}
    heads, backbone = [], []
    for name, conv in model.convs.items():
        if not conv.weight.requires_grad:
            continue
        (heads if name in head_names else backbone).extend([conv.weight, conv.bias])
    optimiser = torch.optim.SGD(
        [{"params": heads, "lr": args.learning_rate},
         {"params": backbone, "lr": args.learning_rate * args.backbone_lr_scale}],
        momentum=args.momentum, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)

    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(4)
        for images, targets in loader:
            images = images.to(args.device, non_blocking=True)
            loc, conf = assemble_heads(model(images), args.num_classes)
            conf_loss, loc_loss, matched = criterion(loc, conf, targets)
            loss = conf_loss + loc_loss
            if args.distil > 0:
                with torch.no_grad():
                    teacher_loc, teacher_conf = assemble_heads(teacher(images), args.num_classes)
                distil = distillation_loss(loc, conf, teacher_loc, teacher_conf,
                                           args.old_classes, args.distil_background)
                loss = loss + args.distil * distil
            else:
                distil = torch.zeros((), device=args.device)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
            optimiser.step()
            totals += [conf_loss.item(), loc_loss.item(), distil.item(), matched]
        schedule.step()
        batches = max(len(loader), 1)
        line = (f"epoch {epoch + 1:3d}/{args.epochs}  conf {totals[0] / batches:.4f}  "
                f"loc {totals[1] / batches:.4f}  distil {totals[2] / batches:.5f}  "
                f"matched/batch {totals[3] / batches:.1f}")
        print(line, flush=True)
        history.append({"epoch": epoch + 1, "conf": totals[0] / batches,
                        "loc": totals[1] / batches, "distil": totals[2] / batches,
                        "matched": totals[3] / batches})

        out_model = args.out_dir / f"epoch{epoch + 1:03d}.caffemodel"
        export_caffemodel(model, args.model, out_model, args.caffe_pb2)

    (args.out_dir / "history.json").write_text(json.dumps(
        {"history": history, "counts": counts, "args": {k: str(v) for k, v in
                                                        vars(args).items()}}, indent=2))
    print(f"\nwrote {args.epochs} checkpoints to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
