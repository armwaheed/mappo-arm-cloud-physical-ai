#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fit the new classes' score heads on FROZEN MobileNet-SSD features.

This is deliberately not "retrain SSD". Three properties of the shipped network collapse
the job to something small, convex, and checkable:

  * **``share_location: true``.** The box regression is shared across classes — one set of
    localisation predictions per prior, not one per class. A new class therefore needs NO
    localisation changes whatsoever, and the boxes it inherits are the ones the network
    already regresses well.
  * **The backbone is frozen.** Only the six ``*_mbox_conf`` convolutions gain channels, so
    only those channels are trained. Everything upstream is a fixed feature extractor,
    which is the right bias/variance trade for a few hundred examples of one rigid object.
  * **A ``1x1`` convolution is a linear map.** For prior slot ``p`` and class ``c`` the
    logit is ``W[p*C + c] . f + b[p*C + c]`` for the channel vector ``f`` at each cell. With
    every other class's parameters fixed, softmax cross-entropy in the free parameters is
    **convex**, so plain gradient descent gets the optimum and there is no schedule to tune.

## Why this runs through cv2.dnn rather than a PyTorch mirror

The features come out of `cv2.dnn` by asking the production network for its intermediate
blobs, and the priors come out of the same forward pass. Nothing here re-derives a prior
box or re-implements a layer, so nothing here can disagree with the network that will
actually run on the robot — which a hand-written mirror silently would, and would do it in
a way no test catches until a real frame goes through it.

It also means the only dependencies are `cv2` and `numpy`, the same pair the robot already
has.

## The gate scores HELD-OUT negatives, and that distinction is the whole point

``--negatives`` are trained on; ``--gate-negatives`` are not. Measured on this stack: a
head fitted against one session's peer-free frames scored **0 of 705** on that same
session and **16 of 80** on another day's footage of the same corridor. Same model, same
corridor, 0% versus 20%. A gate whose negatives share an hour, a light level and a camera
pose with its training set is measuring whether the fit memorised them.

## What it does NOT do

The existing 20 classes are never touched. That is the point — ``person`` stays on the stop
path and must not move — but it also means this cannot fix a class the network is already
bad at, and cannot help the new class by taking probability from a confusable one. If
``lite3`` turns out to be systematically lost to ``dog``, that needs a real fine-tune with
the backbone unfrozen, and a much larger set.

Usage:

    train_new_class.py --dataset lite3_ds --model MobileNetSSD_deploy.caffemodel \\
                       --proto MobileNetSSD_deploy.prototxt --class-name lite3 \\
                       --out-model mnssd22.caffemodel --out-proto mnssd22.prototxt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from add_class import VOC_CLASSES, grow_weights, rewrite_prototxt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot-stack" / "unitree"
                       / "go2" / "visual_nav"))
import inference_profile

#: ``(feature layer, conf-head prefix, priors per cell)`` for each of the six sources.
#:
#: ⚠️ THE FEATURE LAYER IS THE ReLU, NOT THE CONVOLUTION, and the difference is not
#: cosmetic. In Caffe, ``conv11/bn``, ``conv11/scale`` and ``conv11/relu`` all write IN
#: PLACE to the blob named ``conv11``, so ``conv11_mbox_conf``'s ``bottom: "conv11"`` is
#: the post-ReLU value. But ``cv2.dnn.forward("conv11")`` addresses the LAYER named
#: ``conv11`` — the Convolution — and hands back its output from before the batch norm.
#:
#: Asking for the wrong one returns a correctly-shaped tensor of entirely the wrong
#: distribution: measured on a real frame, ``conv11`` has mean 0.818 and min -13.46, while
#: ``conv11/relu`` has mean 0.140 and min 0.000, as a ReLU output must. A head fitted on
#: the first scores 1.000 on every frame in existence, which is exactly what happened, and
#: no shape check anywhere catches it. :meth:`FrozenFeatures._self_check` does.
CONF_SOURCES = (("conv11/relu", "conv11", 3),
                ("conv13/relu", "conv13", 6),
                ("conv14_2/relu", "conv14_2", 6),
                ("conv15_2/relu", "conv15_2", 6),
                ("conv16_2/relu", "conv16_2", 6),
                ("conv17_2/relu", "conv17_2", 6))

#: THE TRAINING SQUARE, from ``inference_profile.MOBILENET_SSD_TRAINED``, so that it is a
#: named role rather than an anonymous 300 -- the anonymous 300 in five files is what issue
#: #129 is about.
#:
#: 300 is correct HERE: these weights were fitted at 300 and a head trained on features from
#: a different square would be fitted to a distribution the backbone never produced.
#:
#: ⚠️ This comment used to say the square "must match what the robot runs". There is no such
#: thing: the robot runs 300 under run-smoke/berth/chair and 224 under
#: ``deploy/run-peer-supervised.sh``. So a fine-tune from this trainer matches three
#: launchers and is train/serve-skewed against the fourth -- and the measurement is stark:
#: fine-tuned checkpoints emit no box at all at 224 (best score 0.000) while the same
#: weights fire at 0.55-0.66 at 300. Anything trained here is unusable by
#: ``run-peer-supervised.sh`` SPECIFICALLY, which is narrower and more actionable than
#: "unusable by the robot".
INPUT_SIZE = inference_profile.MOBILENET_SSD_TRAINED.input_size

#: Preprocessing baked into the published weights, from the same object.
SSD_SCALE = inference_profile.MOBILENET_SSD_TRAINED.scale
SSD_MEAN = inference_profile.MOBILENET_SSD_TRAINED.mean

#: A prior counts as a positive example when it overlaps the box by at least this. 0.5 is
#: SSD's own matching threshold; using anything else here would train the head against a
#: different notion of "this prior is responsible" than the one the boxes were fitted to.
MATCH_IOU = 0.5

#: Negatives per positive. SSD hard-mines at 3:1 and the ratio matters more than it looks:
#: 1917 priors against a handful of matches is a 300:1 imbalance, and a head trained on all
#: of it learns to say "background" and score 99.7%.
#:
#: ⚠️ RAISED 3 -> 24 after the first end-to-end run. At 3:1, sampled at random from the same
#: image, the negatives were too few and too easy: the fitted head scored 1.000 on 60 of 60
#: real peer-free frames and dragged `person` from 0.968 to nothing. The negatives a
#: detector needs are the ones that ALMOST look right, not a random draw.
NEGATIVE_RATIO = 24

#: Fraction of negatives taken by HARD MINING — the background priors the head currently
#: scores highest — rather than at random. This is what SSD itself does, and it is the
#: difference between a head that has seen the corridor's confusing corners and one that
#: has seen its empty floor.
HARD_NEGATIVE_FRACTION = 0.75

#: L2 penalty, expressed as the weight norm to aim for rather than as a raw lambda.
#: The existing twenty classes were trained with decay and land at |W| mean ~0.465; the
#: first unregularised fit reached 0.953 and saturated every logit. A new class should look
#: like its peers, so the target IS the peers' norm, measured from the model at run time
#: rather than hard-coded.
WEIGHT_NORM_TOLERANCE = 1.5

#: Logit margin the fit demands either side of the rivals' best score.
#:
#: Fitting to ``margin > 0`` — merely BEATING the existing classes — is not enough, and the
#: arithmetic says so before any training does. Inference evaluates all 1917 priors on
#: every frame, so a per-prior false-positive rate of 1e-2 is 19 false boxes per frame and
#: 1e-3 is still two. A clean frame 98% of the time needs 1e-5, which a decision boundary
#: sitting exactly on the rivals cannot deliver: half the borderline priors fall the wrong
#: side of it by construction.
#:
#: A margin of 4 nats puts a negative's softmax share around e^-4 = 1.8% even when it is
#: the runner-up, which is comfortably under any usable detection threshold.
#:
#: ASYMMETRIC, because the two errors are not. A negative has to lose decisively — there
#: are ~1900 of them per frame and any one firing costs a false box. A positive only has to
#: WIN; demanding it win by four nats as well drove recall to zero across most slots while
#: buying nothing, since a positive that beats the rivals at all is already detected.
NEGATIVE_MARGIN = 4.0
POSITIVE_MARGIN = 0.5


def _iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """IoU of every row of ``boxes`` against one ``box``, all ``(xmin,ymin,xmax,ymax)``."""
    x0 = np.maximum(boxes[:, 0], box[0])
    y0 = np.maximum(boxes[:, 1], box[1])
    x1 = np.minimum(boxes[:, 2], box[2])
    y1 = np.minimum(boxes[:, 3], box[3])
    overlap = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            + (box[2] - box[0]) * (box[3] - box[1]) - overlap)
    return np.where(area > 0, overlap / np.maximum(area, 1e-12), 0.0)


class FrozenFeatures:
    """The production network, used as a fixed feature extractor.

    Holds one ``cv2.dnn`` net and asks it for the six confidence-head inputs plus the
    priors. The priors depend only on the input size, so they are read once and checked
    against the feature-map geometry — a mismatch there means the prototxt and this
    module's :data:`CONF_SOURCES` have drifted apart, which is worth a hard failure rather
    than a quietly misaligned label.
    """

    def __init__(self, proto: Path, model: Path) -> None:
        self._net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        blank = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        maps, priors = self._forward(blank)
        self.priors = priors
        self.shapes = [m.shape for m in maps]
        expected = sum(m.shape[2] * m.shape[3] * n
                       for m, (_, _, n) in zip(maps, CONF_SOURCES, strict=True))
        if expected != len(priors):
            raise ValueError(f"{expected} priors implied by the feature maps but the "
                             f"network produced {len(priors)} — CONF_SOURCES is stale")
        self._self_check(model)

    def _self_check(self, model: Path) -> None:
        """Refuse to run unless the extracted features REPRODUCE the network's own logits.

        The whole method rests on one identity: the confidence head is a ``1x1``
        convolution, so ``features @ W.T + b`` must equal the blob the network computes.
        If the wrong blob is extracted, that identity breaks while every shape still lines
        up, the fit trains happily on the wrong distribution, and the failure only appears
        as a detector that fires on everything.

        Checking it costs one forward pass per run.
        """
        import caffe_pb2
        parameters = caffe_pb2.NetParameter()
        parameters.ParseFromString(Path(model).read_bytes())
        layers = {layer.name: layer for layer in parameters.layer}
        noise = np.random.default_rng(0).integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3),
                                                  dtype=np.uint8)
        for feature_layer, prefix, _ in CONF_SOURCES:
            self._net.setInput(cv2.dnn.blobFromImage(
                noise, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN))
            feats, conf = self._net.forward([feature_layer, f"{prefix}_mbox_conf"])
            flat = feats[0].reshape(feats.shape[1], -1).T
            blob = layers[f"{prefix}_mbox_conf"].blobs
            outputs = blob[0].shape.dim[0]
            weights = np.array(blob[0].data, np.float32).reshape(outputs, -1)
            biases = np.array(blob[1].data, np.float32)
            theirs = conf[0].reshape(outputs, -1).T
            error = float(np.abs(flat @ weights.T + biases - theirs).max())
            if error > 1e-2:
                raise ValueError(
                    f"{feature_layer} does not reproduce {prefix}_mbox_conf "
                    f"(max abs error {error:.3f}). The head is a 1x1 convolution, so this "
                    f"must hold exactly — the usual cause is naming the Convolution "
                    f"rather than its in-place ReLU. See CONF_SOURCES.")

    def _forward(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        self._net.setInput(cv2.dnn.blobFromImage(
            image, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN))
        outputs = self._net.forward([name for name, _, _ in CONF_SOURCES]
                                    + ["mbox_priorbox"])
        return list(outputs[:-1]), outputs[-1][0, 0].reshape(-1, 4)

    def features(self, image: np.ndarray) -> list[np.ndarray]:
        """Per source, ``(cells, channels)`` — the vector each prior slot sees."""
        maps, _ = self._forward(image)
        # (1, C, H, W) -> (H*W, C). Row-major over cells, matching the priors' own order.
        return [m[0].reshape(m.shape[1], -1).T.copy() for m in maps]


def slot_priors(priors: np.ndarray, shapes: list) -> list:
    """Split the flat prior list into ``[source][slot] -> prior rows for every cell``.

    The concatenation order is source-major, then cell (row-major), then slot — the same
    order ``mbox_conf`` uses, which is what makes a slot's weights shared across cells.
    """
    out, offset = [], 0
    for shape, (_, _, slots) in zip(shapes, CONF_SOURCES, strict=True):
        cells = shape[2] * shape[3]
        block = priors[offset:offset + cells * slots].reshape(cells, slots, 4)
        out.append([block[:, s, :] for s in range(slots)])
        offset += cells * slots
    return out


def gather(dataset: Path, negatives: Path | None, frozen: FrozenFeatures,
           rng: np.random.Generator) -> list:
    """``[source] -> (features, [labels per slot])`` over the dataset plus pure negatives.

    ⚠️ EVERY PRIOR IS A NEGATIVE. An earlier version sampled 24 hard-mined negatives per
    slot per frame, and it is worth recording why that failed, because the per-slot metrics
    looked excellent while the model was unusable: conv11 slot 0 has 361 cells, so 24
    samples is under 7% of it, the fit separated exactly those and left the other 94%
    unconstrained, and at inference 744 of 1917 priors came back as the new class at
    softmax 1.000. Reported false-positive rate 0.000; actual frames fired on, 159 of 159.

    Mining made it worse rather than better. The hard negatives were chosen by the SEED
    class's score — the priors that most look like a dog — but the head stops being
    dog-like on the first gradient step, so the set was mined for a model that no longer
    exists. SSD re-mines every iteration; a single up-front mining pass is not the same
    thing and is not worth the bias it introduces.

    Features are stored ONCE PER SOURCE rather than once per slot. The slots of a source
    share a feature map — they differ only in which prior box each cell carries — so the
    per-slot copy was duplicating 3-6x for nothing, which is what made using every prior
    look unaffordable in the first place.

    ``rng`` is retained for reproducibility of any future subsampling and to keep the
    signature stable; nothing in this path draws from it now.
    """
    del rng
    records = json.loads((dataset / "annotations.json").read_text())["records"]
    per_slot = slot_priors(frozen.priors, frozen.shapes)
    features = [[] for _ in CONF_SOURCES]
    labels = [[[] for _ in slots] for slots in per_slot]

    def absorb(image: np.ndarray, box: np.ndarray | None) -> None:
        for source, feats in enumerate(frozen.features(image)):
            features[source].append(feats)
            for slot, prior_boxes in enumerate(per_slot[source]):
                if box is None:
                    labels[source][slot].append(np.zeros(len(prior_boxes), np.float32))
                    continue
                overlap = _iou(prior_boxes, box)
                # Priors between the two thresholds are AMBIGUOUS — they overlap enough
                # that calling them background teaches the head to reject the object's own
                # edges. Marked -1 and dropped at fit time, which is what SSD does.
                mark = np.where(overlap >= MATCH_IOU, 1.0,
                                np.where(overlap >= 0.3, -1.0, 0.0)).astype(np.float32)
                labels[source][slot].append(mark)

    for record in records:
        image = cv2.imread(str(dataset / record["image"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        x0, y0, x1, y1 = record["box"]
        absorb(image, np.array([x0 / width, y0 / height,
                                x1 / width, y1 / height], np.float32))

    if negatives is not None:
        for path in sorted(negatives.iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image = cv2.imread(str(path))
                if image is not None:
                    absorb(image, None)

    return [(np.concatenate(f), [np.concatenate(y) for y in slots])
            for f, slots in zip(features, labels, strict=True)]


def fit_logit(features: np.ndarray, labels: np.ndarray, rival: np.ndarray,
              steps: int, learning_rate: float, l2: float
              ) -> tuple[np.ndarray, float, dict]:
    """Fit one class's ``(weights, bias)`` against the frozen classes' best rival logit.

    Softmax cross-entropy over all classes, with every other class fixed, reduces to a
    logistic problem against the rivals' best logit — convex in these parameters, so
    gradient descent finds the optimum.

    ``rival`` comes from the UNCHANGED weights, which is what keeps this honest: the head
    is fitted to beat the scores the network genuinely produces, not a stand-in.

    ⚠️ THE L2 TERM IS NOT OPTIONAL. Without it the first end-to-end fit drove ``|W|`` to
    0.953 against the existing classes' 0.465, saturated every logit, scored 1.000 on 60
    of 60 peer-free frames and suppressed ``person`` through the softmax. The features are
    raw post-ReLU MobileNet activations — mean 0.79, max 19 over 512 dimensions — so an
    unpenalised linear fit on a small set separates it by growing without bound.

    Returns the diagnostics too. A fit that reports nothing is a fit whose divergence is
    discovered by a robot.
    """
    weights = np.zeros(features.shape[1], np.float32)
    bias = np.float32(-4.0)          # start well below the rivals: assume nothing is new
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    # Balance the two classes without letting a handful of positives dominate: cap the
    # up-weighting, because an unbounded ratio is the other half of what made the first
    # fit chase its positives at any cost.
    ratio = min(negatives / max(positives, 1.0), 20.0)
    sample_weight = np.where(labels > 0, ratio, 1.0).astype(np.float32)

    # Push the decision boundary AWAY from the rivals in both directions: a positive must
    # clear them by `margin`, a negative must fall short by it. Without this the boundary
    # sits on the rivals and half the borderline priors land on the wrong side.
    offset = np.where(labels > 0, POSITIVE_MARGIN, -NEGATIVE_MARGIN).astype(np.float32)
    history = []
    for step in range(steps):
        z = np.clip(features @ weights + bias - rival - offset, -60, 60)
        probability = 1.0 / (1.0 + np.exp(-z))
        error = (probability - labels) * sample_weight
        gradient = (features.T @ error) / len(labels) + l2 * weights
        weights -= learning_rate * gradient
        bias -= learning_rate * error.mean()
        if step % max(steps // 8, 1) == 0:
            history.append(float(np.mean(sample_weight * np.logaddexp(
                0.0, -z * np.where(labels > 0, 1.0, -1.0)))))
    # Report against the REAL decision the detector makes — logit versus rivals, no
    # margin — because that is what the frame-level gate will measure.
    achieved = features @ weights + bias - rival
    predicted = achieved > 0
    truth = labels > 0
    return weights, float(bias), {
        "loss": history,
        "recall": float((predicted & truth).sum() / max(truth.sum(), 1)),
        "false_positive_rate": float((predicted & ~truth).sum() / max((~truth).sum(), 1)),
        "weight_norm": float(np.abs(weights).mean()),
    }


def _conf_head(net, name: str, classes: int) -> tuple:
    """``(weights, biases)`` of one conf layer, shaped ``(slots, classes, in_ch)``."""
    layer = {layer.name: layer for layer in net.layer}[f"{name}_mbox_conf"]
    slots = layer.blobs[0].shape.dim[0] // classes
    in_channels = layer.blobs[0].shape.dim[1]
    return (np.array(layer.blobs[0].data, np.float32).reshape(slots, classes, in_channels),
            np.array(layer.blobs[1].data, np.float32).reshape(slots, classes), layer)


def verify(proto: Path, model: Path, negatives: Path, new_class: int,
           threshold: float) -> dict:
    """Run the candidate over frames known to contain nothing new.

    THE GATE. The first end-to-end fit scored 1.000 on 60 of 60 peer-free frames and took
    ``person`` from 0.968 to nothing, and nothing in the pipeline objected — the loss went
    down, the model was written, and only a separate evaluation caught it. A trainer that
    can emit a model which suppresses the one safety-critical class is not finished.
    """
    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    fired = frames = 0
    person_scores = []
    for path in sorted(negatives.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        frames += 1
        net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE,
                                           (INPUT_SIZE, INPUT_SIZE), SSD_MEAN))
        rows = net.forward()[0, 0]
        if any(int(r[1]) == new_class and r[2] >= threshold for r in rows):
            fired += 1
        person = [r[2] for r in rows if int(r[1]) == VOC_CLASSES.index("person")]
        if person:
            person_scores.append(max(person))
    return {"frames": frames, "false_positive_frames": fired,
            "false_positive_rate": fired / max(frames, 1),
            "person_detections": len(person_scores),
            "person_mean": float(np.mean(person_scores)) if person_scores else 0.0}


def train_source(net, name: str, slots: int, slot_data: list, old_classes: int,
                 settings: dict) -> tuple[int, list]:
    """Fit every prior slot of one confidence head. Returns ``(trained, warnings)``.

    Split out of ``main`` because the loop carries the whole method — the rival logits, the
    per-slot fit, the norm check and the write-back — and reading it inside argument
    parsing obscured which of those the gate later depends on.
    """
    weights, biases, layer = _conf_head(net, name, old_classes + 1)
    all_features, per_slot_labels = slot_data
    trained, warnings, matched = 0, [], 0
    for slot in range(slots):
        marks = per_slot_labels[slot]
        if (marks > 0).sum() == 0:
            continue              # nothing ever matched this slot; keep the seed
        matched += 1
        # Drop the ambiguous band: those priors overlap the object enough that calling
        # them background would teach the head to reject its own edges.
        keep = marks >= 0
        features, labels = all_features[keep], marks[keep]
        rival = np.max(features @ weights[slot, :old_classes].T
                       + biases[slot, :old_classes], axis=1)
        new_w, new_b, report = fit_logit(features, labels, rival, settings["steps"],
                                         settings["learning_rate"], settings["l2"])
        weights[slot, old_classes] = new_w
        biases[slot, old_classes] = new_b
        trained += 1
        if report["weight_norm"] > settings["target_norm"] * WEIGHT_NORM_TOLERANCE:
            warnings.append(f"{name}[{slot}] |W|={report['weight_norm']:.3f} vs peers "
                            f"{settings['target_norm']:.3f} — raise --l2")
        print(f"  {name:<10} slot {slot}: recall {report['recall']:.2f} "
              f"fp {report['false_positive_rate']:.3f} "
              f"|W| {report['weight_norm']:.3f} "
              f"loss {report['loss'][0]:.3f}->{report['loss'][-1]:.3f}")
    del layer.blobs[0].data[:], layer.blobs[1].data[:]
    layer.blobs[0].data.extend(weights.ravel().tolist())
    layer.blobs[1].data.extend(biases.ravel().tolist())
    if matched < slots:
        # A slot with no training example is still LIVE: it inherits the seed class's
        # weights, so it scores exactly like a dog and can fire wherever a dog would. The
        # first corrected run left conv16_2 and conv17_2 untrained and they contributed
        # false positives on their own. Push their bias below the seed's so they track its
        # dynamics — never a flat floor, which is its own failure — but can never win.
        for slot in range(slots):
            if per_slot_labels[slot] is not None and (per_slot_labels[slot] > 0).sum():
                continue
            biases[slot, old_classes] -= NEGATIVE_MARGIN
        del layer.blobs[1].data[:]
        layer.blobs[1].data.extend(biases.ravel().tolist())
        print(f"  {name:<10} {slots - matched}/{slots} slots had no example — held "
              f"{NEGATIVE_MARGIN} nats below the seed so they cannot fire")
    return trained, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="directory from render_lite3.py (annotations.json + images/)")
    parser.add_argument("--negatives", type=Path, required=True,
                        help="TRAINING frames known to contain no instance of the new "
                             "class. Not optional: without them the head fires on "
                             "ordinary scenes")
    parser.add_argument("--gate-negatives", type=Path, default=None,
                        help="HELD-OUT negatives the gate scores against. Should be a "
                             "different session from --negatives — ideally a different "
                             "day. Defaults to --negatives, which measures memorisation "
                             "rather than generalisation and says so loudly")
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--seed-class", default="dog", choices=VOC_CLASSES)
    parser.add_argument("--out-proto", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.05,
                        help="weight decay. The default lands the new class's weight norm "
                             "near the existing classes'; the gate checks it did")
    parser.add_argument("--max-false-positive-rate", type=float, default=0.05,
                        help="refuse to write a model that fires on more than this "
                             "fraction of the negative frames")
    parser.add_argument("--detect-threshold", type=float, default=0.2)
    parser.add_argument("--caffe-pb2", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.caffe_pb2))
    try:
        import caffe_pb2
    except ImportError:
        parser.error(f"caffe_pb2 not importable from {args.caffe_pb2} — see add_class.py")

    frozen = FrozenFeatures(args.proto, args.model)
    print(f"frozen features: {[tuple(s[1:]) for s in frozen.shapes]}, "
          f"{len(frozen.priors)} priors")

    old_classes = len(VOC_CLASSES)
    seed_index = VOC_CLASSES.index(args.seed_class)
    original = caffe_pb2.NetParameter()
    original.ParseFromString(args.model.read_bytes())
    peer_norm = []
    for _, name, _ in CONF_SOURCES:
        weights, _, _ = _conf_head(original, name, old_classes)
        peer_norm.append(np.abs(weights).mean())
    target_norm = float(np.mean(peer_norm))
    print(f"existing classes' mean |W| = {target_norm:.4f} — the new class should land "
          f"within {WEIGHT_NORM_TOLERANCE}x of it")

    rng = np.random.default_rng(args.seed)
    data = gather(args.dataset, args.negatives, frozen, rng)

    net = caffe_pb2.NetParameter()
    net.ParseFromString(args.model.read_bytes())
    grow_weights(net, old_classes, old_classes + 1, seed_index)

    settings = {"steps": args.steps, "learning_rate": args.learning_rate,
                "l2": args.l2, "target_norm": target_norm}
    trained, warnings = 0, []
    for (_, name, slots), slot_data in zip(CONF_SOURCES, data, strict=True):
        count, issues = train_source(net, name, slots, slot_data, old_classes, settings)
        trained += count
        warnings.extend(issues)

    proto_text = rewrite_prototxt(args.proto.read_text(), old_classes, old_classes + 1)
    args.out_proto.write_text(proto_text)
    candidate = args.out_model.with_suffix(".candidate")
    candidate.write_bytes(net.SerializeToString())

    # THE GATE MUST NOT SCORE THE FRAMES IT TRAINED ON. Measured: a head fitted on one
    # session's negatives scored 0 of 705 against that same session and 16 of 80 — 20% —
    # against another day's footage of the same corridor. The first number is
    # memorisation and it is the one an unsplit gate reports.
    gate_dir = args.gate_negatives or args.negatives
    if args.gate_negatives is None:
        print("WARNING: --gate-negatives not given, so the gate is scoring the frames it "
              "trained on. That measures memorisation. Expect the real rate to be far "
              "worse on another session.", flush=True)
    checked = verify(args.out_proto, candidate, gate_dir, old_classes,
                     args.detect_threshold)
    print(f"gate: fired on {checked['false_positive_frames']}/{checked['frames']} "
          f"negative frames ({100 * checked['false_positive_rate']:.1f}%), "
          f"person seen on {checked['person_detections']} of them "
          f"(mean {checked['person_mean']:.3f})")
    for warning in warnings:
        print(f"  WARNING {warning}")

    if checked["false_positive_rate"] > args.max_false_positive_rate:
        candidate.unlink()
        args.out_proto.unlink()
        print(f"REFUSED to write: "
              f"{100 * checked['false_positive_rate']:.1f}% false positives "
              f"exceeds --max-false-positive-rate "
              f"{100 * args.max_false_positive_rate:.1f}%. A head this loose "
              f"suppresses `person` through the softmax, which is the one class "
              f"that must not move. Raise --l2, add negatives, or lower "
              f"--learning-rate.")
        return 1

    candidate.replace(args.out_model)
    print(f"trained {trained} prior slots for '{args.class_name}' "
          f"(class {old_classes})")
    print(f"{args.out_proto}")
    print(f"{args.out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
