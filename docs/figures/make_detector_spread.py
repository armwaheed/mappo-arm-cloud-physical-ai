#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One robot, four detectors -- drawn from the committed sweep, not retyped.

The figure for the whitepaper's "one robot, four detectors" section. Small multiples,
one panel per launcher configuration, sharing both axes: false-alarm rate across, peer
recall up. Each panel plots the same 800 fine-tuned checkpoints and the same shipped
weights, scored on the same 284 frames. Only the preprocessing differs, and the cloud
moves bodily between panels -- which is the finding.

Why small multiples rather than one scatter coloured by configuration: four categorical
hues do not clear the all-pairs colour-separation floors that a scatter needs, and three
do. One series per axis-role per panel sidesteps that entirely, and the panel title
carries the identity a legend would have had to.

Inputs are the JSON this repository already commits, byte-for-byte as the training host
wrote it -- so this script retypes no number:

    evidence/2026-08-27-one-robot-four-detectors/sweep/candidates.json   800 checkpoints
    evidence/2026-08-27-one-robot-four-detectors/sweep/incumbent.json    shipped weights

Regenerate with:

    python3 docs/figures/make_detector_spread.py

`--check` re-derives the four headline recalls and exits non-zero if any of them has
moved, so a stale figure is a failing command rather than a picture nobody re-ran.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette: dataviz reference slots 1 and 2, the same two this repository's other figures
# use (evidence/2026-08-14-first-policy-driven-walk/make_charts.py). Two series only,
# which is what lets a scatter use them: the first three slots are the all-pairs-safe set.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
S1 = "#2a78d6"   # slot 1 -- the 800 fine-tuned checkpoints
S2 = "#eb6834"   # slot 2 -- the shipped weights the robot actually runs
GRID = "#e6e5e1"

# Panel order: the three configurations a launcher in this repository really uses, then
# the reference one the checkpoint sweep scored at. Left to right is increasing recall,
# which is also increasing false alarms -- the trade the section is about.
PANELS = [
    ("go2-run-smoke", "deployed"),
    ("go2-navigator-default", "deployed"),
    ("go2-peer-supervised", "deployed — what the robot runs"),
    ("mobilenet-ssd-trained", "reference — what the sweep scored"),
]

# Recomputed from the committed JSON on 2026-08-27 and asserted by --check. These are the
# numbers the whitepaper quotes; if the JSON moves, the command fails rather than the page
# quietly going stale.
EXPECTED_RECALL_PCT = {
    "go2-run-smoke": 13.3,
    "go2-navigator-default": 13.3,
    "go2-peer-supervised": 50.0,
    "mobilenet-ssd-trained": 68.3,
}

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent.parent / "evidence" / "2026-08-27-one-robot-four-detectors" / "sweep"


def rates(entry, profile):
    """(false-alarm %, peer-recall %) for one model under one preprocessing profile.

    `whole` is the whole labelled day: 60 peer-present frames and 221 peer-free ones.
    A profile a model was not scored at is skipped rather than plotted at zero, because
    a missing score and a score of zero are different facts.
    """
    got = entry["results"].get(profile)
    if got is None:
        return None
    w = got["whole"]
    if not w["recall_d"] or not w["fp_d"]:
        return None
    return 100.0 * w["fp_n"] / w["fp_d"], 100.0 * w["recall_n"] / w["recall_d"]


def load():
    with (SWEEP / "candidates.json").open() as fh:
        candidates = json.load(fh)
    with (SWEEP / "incumbent.json").open() as fh:
        incumbent = json.load(fh)
    profiles = {p["profile"]: p for p in incumbent["preprocessing"]}
    return candidates, incumbent, profiles


def check(incumbent):
    """Fail loudly if the committed JSON no longer says what the whitepaper says."""
    entry = incumbent["models"][0]
    bad = []
    for profile, expected in EXPECTED_RECALL_PCT.items():
        got = rates(entry, profile)
        if got is None:
            bad.append(f"{profile}: not scored in incumbent.json")
            continue
        if abs(got[1] - expected) > 0.1:
            bad.append(f"{profile}: figure says {expected:.1f}%, JSON says {got[1]:.1f}%")
    for line in bad:
        print(f"  FAIL  {line}")
    if bad:
        return 1
    print(f"  ok  four configurations, recalls unchanged: "
          f"{', '.join(f'{k} {v:.1f}%' for k, v in EXPECTED_RECALL_PCT.items())}")
    return 0


def draw(candidates, incumbent, profiles, out):
    entry = incumbent["models"][0]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.6, 3.3), sharex=True, sharey=True)

    for ax, (profile, role) in zip(axes, PANELS):
        cloud = [rates(m, profile) for m in candidates["models"]]
        cloud = [c for c in cloud if c is not None]
        ax.scatter([c[0] for c in cloud], [c[1] for c in cloud],
                   s=9, c=S1, alpha=0.30, linewidths=0, zorder=2,
                   label="800 fine-tuned checkpoints")

        shipped = rates(entry, profile)
        if shipped is not None:
            ax.scatter([shipped[0]], [shipped[1]], s=90, c=S2,
                       edgecolors=SURFACE, linewidths=2.0, zorder=4,
                       label="shipped weights")
            # Label above the marker when it sits low in the panel, below when it does
            # not, so the two lines never fall off the bottom axis.
            offset = (9, 6) if shipped[1] < 30 else (9, -24)
            ax.annotate(f"{shipped[1]:.0f}% recall\n{shipped[0]:.0f}% false alarms",
                        xy=shipped, xytext=offset, textcoords="offset points",
                        fontsize=7.5, color=INK2, zorder=5)

        cfg = profiles[profile]
        classes = len(cfg.get("classes") or [])
        ax.set_title(f"{profile}\n{role}\n"
                     f"{cfg['input_size']} px · floor {cfg['confidence']} · "
                     f"{classes} class{'' if classes == 1 else 'es'}",
                     fontsize=8.5, color=INK, pad=9, linespacing=1.5)
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("false alarms (%)", fontsize=8, color=INK2)

    axes[0].set_ylabel("peer recall (%)", fontsize=8, color=INK2)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK2,
                   handletextpad=0.4, borderpad=0.2)
    fig.suptitle("Same weights, same 284 frames, four launcher configurations",
                 fontsize=10.5, color=INK, y=1.06)
    fig.text(0.5, -0.11,
             "Every point is one checkpoint scored on the whole labelled day "
             "(60 peer-present frames, 221 peer-free). Only the preprocessing differs "
             "between panels.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "detector-configuration-spread.png"))
    ap.add_argument("--check", action="store_true",
                    help="re-derive the quoted recalls and exit non-zero if any moved")
    args = ap.parse_args()

    candidates, incumbent, profiles = load()
    if args.check:
        return check(incumbent)
    draw(candidates, incumbent, profiles, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
