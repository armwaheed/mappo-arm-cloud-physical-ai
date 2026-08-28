#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Push the dataset and the checkpoints a reader needs, to the PRIVATE corpus repo.

    python3 publish_to_hub.py --selection published_checkpoints.json \\
        --runs ~/lite3-ratio-20260827/runs --dataset ~/lite3ds-20260827 --dry-run
    ... then without --dry-run.

⛔ **THE TOKEN COMES FROM STDIN AND IS NEVER WRITTEN DOWN.** The training host is shared.
A token in ``argv`` is in every user's ``ps``; a token in the environment is in
``/proc/<pid>/environ``; a token written to ``~/.cache/huggingface/token`` outlives the
process and grants whatever it grants to whoever has the account afterwards. So it is read
from stdin, held in a local, and passed to each call:

    ssh host 'python3 publish_to_hub.py ... ' < token-file

It must be a **fine-grained token scoped to this one repository**. A fine-grained token
cannot CREATE a repo, which is why the target is the existing one rather than a new one —
and that is the better home anyway: ``aug20_crossday/`` already lives there, and it is the
284-frame cross-day corpus every person number in this wave is measured on. Weights and the
frames they were scored against belong behind one access grant, not two.

⛔ **PRIVATE, AND IT STAYS PRIVATE.** The corpus contains an identifiable person. This
script refuses to run if the repo is public, rather than trusting that it is not, and it
never passes ``private=False`` anywhere.

⚠️ **WHY WEIGHTS AT ALL, WHEN THE TRAINER IS ON GITHUB.** Because re-running it does not
reproduce them. ``probe_determinism.sh`` runs the same command five times with ``--seed 0``
and gets five different ``.caffemodel`` files. A checkpoint here is therefore not a cache of
something regenerable — it is the only copy, and a negative result whose weights are gone
cannot be checked by anyone.

What goes up, and the reason each is not optional:

``lite3_20260827/images/``      the labelled corpus, 3,179 frames
``lite3_20260827/*.json``       the manifests, including the eval split
``runs/<arm>/history.json``     per-epoch losses and the full argv of every arm
``runs/<arm>/pseudo_labels_*``  the teacher's old-class supervision. It differs between 224
                                and 300 and CANNOT be rebuilt from the manifests, so the
                                teacher confound is unverifiable without it.
``runs/<arm>/epoch*.caffemodel``  only those ``select_for_publication.py`` names, by rule.
``PUBLISHED.md``                what went up, what stayed on the training host, and where.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "armwaheed/go2-peer-detection"
REPO_TYPE = "dataset"
#: One prefix, so this wave never collides with what is already in the repo.
PREFIX = "lite3_20260827"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path(),
                        help="this directory, for the card and the scored JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())

    uploads: list = []            # (local path, path in repo)
    for name in sorted(args.dataset.glob("*.json")):
        uploads.append((name, f"{PREFIX}/manifests/{name.name}"))
    images = sorted(args.dataset.glob("images/*.jpg"))
    for arm, block in sorted(selection["arms"].items()):
        for extra in ("history.json", "pseudo_labels_224.json", "pseudo_labels_300.json"):
            path = args.runs / arm / extra
            if path.exists():
                uploads.append((path, f"{PREFIX}/runs/{arm}/{extra}"))
        for model in sorted(block["checkpoints"]):
            path = args.runs / arm / model
            if not path.exists():
                raise SystemExit(f"{path} is named in the selection and is not on disk")
            uploads.append((path, f"{PREFIX}/runs/{arm}/{model}"))
    for name in sorted(args.evidence.glob("scored_*.json")):
        uploads.append((name, f"{PREFIX}/scores/{name.name}"))
    for name in sorted(args.evidence.glob("incumbent_*.json")):
        uploads.append((name, f"{PREFIX}/scores/{name.name}"))
    for name in ("run_facts.json", "published_checkpoints.json", "determinism.json"):
        path = args.evidence / name
        if path.exists():
            uploads.append((path, f"{PREFIX}/scores/{name}"))

    total = sum(p.stat().st_size for p, _ in uploads) + sum(
        p.stat().st_size for p in images)
    print(f"  {len(uploads)} files + {len(images)} frames = "
          f"{total / 1e9:.2f} GB -> {REPO_TYPE}:{REPO}/{PREFIX}/")
    print(f"  {selection['published_checkpoints']} checkpoints published, "
          f"{selection['held_on_training_host']} held on the training host "
          f"({selection['held_bytes'] / 1e9:.2f} GB)")
    if args.dry_run:
        for path, target in uploads[:6]:
            print(f"    {path.name:34s} -> {target}")
        print(f"    ... and {len(uploads) - 6} more, plus images/ as a folder")
        return

    # ⚠️ ~/.cache/huggingface/{xet,hub,datasets} are ROOT-OWNED on this shared training
    # host, so the Xet uploader cannot write its own chunk cache or its log and dies with
    # `OSError: Permission denied (os error 13)` several seconds into the transfer. Point
    # every cache at a directory this user owns. Setting HF_HOME alone is not enough --
    # HF_XET_CACHE is read independently, and the token is passed on stdin rather than
    # through HF_HOME so moving the home does not hide it.
    cache = Path.home() / "lite3-ratio-20260827" / ".hf-cache"
    (cache / "xet").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(cache / "xet"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

    token = sys.stdin.read().strip()
    if not token:
        raise SystemExit("no token on stdin. See the module docstring; it is never a flag.")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.repo_info(REPO, repo_type=REPO_TYPE)
    if not info.private:
        raise SystemExit(f"{REPO} is PUBLIC. This corpus contains an identifiable person "
                         f"and must not be published. Refusing.")
    print(f"  {REPO} is private — proceeding")

    api.upload_folder(folder_path=str(args.dataset / "images"), repo_id=REPO,
                      repo_type=REPO_TYPE, path_in_repo=f"{PREFIX}/images",
                      commit_message=f"{PREFIX}: the labelled Lite3 corpus, 3,179 frames")
    print(f"  uploaded {len(images)} frames")
    for path, target in uploads:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=REPO,
                        repo_type=REPO_TYPE)
    print(f"  uploaded {len(uploads)} files")


if __name__ == "__main__":
    main()
