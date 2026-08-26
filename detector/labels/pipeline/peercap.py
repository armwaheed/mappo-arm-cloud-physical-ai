#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the peercap frames are, and what these scripts do when they are not on this box.

Every script in this directory used to open one hard-coded absolute path -- a local
scratchpad directory, `/private/tmp/.../scratchpad/peercap/`, that no longer exists.

That is not a cosmetic defect, and this repository has already paid for it. The same
string was recorded as `source_dir` in `detector/labels/peer_go2wheel_20260824.json`, and
on 2026-08-26 it was read as *the corpus has been deleted*: 2,800 frames were written off
in issue #77 before anyone looked at the machine the training had actually run on. They
were never lost. #86 pointed both manifests at a location a reader can check and kept the
dead path under `source.was` so the lesson survives; it named these scripts as the rest of
the same defect and left them alone. This is that half.

So no location is hard-coded here. `PEERCAP_FRAMES` names the directory holding the
frames, and every script in this directory refuses without it, printing the locations the
corpus is *reported* to be in together with which of them this repository has verified
(none of them -- the frames are not in the repository and cannot be).

**Refusing is the whole point.** The previous failure mode was not an error message: it
was `glob.glob` returning `[]` against a missing directory and an `IndexError` on `f[0]`
twenty lines later, which reads as a broken script rather than as a corpus that is
somewhere else. A tool that cannot find its input should say what the input is and where
it was last seen.

    export PEERCAP_FRAMES=~/go2-peer-dataset-20260824      # the 2,800 jpg
    export PEERCAP_WORK=...                                # optional, see below
    export PEERCAP_LABELLED=...                            # optional, see below

`PEERCAP_WORK` (per-segment intermediates) and `PEERCAP_LABELLED` (where `build.py`
writes `annotations.json`) default to siblings of the frames directory and are created on
demand, so naming the one durable thing is enough to run the whole pipeline. They are
separable because the frames may sit on a read-only mount or a shared dataset directory.
"""
import os
import sys
from pathlib import Path
from typing import NoReturn

FRAMES_ENV = "PEERCAP_FRAMES"
WORK_ENV = "PEERCAP_WORK"
LABELLED_ENV = "PEERCAP_LABELLED"

#: Reported locations, and what each one is worth to a reader. Neither was verified from
#: this repository, and both are recorded that way in the manifests by #86 -- the Hugging
#: Face dataset is not public, and an unauthenticated request cannot tell a private
#: dataset apart from one that does not exist, so it is not the location to check first.
WHERE_THE_CORPUS_IS = """\
The 2,800 frames of the 2026-08-24 staged capture are reported to be in two places.
NEITHER has been verified from this repository:

  arm-seattle-spark-02:~/go2-peer-dataset-20260824/   2,800 jpg, 589 MB   (issue #77)
  Hugging Face dataset armwaheed/go2-peer-detection   not public          (issue #77)

The 1,903 boxes are IN this repository and need none of the above:

  detector/labels/peer_go2wheel_20260824.json         checked by check_manifest.py

Do not point PEERCAP_FRAMES at a scratchpad directory. These scripts carried
`/private/tmp/.../scratchpad/peercap/` until 2026-08-26, and reading that one dead string
as the corpus's only location is why 2,800 frames were declared lost in issue #77. They
were on the training host the whole time. See #86.\
"""


def _refuse(problem: str) -> NoReturn:
    """Exit with the problem and the locations, on stderr. Never returns."""
    sys.stderr.write(
        f"\n{problem}\n\n"
        f"Set {FRAMES_ENV} to the directory holding the frames"
        f" (p1_close_broadside_0000.jpg, ...).\n\n"
        f"{WHERE_THE_CORPUS_IS}\n"
    )
    raise SystemExit(2)


def frames_dir() -> str:
    """The directory holding the capture's jpgs, with a trailing separator.

    Refuses if it is unset, absent, or holds no jpg. The last of those matters: a
    directory that exists but is empty is exactly how a moved corpus presents, and
    without the check every downstream count would come out zero and look like a result.

    The jpg test is deliberately case-sensitive and deliberately `*.jpg`: that is the
    exact glob every consumer in this directory uses, so the pre-flight is neither
    stricter nor looser than the thing it is standing in front of. A corpus this refuses
    is a corpus the scripts would have read as empty.

    The trailing separator is `/`, not `os.sep`. Every call site here concatenates
    (`SRC + tag + "_[0-9]..."`) and splits on `"/"`; this directory is POSIX throughout.
    """
    raw = os.environ.get(FRAMES_ENV, "").strip()
    if not raw:
        _refuse(f"{FRAMES_ENV} is not set, so there is nothing to read.")
    path = Path(raw).expanduser()
    if not path.is_dir():
        _refuse(f"{FRAMES_ENV}={raw} is not a directory.")
    if next(path.glob("*.jpg"), None) is None:
        _refuse(f"{FRAMES_ENV}={raw} holds no .jpg -- this is not the capture.")
    return str(path) + "/"


def _sibling(env: str, name: str) -> str:
    """A writable directory: `env` if set, else `name` beside the frames. Created."""
    raw = os.environ.get(env, "").strip()
    path = Path(raw).expanduser() if raw else Path(frames_dir()).parent / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + "/"


def work_dir() -> str:
    """Per-segment intermediates -- plates, blob lists, tracked boxes."""
    return _sibling(WORK_ENV, "peercap_work")


def labelled_dir() -> str:
    """Where `build.py` writes `annotations.json`."""
    return _sibling(LABELLED_ENV, "peercap_labelled")
