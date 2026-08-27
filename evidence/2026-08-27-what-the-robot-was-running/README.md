<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — the tree the robot actually ran, and why it is 20 KB and not 3.8 MB

Until tonight the lab Go2's launch wrappers sourced `/home/unitree/mappo-main`. That tree
**matched no commit**, so no run made from it could name the code it executed. It has now
been superseded by a stamped deploy, and this directory is what was preserved of it.

## What it was

**134 files spanning 15 commits across eight days**, 2026-08-11 → 2026-08-19.

| | files |
|---|---|
| identical to `main` | 86 |
| last current at an older commit | 46, across **15** distinct commits |
| held at that path by no commit at all | 2 |

The two are `integration/mappo_drive.py.bak-preberth` and
`robot-stack/unitree/go2/visual_nav/visual_nav.py.orig` — editor leftovers that git has
never tracked at those paths, and the reason dropping them still resolves to nothing.

Oldest contributing commit: `c69016e` (2026-08-11, *"MAPPO demo: robot stack, telemetry
contract"*). Newest: five commits from 2026-08-19. So a run in late August was executing a
mixture up to **eight days stale**, silently, and the telemetry recorded none of it.

## Why the tree itself is not committed here

**Every one of the 134 files is a blob this repository already holds.** They were simply
never assembled into one commit. Committing the tree would add 3.8 MB of duplicate content,
and — worse — would put a tree that resolves to no version *inside* the repository, where
the next person to find it will reasonably assume it is one. That confusion is the thing
this directory exists to end, not to propagate.

So the artefact is the **manifest**: `path`, `blob sha`, and the commit that blob was
current at. 20 KB, diffable, greppable, and sufficient.

Even the two untracked leftovers need no bytes stored — git holds their content under other
paths, so the manifest addresses them by sha like everything else.

## Rebuild it

```bash
bash rebuild.sh /tmp/mappo-main-reconstructed
```

**Verified, not asserted.** Rebuilding from `manifest.tsv` and diffing against the tarball
pulled off the robot on 2026-08-27 reports no differences across all 134 files. The check
is the one worth repeating if you ever doubt this page:

```bash
bash rebuild.sh /tmp/rebuilt
diff -r --exclude=__pycache__ --exclude='*.pyc' /tmp/rebuilt <tarball extracted>
```

## What this does not cover

The manifest reconstructs the tree; it does not tell you which run used which file. Runs
before 2026-08-27 carry no commit identity at all — that is precisely the gap the deployed
stamp closes, and it closes it going forward only. Every measurement published from a run
before tonight rests on the tree described here, and on nothing more specific than that.

Robot-side telemetry and video from those runs — 84 videos and 81 telemetry files, 536 MB —
were pulled off the robot the same night. They do not belong in git; they belong in the
dataset store, keyed back to this manifest.
