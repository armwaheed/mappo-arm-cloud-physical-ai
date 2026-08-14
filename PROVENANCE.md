<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Provenance of `robot-stack/`

`robot-stack/` is a **vendored copy**, not a fork with a remote. It is Arm code,
Apache-2.0, copied here so this demo repository is self-contained for a collaboration
that reaches outside Arm.

Three things were changed on the way in, and all of them matter if you re-vendor:

- **The Device Connect driver was dropped.** `driver/` and its `pyproject.toml` package
  the Go2 as a Device Connect device. The demo never touches them — `visual_nav` imports
  nothing from either — so they are not carried here.
- **The shared core is imported as `arm_dc_robotkit`**, matching the Device Connect
  naming. It is one import, in `unitree/go2/locomotion/go2_locomotion.py`. Deploy the
  shared core's `lib/` directory under that name (see below).
- **Thirteen files were renamed away from the upstream product name.** This repository is
  shared outside Arm and carries the Device Connect naming throughout; upstream does not.
  Two are at the root and eleven under `unitree/go2/`, and **three of the eleven are
  `.py`** — so a sweep that assumes the renaming only touched prose misses them. See
  *Re-vendoring* below, which lists all thirteen, because a plain `rsync` puts every one
  back.

| | |
| --- | --- |
| Upstream | `github.com/the Arm Device Connect Go2 stack` (**internal** visibility) |
| Branch | `main` |
| Commit | `3f11b53` |
| Corresponds to | the planner-substitution seam and the public feasibility predicate, on top of the telemetry fields the policy integration needs |
| Copied | 2026-08-14 |

> Re-vendored 2026-08-14 after two upstream changes made for this demo and merged there
> first, which is the order `PROVENANCE.md` asks for: `visual_nav.main()` now accepts a
> `planner_factory`, and `DynamicWindowPlanner` has a public `is_feasible`. Together they
> let `integration/mappo_drive.py` substitute a controller through a supported seam
> instead of swapping three module globals, and delete about forty lines of it.
>
> **The recipe below under-syncs and was corrected in the same pass.** Its rsync filter is
> `*.py` only, but `visual_nav/README.md` and `visual_nav/SKILL.md` are NOT in the
> deliberately-different list — they are supposed to track upstream, and a `.py`-only sync
> silently leaves them behind. They are copied explicitly now.

> The previous entry recorded `4ceda53` on `feat/static-obstacle-nav`. That was wrong:
> the tree actually held `95550b8`, one commit later, which is a whole live-run fix pass
> — and the branch has since merged, so `main` is the ref to track. Verified by
> comparing each vendored file against both refs rather than trusting the table, which
> is the only way this kind of drift is ever found.

## Read this before editing `robot-stack/`

**Upstream is the source of truth.** A vendored copy that gets edited in place is how a
tree ends up carrying fixes that exist nowhere else — this project has been bitten by
exactly that three times, and each one was found by accident:

- `d1_arm/d1_fk.py` and `_arm_idl.py` existed **only on the robot**; `visual_nav` could
  not run from a clean clone at all (rescued in PR #11).
- `install/setup_env.sh` — referenced by five places in the tree, including the
  `FileNotFoundError` that tells you to source it — likewise (rescued in PR #14).
- `controller/panic_damp.py` and `controller/arm_panic.py` still exist only on the robot.

So: fix things upstream and re-vendor. If something must change here first, say so in the
commit message and open the upstream PR in the same session.

## Re-vendoring

**Do not `rsync` the whole tree.** The recipe that used to be here did, and it would have
undone all three of the changes listed above in one command: restoring `driver/` and
`pyproject.toml`, and putting the upstream product name back into ten files. None of that
fails a test, so it would have shipped.

Sync only the source, then put back the one rename that lives inside it:

```bash
git -C ../arm-dc-unitree-go2 fetch origin && git -C ../arm-dc-unitree-go2 checkout <ref>
rsync -a --delete --include '*/' --include '*.py' --exclude '*' \
      ../arm-dc-unitree-go2/unitree/ robot-stack/unitree/

# visual_nav's docs are NOT in the deliberately-different list, so they track upstream —
# and the .py-only filter above silently leaves them stale. Copy them explicitly.
cp ../arm-dc-unitree-go2/unitree/go2/visual_nav/{README.md,SKILL.md} \
   robot-stack/unitree/go2/visual_nav/

# Three of those .py files reference the shared core under the upstream name, so the
# rsync undoes the rename in exactly the files a prose-only sweep would miss. The
# upstream string is read out of upstream rather than written here, for the same reason
# it is not written anywhere else in this repository.
UPSTREAM_PKG=$(grep -ohm1 'arm_[a-z]*_robotkit' \
               ../arm-dc-unitree-go2/unitree/go2/locomotion/go2_locomotion.py)
grep -rl "$UPSTREAM_PKG" robot-stack/ \
  | xargs sed -i '' "s/$UPSTREAM_PKG/arm_dc_robotkit/g"
```

Then, before committing, confirm the only differences left are the intended ones:

```bash
diff -rq --exclude=.git --exclude=__pycache__ --exclude=.ruff_cache \
     ../arm-dc-unitree-go2/ robot-stack/
```

Expect exactly these thirteen, and nothing else — anything more is unintended drift:

| Deliberately different (13) | Deliberately absent |
| --- | --- |
| `README.md`, `SAFETY.md` | `driver/` |
| `unitree/go2/README.md` | `pyproject.toml` |
| `unitree/go2/{connect,install,lidar_sight}/SKILL.md` | |
| `unitree/go2/locomotion/{README.md,SKILL.md,go2_locomotion.py}` | |
| `unitree/go2/{deploy,depth_camera_sight}/README.md` | |
| `unitree/go2/deploy/go2_robot_io.py` | |
| `unitree/go2/lidar_sight/go2_lidar_sight.py` | |

Finally, sweep for the names this repository does not carry — the upstream product name,
its acronym, and the assistant vendor that appears in upstream commit trailers. They are
not written here either; take the three terms from an upstream `README.md` title line and
an upstream `git log`, then check **both** the tree and the history, because a trailer is
not a file:

```bash
grep -rniE "<term1>|<term2>|<term3>" --exclude-dir=.git . && echo "LEAKED — do not commit"
git log --format='%B%n%an %ae' | grep -icE "<term1>|<term2>|<term3>"   # must print 0
```

Update the commit in the table above, and re-run both test suites.

## Deploying to the robot

The stack imports `arm_dc_robotkit`, which is **not installed on the Go2** and whose
packaging declares `requires-python >=3.10` against the robot's Python 3.8.10 — so `pip`
refuses it. Deploy the package directory instead:

```bash
rsync -a <robotkit>/lib/ unitree@<robot>:~/deps/arm_dc_robotkit/
ssh unitree@<robot>
source ~/<tree>/unitree/go2/install/setup_env.sh    # MANDATORY — RPC segfaults without it
export PYTHONPATH=~/deps
```
