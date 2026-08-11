<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Provenance of `robot-stack/`

`robot-stack/` is a **vendored copy**, not a fork with a remote. It is Arm code, Apache-2.0,
copied here so this demo repository is self-contained for a two-person collaboration.

| | |
| --- | --- |
| Upstream | `github.com/arm/arm-mhs-unitree-go2` (**internal** visibility) |
| Branch | `feat/static-obstacle-nav` |
| Commit | `4ceda535be7208c24e92e65b7dec66a3331988d2` |
| Corresponds to | [PR #14](https://github.com/arm/arm-mhs-unitree-go2/pull/14), which builds on merged PRs #10 and #11 |
| Copied | 2026-08-11 |

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

```bash
git -C ../arm-mhs-unitree-go2 fetch origin && git -C ../arm-mhs-unitree-go2 checkout <ref>
rsync -a --delete --exclude '.git' --exclude '__pycache__' --exclude '.ruff_cache' \
      ../arm-mhs-unitree-go2/ robot-stack/
# then update the commit above, and re-run both test suites
```

## Deploying to the robot

The stack imports `arm_mhs_robotkit`, which is **not installed on the Go2** and whose
packaging declares `requires-python >=3.10` against the robot's Python 3.8.10 — so `pip`
refuses it. Deploy the package directory instead:

```bash
rsync -a ../arm-mhs-robotkit/lib/ unitree@192.168.123.18:~/deps/arm_mhs_robotkit/
ssh unitree@192.168.123.18
source ~/<tree>/unitree/go2/install/setup_env.sh    # MANDATORY — RPC segfaults without it
export PYTHONPATH=~/deps
```
