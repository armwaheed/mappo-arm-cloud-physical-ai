#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Refuse to reach real robot hardware from a robot's SYSTEM Python.

Custom Python on a robot runs in a virtualenv. That sentence was in the deployment docs
and it did not stop anybody, so it is a check here instead: on a robot, a run that is
about to open a vendor transport and is NOT in a virtualenv raises ``SystemExit`` with a
message naming the venv to activate and the interpreter to build one from.

## The rule argues for package isolation, not version isolation

It is tempting to say "use the venv so you get the right Python". On this fleet that
would be false, and a rule that argues from a false premise gets discarded the first time
someone checks it. Measured on the lab Go2 (Orin NX, Ubuntu 20.04.5, aarch64) on
2026-08-26::

    /usr/bin/python3.8   3.8.10          <- what the venv was built from
    /usr/bin/python3.9   3.9.5
    ~/robotics-connect-envs/armwaheed/pyvenv.cfg:
        home = /usr/bin
        version = 3.8.10
        include-system-site-packages = true

The venv is 3.8 on a 3.8 system. It supplies **no version isolation whatsoever**, and
because it inherits system site-packages it is not even a clean room -- ``numpy`` and
``cv2`` import in both. What it supplies is *package* isolation for the one thing that
matters, and that half is total::

    module            /usr/bin/python3        ~/robotics-connect-envs/armwaheed
    cyclonedds        ModuleNotFoundError     0.10.2
    unitree_sdk2py    ModuleNotFoundError     editable install (egg-link)
    numpy             OK                      OK  (inherited from the system)
    cv2               OK                      OK  (inherited from the system)

So the vendor DDS stack exists **only** inside the venv. A ``pip install`` aimed at the
system Python writes to an interpreter every other user, every vendor tool and every ROS
node on that robot shares, and no ``uninstall`` puts a shadowed vendor package back the
way it was. Installing into the venv can, at worst, break one directory that can be
deleted and rebuilt. That asymmetry is the entire argument.

## A virtualenv cannot supply a Python the machine does not have

``python3.8 -m venv`` produces a 3.8 environment. That is why ``device-connect-edge``,
which requires >= 3.11, does **not** run on the Go2 at all: the robot has 3.8.10 and
3.9.5, ``apt-cache policy python3.11`` returns nothing and ``python3.10`` has no
candidate. No venv fixes that, and the hours to spend proving it belong to a demo day.
``dashboard/drive_bridge.py`` is the 3.8 half that does run there; see
``deploy/README.md`` for the whole split.

## What "on a robot" means here, and why it is two conditions and not one

The refusal fires only when a caller says *this run reaches real hardware* AND the host
shows *positive evidence* of being a robot compute module. Both are required, and each
one on its own is wrong:

* **Intent alone** would redden the offline suites. ``robot_bindings``'s own tests build
  ``--live`` argument namespaces on a laptop and on GitHub's runners specifically to
  assert on the refusal messages; ``--live`` is a statement about the flag, not about the
  machine.
* **Host alone** would refuse a shadow run, a telemetry replay or ``--help`` on the
  robot, none of which touch a leg or import a vendor SDK.

Evidence is checked in :func:`robot_host_evidence` and is **positive only** -- the
default answer is "not a robot", so a machine nobody anticipated is never refused. That
direction is deliberate and it has a cost: it is the shape of a gate that never fires.
The cost is paid down two ways rather than ignored. :func:`describe` returns a one-line
verdict that the live paths print on every run, so a guard that is not enforcing says so
out loud instead of being invisible; and ``MAPPO_ROBOT_HOST=1`` lets a host declare
itself, which is what the Lite3 deployment SOP sets, because the only robot this
repository has measured is the Go2 and inventing a Lite3 marker would be a guess wearing
a measurement's clothes.

## There is no bypass, because there is always a correct action

There is deliberately no ``MAPPO_ALLOW_SYSTEM_PYTHON``. An escape hatch documented next
to a refusal is an instruction to use it, and the reader most likely to reach for one is
an agent that wants an ``ImportError`` to go away. None is needed: if a machine really
does keep its vendor stack in the system Python, ``python3 -m venv
--system-site-packages`` inherits it -- measured above, that is exactly how the Go2's
venv sees ``numpy`` -- so activating a venv is correct there too. The only thing this
guard can be wrong about is whether the host is a robot, and ``MAPPO_ROBOT_HOST=0``
answers that question honestly rather than granting an exemption.

Python 3.8, standard library only, no imports from this repository: it has to be
importable by the interpreter that is about to be refused.

``python3 test_venv_guard.py``. Run ``python3 venv_guard.py`` to print this machine's
verdict.
"""

from __future__ import annotations

import os
import sys

#: Declares -- or denies -- a robot compute module. ``1/true/yes/on`` and ``0/false/no/off``
#: are both decisive; anything else is treated as unset, because a typo must not silently
#: disarm the guard. The Lite3 deployment SOP exports ``1``.
ROBOT_HOST_ENV = "MAPPO_ROBOT_HOST"

#: Continuous integration never drives a robot, and a self-hosted aarch64 runner would
#: otherwise trip the Jetson markers below. Checked before any filesystem evidence.
CI_ENV_VARS = ("GITHUB_ACTIONS", "CI", "GITLAB_CI", "JENKINS_URL", "BUILDKITE")

#: Present on the Go2's Orin NX -- measured 2026-08-26, first line reads
#: ``# R35 (release), REVISION: 3.1, ... EABI: aarch64``.
JETSON_RELEASE_FILE = "/etc/nv_tegra_release"

#: Fallback for a Tegra carrier that ships no release file. The Go2 reads
#: ``NVIDIA Orin NX Developer Kit``.
DEVICE_TREE_MODEL_FILE = "/sys/firmware/devicetree/base/model"
DEVICE_TREE_MARKERS = ("jetson", "orin", "xavier", "tegra")

#: Where a deployment keeps its virtualenvs. ``install.sh``'s default ``ENV_DIR`` is
#: ``~/robotics-connect-go2``; the lab Go2 was installed against a pre-existing
#: per-researcher env under ``~/robotics-connect-envs/``, which is why both are searched
#: and why the manifest is quoted in the message rather than either being called canonical.
VENV_SEARCH_GLOBS = (
    "~/robotics-connect-envs/*",
    "~/robotics-connect-go2",
    "~/robotics-connect-lite3",
    "~/mappo-env",
)

#: The install manifest that records which env an install run actually used.
DEPLOY_MANIFEST = "~/.mappo-go2-deploy.manifest"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

_PREFIX = "[venv-guard]"


class Decision:
    """The verdict, separated from the act of refusing so both halves are testable.

    :func:`evaluate` builds one and never raises; :func:`require_virtualenv` raises on it.
    A caller that wants to log rather than die -- or a test that wants to assert on the
    reasoning without catching ``SystemExit`` -- uses the first.
    """

    __slots__ = ("evidence", "message", "reason", "refuse")

    def __init__(self, refuse: bool, evidence: str | None, reason: str,
                 message: str | None = None) -> None:
        self.refuse = refuse
        self.evidence = evidence
        self.reason = reason
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Decision(refuse={self.refuse!r}, evidence={self.evidence!r}, "
                f"reason={self.reason!r})")


def in_virtualenv(prefix: str | None = None, base_prefix: str | None = None) -> bool:
    """True when this interpreter is running inside a virtualenv.

    ``sys.prefix != sys.base_prefix`` is the test, and it is the only one used here.
    ``VIRTUAL_ENV`` is an environment variable that ``bin/activate`` sets -- a wrapper
    script, a ``systemd`` unit, a ``subprocess`` call with a scrubbed ``env=``, ``sudo``
    without ``-E`` or a cron entry all reach the interpreter without it, and every one of
    those is a normal way to start something on a robot. ``sys.base_prefix`` is written
    by the interpreter itself at startup from ``pyvenv.cfg`` and no caller can forget it.
    """
    prefix = sys.prefix if prefix is None else prefix
    base_prefix = sys.base_prefix if base_prefix is None else base_prefix
    return prefix != base_prefix


def _declared(value: str | None) -> bool | None:
    """``True``/``False`` for a decisive value, ``None`` for unset or unrecognised."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in _TRUE:
        return True
    if cleaned in _FALSE:
        return False
    return None


def robot_host_evidence(env=None, read_file=None) -> str | None:
    """Positive evidence that this machine is a robot compute module, or ``None``.

    The returned string is quoted back at the operator, so it names the evidence rather
    than asserting a conclusion: it has to survive being read by somebody who thinks the
    guard is wrong.

    ``env`` and ``read_file`` are injected by the tests. ``read_file`` returns the file's
    text or ``None`` when it does not exist -- ``None`` rather than an exception because
    "absent" is the common case on every developer machine and is not an error.
    """
    env = os.environ if env is None else env
    read_file = _read_file if read_file is None else read_file

    declared = _declared(env.get(ROBOT_HOST_ENV))
    if declared is True:
        return f"{ROBOT_HOST_ENV}={env.get(ROBOT_HOST_ENV)}"
    if declared is False:
        return None

    for name in CI_ENV_VARS:
        if env.get(name):
            return None

    if read_file(JETSON_RELEASE_FILE) is not None:
        return f"NVIDIA Jetson carrier: {JETSON_RELEASE_FILE} exists"

    model = read_file(DEVICE_TREE_MODEL_FILE)
    if model:
        cleaned = model.replace("\x00", "").strip()
        if any(marker in cleaned.lower() for marker in DEVICE_TREE_MARKERS):
            return f"device-tree model: {cleaned!r}"

    return None


def _read_file(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None


def find_virtualenvs(globs=VENV_SEARCH_GLOBS, expanduser=None, iglob=None,
                     read_file=None) -> list:
    """``[(path, python_version)]`` for every virtualenv this machine already has.

    Best-effort and never raises: it is decoration on a refusal that is correct without
    it. The version comes from ``pyvenv.cfg``, which is what actually built the env --
    not from ``bin/python``, which on the Go2 is a symlink to a symlink to
    ``/usr/bin/python3`` and would report whatever that points at today.
    """
    import glob as _glob

    expanduser = os.path.expanduser if expanduser is None else expanduser
    iglob = _glob.iglob if iglob is None else iglob
    read_file = _read_file if read_file is None else read_file

    found = []
    for pattern in globs:
        for candidate in sorted(iglob(expanduser(pattern))):
            config = read_file(os.path.join(candidate, "pyvenv.cfg"))
            if config is None:
                continue
            version = ""
            for line in config.splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "version":
                    version = value.strip()
                    break
            found.append((candidate, version))
    return found


def refusal_message(component: str, evidence: str, executable: str, prefix: str,
                    venvs=None) -> str:
    """The whole refusal, as one string.

    Long on purpose. This text is the only thing between a tired operator and an hour of
    guessing, and every line of it either states a measurement or gives a command to run.
    Built as unprefixed lines and stamped once, so the wording is readable in the source.
    """
    venvs = [] if venvs is None else venvs
    if venvs:
        where = ["  Virtualenvs already on this machine:"]
        for path, version in venvs:
            suffix = f"  (Python {version})" if version else ""
            where.append(f"    source {path}/bin/activate{suffix}")
        where.append("  The one an install run actually used is recorded here:")
    else:
        where = ["  This machine has none that this guard can find. The one an install run",
                 "  used, if there was one, is recorded here:"]
    where.append(f"    grep '^env_dir' {DEPLOY_MANIFEST} | tail -1")

    lines = [
        f"REFUSING TO RUN: {component} is about to reach real robot hardware from this",
        "  robot's SYSTEM Python. Custom Python on a robot runs in a virtualenv.",
        "",
        f"  robot host  : {evidence}",
        f"  interpreter : {executable}",
        f"  sys.prefix  : {prefix}  (== sys.base_prefix, so this is NOT a virtualenv)",
        "",
        "WHY. The vendor stack is installed inside a virtualenv on this fleet and NOT in",
        "  the system Python. Measured on the Go2 2026-08-26: cyclonedds and unitree_sdk2py",
        "  import in ~/robotics-connect-envs/<user> and raise ModuleNotFoundError in",
        "  /usr/bin/python3. A pip install here writes to the interpreter every other user,",
        "  every vendor tool and every ROS node on this robot shares, and no uninstall puts",
        "  a shadowed vendor package back. A venv can be deleted and rebuilt; this cannot.",
        "",
        "WHAT TO DO. Activate a virtualenv, then re-run this exact command.",
        *where,
        "  If there is genuinely none, build one from THIS interpreter -- the same Python",
        "  the vendor stack was installed for, which is the only one that can import it:",
        f"    {executable} -m venv --system-site-packages ~/robotics-connect-envs/$USER",
        "    source ~/robotics-connect-envs/$USER/bin/activate",
        "  --system-site-packages is not optional: it is what the Go2's deployed venv uses",
        "  (pyvenv.cfg: include-system-site-packages = true) and it is how numpy and cv2",
        "  stay importable. It also means a stack that really does live in the system",
        "  Python is still visible from inside the venv, so there is nothing this refusal",
        "  can cost you.",
        "",
        "DO NOT pip install into the system Python to make an import succeed, and do not",
        "  try to install a newer Python on this robot -- the Go2 has 3.8.10 and 3.9.5, and",
        "  apt has no python3.11 candidate at all. If an import still fails INSIDE the venv,",
        "  that is a finding to report, not a dependency to add.",
        "",
        "Docs: deploy/README.md and robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md.",
        f"If this machine is NOT a robot, say so with {ROBOT_HOST_ENV}=0 rather than "
        "working around this.",
    ]
    return "\n".join(f"{_PREFIX} {line}".rstrip() for line in lines)


def evaluate(component: str, reaching_hardware: bool, env=None, read_file=None,
             prefix: str | None = None, base_prefix: str | None = None,
             executable: str | None = None, venvs=None) -> Decision:
    """Decide, without raising. See :class:`Decision`.

    ``reaching_hardware`` is the caller's assertion that this run opens a vendor
    transport -- ``args.live`` for the Lite3 binding, a non-``sim`` platform for the
    dashboard bridge. It is a parameter and not something sniffed from ``sys.argv``
    because only the call site knows, and a guard that guesses at intent is a guard that
    fires during ``--help``.
    """
    if not reaching_hardware:
        return Decision(False, None, "run does not reach real robot hardware")

    evidence = robot_host_evidence(env=env, read_file=read_file)
    if evidence is None:
        return Decision(
            False, None,
            f"no robot-host evidence on this machine (declare one with {ROBOT_HOST_ENV}=1)")

    if in_virtualenv(prefix=prefix, base_prefix=base_prefix):
        return Decision(False, evidence,
                        f"virtualenv active ({sys.prefix if prefix is None else prefix})")

    executable = sys.executable if executable is None else executable
    resolved_prefix = sys.prefix if prefix is None else prefix
    if venvs is None:
        try:
            venvs = find_virtualenvs()
        except Exception:  # pragma: no cover - decoration must never mask the refusal
            venvs = []
    return Decision(
        True, evidence, "system Python on a robot",
        refusal_message(component, evidence, executable, resolved_prefix, venvs))


def describe(decision: Decision) -> str:
    """One line naming what the guard decided, for the live paths to print.

    A guard nobody can see is indistinguishable from a guard that is broken, and this
    one's detection deliberately defaults to "not a robot" -- so every enforcing path
    prints this, including when the answer is "not enforced here".
    """
    if decision.refuse:
        return f"venv-guard: REFUSED -- {decision.reason} ({decision.evidence})"
    if decision.evidence:
        return f"venv-guard: pass -- {decision.reason}; robot host: {decision.evidence}"
    return f"venv-guard: not enforced -- {decision.reason}"


def require_virtualenv(component: str, reaching_hardware: bool, **kwargs) -> Decision:
    """Refuse, loudly, or return the :class:`Decision` that explains why it did not.

    Raises ``SystemExit`` with the full message. ``SystemExit`` rather than a custom
    exception because every call site here is a preflight whose other refusals already
    raise it -- ``robot_bindings.preflight_navigation`` raises
    ``SystemExit("[lite3] REFUSING TO WALK: ...")`` -- and one refusal type means one
    handler and one exit status.
    """
    decision = evaluate(component, reaching_hardware, **kwargs)
    if decision.refuse:
        raise SystemExit(decision.message)
    return decision


# ⚠️ Keep this at the END of the file. A `__main__` block placed mid-module stops
# everything below it from being defined when the file is run directly, and this
# repository has already lost ten tests across two files to exactly that.
if __name__ == "__main__":  # pragma: no cover
    _decision = evaluate("venv_guard --self-check", reaching_hardware=True)
    print(describe(_decision))
    print(f"  sys.executable   : {sys.executable}")
    print(f"  sys.prefix       : {sys.prefix}")
    print(f"  sys.base_prefix  : {sys.base_prefix}")
    print(f"  in_virtualenv()  : {in_virtualenv()}")
    print(f"  robot evidence   : {robot_host_evidence()}")
    for _path, _version in find_virtualenvs():
        print(f"  virtualenv       : {_path}  (Python {_version or '?'})")
    if _decision.refuse:
        print()
        print(_decision.message)
