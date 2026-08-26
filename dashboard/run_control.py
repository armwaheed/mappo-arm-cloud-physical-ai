#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What it takes to start an autonomous MAPPO run from a button, stated as a command line.

⛔ **``robot-stack/SAFETY.md`` governs this file.** A live run hands the legs to a learned
policy, which is a larger commitment than any nudge on the motion pad.

This module is the pure half of run control: it decides whether a run may start, builds the
exact argv that will run, renders it for whichever machine will run it, and builds the
command that stops it. It owns no process and imports nothing from Device Connect, so every
refusal in here can be tested without a robot, a mesh or an event loop.
``robot_driver.MappoRobotDriver`` owns the process and the events.

## Where the run actually runs, said plainly

``integration/mappo_drive.py`` needs the robot's camera, its detector and its DDS stack, so
it runs **on the robot**, under the robot's Python 3.8. ``robot_driver.py`` needs
``device-connect-edge``, which requires Python >= 3.11, which the Go2's Jetson does not have
and cannot be given. So the two are on different machines and the driver reaches the run the
only way it can: **it spawns a process over SSH.**

That is not elegant and it is not hidden. It is the same wall ``drive_bridge.py`` exists for
and the same shape ``go2_frame_server.py`` uses from the other side; naming it is better than
a `launch()` that reads like a local call and is not one. What it costs is written down:

* **A remote run does not die when this driver does.** SIGTERM to the local ``ssh`` client
  closes a socket; it does not signal the process on the far end. So the launch writes the
  remote PID to a file and :func:`stop_command` signals **that** — SIGTERM, never SIGKILL
  (``SAFETY.md`` §0). A driver that is hard-killed leaves the run driving, bounded only by
  the run's own ``--max-seconds`` and its own ``SafeStop``.
* **The stop can fail to confirm**, because it is a second SSH round trip. The caller is
  told which, so an unconfirmed stop reads like ``STOP ALL``'s unconfirmed robot: the one
  you now have to walk over to.
* **Nothing here can name the code it launches.** See :attr:`RunProfile.tree_note`.

## The gate, which is two gates and a default that cannot move a robot

**The default run does not move the robot, and is not capable of moving it.**
``start_run()`` with no arguments builds the command line ``run-smoke.sh scene`` builds:
perception, the policy, the veto and the telemetry, with **no** ``--live``. ``--live`` is
the only flag in ``mappo_drive.py`` that commands a leg, and a command line without it has
no path to one — this is not a permission that is checked, it is a capability that is
absent. That is what makes it a safe thing for a button to do with no flags.

**Motion is opted into twice, at two layers, by two different people.**

1. The **driver** must have been started with ``--allow-motion`` — an operator's decision,
   made at the command line, with a clear area and somebody on the abort. Exactly the gate
   ``walk_forward`` has, and the same one again.
2. The **request** must ask for it: ``start_run(arm_motion=true)``. A page cannot get a
   live run by omitting a field, and an agent cannot get one by calling with defaults.

**Neither gate downgrades.** ``arm_motion`` on a driver without ``--allow-motion`` is a
REFUSAL with the reason in it, not a quiet dry run. A silent downgrade is the worse failure
of the two: the operator asked for a live run, watched a run start, and is now watching a
robot that was never going to move — which is the same shape as ``mode='mcf'`` and the same
shape as a sub-gait-floor command, both of which this repository refuses rather than warns
about.

:func:`build_run_argv` enforces gate 1 in the same words ``drive_bridge.dispatch`` refuses a
motion command without ``--allow-motion``, and the driver enforces it again before spawning
anything. That duplication is deliberate: two processes with one shared assumption is one
process with an unwritten contract.

**A run is gated harder than a nudge, and the extra gate is a bound.** A nudge is capped at
``drive_bridge.MAX_SECONDS`` = 5 s, which is what stops a web button starting a walk that
outlives the operator's attention. An autonomous run has no such ceiling of its own, so this
module supplies one: :data:`MAX_RUN_SECONDS`, clamped here, passed as ``--max-seconds``, and
backstopped by a watchdog in the driver.

**And a run is refused on a robot that would silently ignore it.** See
:func:`check_control_mode`.

Python 3.8, stdlib only, no Device Connect. ``python3 test_run_control.py``.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Hard ceiling on one dashboard-started run, seconds. The motion pad's equivalent is
#: ``drive_bridge.MAX_SECONDS`` (5 s); a run needs a much larger number to be useful and
#: still needs one, for the same reason: nothing a browser starts should be able to outlive
#: the attention of the person who started it. 120 s is about three times the longest run in
#: ``evidence/`` (``--max-seconds 45`` in the runbook, 40 in ``run-peer-supervised.sh``).
MAX_RUN_SECONDS = 120.0

#: What ``start_run`` uses when the caller says nothing. Deliberately shorter than the
#: runbook's 45: the first thing anybody does from a dashboard is press the button to see
#: what happens, and the default should be the length of a look rather than of a demo.
DEFAULT_RUN_SECONDS = 30.0

#: Added to a run's own bound before the driver's watchdog terminates it. It is NOT the
#: run's duration budget — it is startup: opening the camera, loading the detector, DDS
#: discovery, and the policy package. On a cold Jetson a single ``drive_bridge status`` is
#: 1.94 s of SDK import and discovery alone (measured on the lab Go2, three runs each on two
#: sessions on 2026-08-26), and a full ``visual_nav`` startup is a multiple of that.
RUN_STARTUP_GRACE_S = 30.0

#: How long to wait for each half of stopping a run: the SSH round trip that signals it, and
#: then the run's own exit. Short, and a miss is REPORTED rather than retried — an operator
#: whose stop did not confirm needs to know that now, not after a retry loop.
#:
#: **The number is set by what it has to fit inside.** ``server.STOP_TIMEOUT_S`` is 12 s and
#: is the dashboard's ceiling on one stop, deliberately short so that a stop which has not
#: landed becomes a reason to reach for the physical abort rather than to watch a spinner.
#: ``stop`` spends at most this on the signal, this again on the exit, and then one bridge
#: ``stop`` — 1.94 s on the lab Go2's Jetson, measured. 4 + 4 + 2 = 10 s, which leaves a
#: margin rather than racing its own caller.
RUN_STOP_TIMEOUT_S = 4.0

#: ``mappo_drive.py --policy-mode``. ``raw`` removes the planner's feasibility veto; the
#: closed-loop simulation had raw colliding in every configuration tested and supervised not.
POLICY_MODES = ("supervised", "raw")

#: ``mappo_drive.py --heading-servo``. ``off`` is the default since issue #16 and is the only
#: configuration that has not driven a robot into something. ``travel`` is issue #16's own
#: control law, kept so those runs stay reproducible.
HEADING_SERVOS = ("off", "goal", "travel")

#: The two spellings of the heading servo, and which tree understands which. See
#: :attr:`RunProfile.heading_servo_flag`.
SERVO_SPELLINGS = ("modern", "legacy")

#: Go2 motion-service modes in which ``SportClient.Move`` is accepted. See
#: :func:`check_control_mode` — this is not a style list, it is the difference between a
#: robot that walks and a robot that agrees with every command and never steps.
SPORT_MODES = ("normal", "ai")

#: Flags this module always spells for itself, and therefore refuses to accept from a
#: profile. A profile that could set ``--live`` would be a second, quieter motion gate, and
#: one that could set ``--policy-mode`` would make the operator's choice on the page a
#: suggestion. ``--no-heading-servo`` is in here because it is ``--heading-servo off`` under
#: another name and argparse would let the later one win silently.
RESERVED_FLAGS = (
    "--live", "--policy-mode", "--heading-servo", "--no-heading-servo",
    "--max-seconds", "--package", "--telemetry", "--record",
)


class RunRefused(RuntimeError):
    """A run that will not be started, reported as JSON rather than as a traceback.

    Named for what it is. ``drive_bridge`` raises :class:`~drive_bridge.BridgeError` for the
    same purpose one layer down, and the driver turns both into ``{"ok": false,
    "refused": true, "error": ...}`` so the page can colour a refusal differently from a
    fault. A refusal is the interesting event.
    """


@dataclass(frozen=True)
class RunProfile:
    """Where a run happens and what is constant about it, fixed when the driver starts.

    **Split this way on purpose.** Everything in here is a property of the DEPLOYMENT — the
    camera calibration file, the goal class, the robot's own radius, which machine the code
    is on — and none of it is a decision an operator makes per press. It is the same
    argument ``--model-sources`` makes: the robot's deployment knows these, a dashboard does
    not, and a browser must not be able to change them. What the RPC chooses is the short
    list that actually varies: how long, supervised or raw, servo or not, live or dry.

    That split is also what keeps a browser out of the robot's command line. ``extra_args``
    comes from a file on the machine running the driver; the RPC's parameters are checked
    against fixed tuples. There is no path from a text field to an argv element.
    """

    #: What to call this in the UI. Not an identifier.
    label: str = "local"
    #: Directory to run from, on the machine that will run it.
    workdir: str = "."
    #: Interpreter on THAT machine. On a Go2 it is the SDK venv's python, not this one —
    #: the same trap as ``--bridge-python``, which reports itself for the same reason.
    python: str = "python3"
    #: The script, relative to ``workdir`` or absolute.
    script: str = "mappo_drive.py"
    #: ``--package``: the policy package holding config.json and models/.
    package: str = "../policy"
    #: A shell file to source before the run — the robot's own ``setup_env.sh``. Empty for
    #: a local run.
    env_setup: str = ""
    #: Environment variables to export after sourcing :attr:`env_setup`, as ``NAME=value``
    #: strings. In JSON it is an object.
    #:
    #: **This is the half ``env_setup`` does not cover, and the split is not the obvious
    #: one.** Sourcing the Go2's ``setup_env.sh`` supplies two things:
    #:
    #: * it **prepends the venv to PATH**, so a bare ``python3`` is the interpreter that has
    #:   the SDK. ``cyclonedds`` lives only in that venv — ``/usr/bin/python3 -c 'import
    #:   cyclonedds'`` is a ``ModuleNotFoundError`` on this machine (checked 2026-08-26).
    #:   So :attr:`python` being ``"python3"`` is correct *because* the source ran, and a
    #:   profile that "helpfully" names an absolute system interpreter would break it.
    #: * it exports ``LD_LIBRARY_PATH``, and **that** is what stops the SDK segfaulting.
    #:   The Jetson ships two 0.10.2 builds of ``libddsc`` and ``ldconfig`` resolves to the
    #:   wrong one, so an RPC client crashes at rc 139 while serializing its first request,
    #:   before any banner — a segfault, not a hang, and not a broken venv.
    #:   ``setup_env.sh``'s own header documents it; diagnosed on this unit 2026-07-08.
    #:
    #: What it does **not** set is ``PYTHONPATH``, and ``unitree_sdk2py`` and
    #: ``arm_dc_robotkit`` live outside the venv's site-packages
    #: (``/home/unitree/unitree_sdk2_python`` and ``/home/unitree/deps``). That is what this
    #: field carries, and it is why ``/home/unitree/run-smoke.sh`` — the known-good wrapper
    #: this profile shape is copied from — exports it on the line *after* the source.
    env: tuple = ()
    #: The command that reaches the machine the run happens on, e.g.
    #: ``["ssh", "-o", "BatchMode=yes", "unitree@192.168.123.18"]``. **Empty means the run
    #: is a child of this process**, which is right for ``--platform sim`` and wrong for
    #: every real robot, because the driver cannot be on one.
    launch_prefix: tuple = ()
    #: The deployment's own constants, verbatim. Validated against :data:`RESERVED_FLAGS`.
    extra_args: tuple = ()
    #: Where telemetry and any recording are written, on the run's machine. Empty means the
    #: run writes neither, which is a run with no evidence — see :func:`build_run_argv`.
    output_dir: str = ""
    #: Whether to pass ``--record``. Off by default: the recorder's codec check fails a run
    #: outright on a host without one, and a demo that cannot start is worse than a demo
    #: that is not filmed.
    record: bool = False
    #: Which spelling of the heading servo the tree at :attr:`workdir` understands.
    #:
    #: ⚠️ **This exists because of a measurement, not a preference.** The flag was renamed
    #: in #106: ``--heading-servo {off,goal,travel}`` replaced a bare ``--no-heading-servo``,
    #: and the old flag survives on ``main`` only as a hidden alias. The lab Go2's
    #: ``~/mappo-main`` is 67 commits behind that, and on 2026-08-26 its own ``--help``
    #: listed ``[--no-heading-servo]`` and **no** ``--heading-servo`` at all. Spelling the
    #: new flag at that tree makes argparse exit 2 with a usage message, on the far end of
    #: an SSH connection, and the run never starts.
    #:
    #: ``"modern"`` writes ``--heading-servo <mode>``. ``"legacy"`` writes
    #: ``--no-heading-servo`` and REFUSES any mode but ``off``, because a tree that predates
    #: the rename cannot select one — see :func:`servo_flags`.
    heading_servo_flag: str = "modern"
    #: Where the PID of a remote run is written so it can be signalled. Ignored locally.
    pidfile_dir: str = "/tmp"
    #: What is known about the code at ``workdir``. Filled in by :func:`describe`.
    tree_note: str = ""
    #: Set by ``load_profile`` so a failure can name the file it came from.
    source_path: str = field(default="", compare=False)


#: What every launch target says about its own provenance, because none of them can say more.
#:
#: **The deployed tree is not a checkout.** It has no ``.git``, so it has no branch and no
#: commit, and nothing on the robot reports its own staleness. Measured on the lab Go2 on
#: 2026-08-26: ``~/mappo-run`` matched no single commit on ``main`` — its
#: ``integration/mappo_drive.py`` last matched a commit **43** behind ``main``, and its
#: ``dashboard/drive_bridge.py`` and ``dashboard/robot_driver.py`` a *different* commit,
#: **50** behind. Two files, two commits, one directory.
#:
#: So this API reports the one thing it can actually name — **the command**, in full, in the
#: reply and in the ``run_started`` event — and states that it cannot name the code. A run
#: started from this dashboard is evidence about a command line, not about ``main``.
TREE_NOTE = (
    "the launch target is a directory, not a git checkout: no .git, so no branch and no "
    "commit, and nothing on it reports its own staleness. Measured on the lab Go2 on "
    "2026-08-26: ~/mappo-main -- the tree the known-good run-smoke.sh actually runs, and "
    "whose name says 'main' -- matched NO single commit on main. Its "
    "integration/mappo_drive.py last matched a commit 67 behind, its policy/config.json one "
    "95 behind and its README.md one 93 behind: three files, three commits, one directory. "
    "The other tree beside it, ~/mappo-run, was 43 and 50 behind on a different pair. This "
    "API names the COMMAND it ran, which is reported in full; it cannot name the code."
)


def load_profile(path: str) -> RunProfile:
    """Read a run profile, or refuse to start.

    A malformed profile is a startup error and not a silently empty one, for the reason
    ``_load_sources`` gives: a driver that quietly advertises no way to start a run looks
    exactly like a driver deliberately configured without one.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object describing one run profile")

    # `_comment` is the one key a profile may carry that this does not read. JSON has no
    # comments and the alternative is a README nobody opens beside the file they are editing.
    data.pop("_comment", None)
    unknown = sorted(set(data) - {f for f in RunProfile.__dataclass_fields__ if f != "source_path"})
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}. A typo in a profile key is a "
                         f"setting that silently did not apply.")
    check_no_credential(data.get("launch_prefix", []), data.get("env", {}), path)
    if data.get("heading_servo_flag", "modern") not in SERVO_SPELLINGS:
        raise ValueError(f"{path}: 'heading_servo_flag' must be one of "
                         f"{list(SERVO_SPELLINGS)} — which spelling of the servo flag the "
                         f"tree at 'workdir' understands. Run its own --help and look.")
    for name in ("workdir", "script"):
        if not data.get(name):
            raise ValueError(f"{path}: '{name}' is required; there is no sensible default "
                             f"for where somebody else's robot keeps its code")
    for name in ("launch_prefix", "extra_args"):
        value = data.get(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{path}: '{name}' must be a list of strings")
        data[name] = tuple(value)
    data["env"] = _env_pairs(data.get("env", {}), path)

    profile = RunProfile(**data)
    check_extra_args(profile.extra_args, source=path)
    return replace(profile, source_path=path)


#: A shell variable name. Anything else is a typo that would become a shell syntax error on
#: the far end, seconds later, in a log nobody is reading.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _env_pairs(env, path: str = "") -> tuple:
    """``{"PYTHONPATH": "/a:/b"}`` as ``("PYTHONPATH=/a:/b",)``, or a startup error."""
    if isinstance(env, (list, tuple)):
        pairs = [str(item) for item in env]
    elif isinstance(env, dict):
        pairs = [f"{name}={value}" for name, value in sorted(env.items())]
    else:
        raise ValueError(f"{path}: 'env' must be an object of NAME: value")
    for pair in pairs:
        name = pair.split("=", 1)[0]
        if not _ENV_NAME.match(name):
            raise ValueError(f"{path}: {name!r} is not a shell variable name")
    return tuple(pairs)


#: Ways a profile could carry a robot's SSH password, all of them refused.
#:
#: This is not hypothetical about the wrong thing. ``launch_prefix`` is **published**: it
#: goes into ``get_capabilities``, so it reaches the dashboard, the browser, and any
#: screenshot of either; and the rendered command goes into ``run_started`` on the event
#: stream and into every operator's log. AGENTS.md's rule is absolute — robot SSH passwords
#: are never committed, logged, or put in an issue — and "scrubbing it afterwards does not
#: remove it from the event log" is exactly the situation here.
#:
#: So the credential class is removed rather than redacted. A redaction is a pattern that
#: has to keep matching; a refusal is a state the file cannot be in. The correct answer is
#: a key on the robot, and ``BatchMode=yes`` so ssh fails fast instead of sitting on a
#: prompt nobody can see.
_CREDENTIAL_SMELLS = ("sshpass", "password", "SSHPASS", "PASSWORD", "SSH_ASKPASS")


def check_no_credential(launch_prefix, env, source: str = "") -> None:
    """Refuse a profile that could put a robot's password on the mesh."""
    names = list(env) if isinstance(env, dict) else [str(item).split("=", 1)[0]
                                                     for item in (env or ())]
    for token in list(launch_prefix or []) + names:
        for smell in _CREDENTIAL_SMELLS:
            if smell in str(token):
                raise ValueError(
                    f"{source}: {token!r} looks like a credential. This profile's "
                    f"launch_prefix is PUBLISHED in get_capabilities and its rendered "
                    f"command goes out on the event stream, so a password here reaches the "
                    f"browser and every operator's log, and scrubbing it afterwards does "
                    f"not remove it from the log. Put an SSH key on the robot and keep "
                    f"-o BatchMode=yes.")


def check_extra_args(extra_args, source: str = "") -> None:
    """Refuse a profile that spells a flag this module owns.

    ``--live`` is the one that matters and the rest travel with it. argparse takes the LAST
    occurrence of an option, so a profile carrying ``--policy-mode raw`` and an RPC asking
    for ``supervised`` would resolve to whichever this module happened to append second —
    an arbitration decided by string order in a build function. Refusing is the only
    version of that anybody can reason about.
    """
    where = f"{source}: " if source else ""
    for item in extra_args:
        head = item.split("=", 1)[0]
        if head in RESERVED_FLAGS:
            raise ValueError(
                f"{where}extra_args may not contain {head!r}; this driver spells it itself "
                f"from the RPC's parameters. A profile that can set --live is a second "
                f"motion gate nobody would think to look for.")


def clamp_seconds(seconds) -> float:
    """Clamp a requested run length into ``(0, MAX_RUN_SECONDS]``.

    Clamped rather than refused, and only in this direction: a caller asking for longer than
    the ceiling gets the ceiling, and the reply says what it got. A caller asking for zero
    gets the smallest run that can happen, because a run of no seconds is a request nobody
    means literally.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = DEFAULT_RUN_SECONDS
    if value != value or value <= 0.0:            # NaN, zero, negative
        value = DEFAULT_RUN_SECONDS
    return min(value, MAX_RUN_SECONDS)


def check_choice(name: str, value: str, allowed) -> str:
    """One RPC parameter against its fixed set, or :class:`RunRefused`.

    This is the boundary between a browser and a robot's command line. The values are
    shell-quoted downstream and would not execute anything either way, but an unchecked
    string reaching ``mappo_drive``'s argparse fails several seconds and one SSH connection
    later, on the far end, as a usage message nobody sees.
    """
    if value not in allowed:
        raise RunRefused(f"{name}={value!r} is not one of {list(allowed)}")
    return value


def check_control_mode(platform: str, mode, simulated: bool = False):
    """Refuse a run the robot would accept and never act on. Returns a note, or raises.

    ⚠️ **This is the Go2 failure that has no symptom.** On 2026-08-26 the lab Go2 answered
    ``mode='mcf'``, and ``Go2Locomotion`` warned that ``Move`` commands may be ignored.
    ``mcf`` is not a sport mode: ``SportClient`` commands are accepted in ``normal`` and
    ``ai`` and nowhere else, so a run started in this state produces no fall, no fault and
    no error code — the robot agrees with every command and never steps. It is the same
    shape as the gait floor, which is the most expensive failure in this repository's
    history, and it is why that one is refused from a table rather than warned about.

    So the run is REFUSED, and the refusal names the mode it read.

    **It does not switch the mode for you, and that is the safety decision here.**
    ``ensure_sport_mode()`` reconfigures the motion controller and its own docstring says it
    "can make the robot shift/stand — only call it with a clear area + e-stop ready". A
    dashboard button that silently reconfigures a controller is a worse posture than the
    command line it replaces, which is the argument ``--allow-motion`` is already built on.
    The operator selects the mode; this refuses until they have.

    A **Lite3** has no readable equivalent — its posture and navigation mode are confirmed
    by the operator on the vendor interface, which is what ``--operator-ready`` asserts. So
    this returns a note rather than inventing a gate whose truth nothing here can check.
    """
    if simulated or platform == "sim":
        return (f"the bench double reports mode={mode!r}. No sport-mode gate applies: there "
                f"is no motion controller to be in the wrong mode.")
    if platform == "lite3":
        return ("the Lite3 exposes no motion-service mode to read. Standing and high-level "
                "navigation mode are the operator's confirmation on the vendor interface "
                "(--operator-ready), and nothing here can verify it.")
    if mode in SPORT_MODES:
        return None
    raise RunRefused(
        f"the Go2's motion service reports mode={mode!r}, which is not a sport mode. "
        f"SportClient.Move is accepted in {' or '.join(SPORT_MODES)} and nowhere else, so "
        f"this run would start, command velocities for its whole length, and the robot "
        f"would never step — no fall, no fault, no error code. Measured on the lab Go2 on "
        f"2026-08-26, twice, reading 'mcf'. Put the robot in a sport mode first "
        f"(Go2Locomotion.ensure_sport_mode('normal'), or the vendor remote). This driver "
        f"will not do it for you: switching modes reconfigures the controller and can make "
        f"the robot shift or stand, which is not something a web button gets to do.")


def servo_flags(profile: RunProfile, heading_servo: str) -> list:
    """How to spell the heading servo at THIS tree, or refuse.

    ⛔ Never an omission. On the deployed tree, leaving the flag out is not "the default" —
    it is issue #16's ``travel`` law, the one that saturated the yaw rate and put this robot
    into a cubicle panel or a cabinet on three runs out of four on 2026-08-17. #106 made the
    servo opt-in on ``main``; the tree that would actually run has not seen #106. So both
    spellings are explicit and the only question is which word this tree knows.

    A ``legacy`` tree is REFUSED a mode other than ``off`` rather than sent one it will
    reject: the rejection would arrive as an argparse usage message on the far end of an SSH
    connection, several seconds later, and read as "the run did not start" with no reason.
    """
    if profile.heading_servo_flag == "legacy":
        if heading_servo != "off":
            raise RunRefused(
                f"heading_servo={heading_servo!r} cannot be selected on this tree: it "
                f"predates #106, its --help lists --no-heading-servo and no --heading-servo, "
                f"and argparse would refuse the flag on the far end. 'off' is the only mode "
                f"it can be told to use — and it is the only one that has not driven a robot "
                f"into something. Update the tree at {profile.workdir} to get the others.")
        return ["--no-heading-servo"]
    return ["--heading-servo", heading_servo]


def build_run_argv(profile: RunProfile, *, seconds: float, policy_mode: str,
                   heading_servo: str, live: bool, allow_motion: bool,
                   run_id: str = "") -> list:
    """The exact command line for one run, or :class:`RunRefused`.

    **The motion gate, in the same words the worker uses.** ``drive_bridge.dispatch``
    refuses a motion command without ``--allow-motion``; this refuses ``--live`` without it.
    Both are checked again by their caller, and that duplication is the point: the driver's
    check stops the process being spawned at all, and this one means a build that skipped
    the driver still cannot produce a command line that moves a robot.

    **Every flag is spelled, none is inherited.** ``--policy-mode``, ``--heading-servo`` and
    ``--max-seconds`` go out on every run even when the value is the default, because the
    deployed tree is 67 commits behind ``main`` and its defaults are whatever they were that
    day. ``--heading-servo`` is the one that matters: it became opt-in in #106, the deployed
    tree still has the old behaviour, and issue #16's servo put this robot into a wall on
    three runs out of four. A run that omits the flag is a run whose control law is decided
    by the age of the checkout.

    ⚠️ That is also why ``--no-heading-servo`` is reserved. ``run-smoke.sh`` — the known-good
    wrapper on the robot — passes it, and it is a hidden alias for ``--heading-servo off``.
    Copying it into a profile alongside the ``--heading-servo`` this function writes would
    put two spellings of one setting on one command line, and argparse would settle it by
    position. A profile that names it is refused, and this writes the explicit spelling.
    """
    if live and not allow_motion:
        # REFUSED, not downgraded to a dry run. A silent downgrade is the worse failure:
        # the operator asked for motion, watched a run start, and is now watching a robot
        # that was never going to move — the same shape as `mode='mcf'`, which is refused
        # here for exactly this reason.
        raise RunRefused(
            "arm_motion was requested and this device was started WITHOUT --allow-motion, "
            "so it is status-and-checkpoints only. This is refused rather than quietly "
            "downgraded to a dry run: a run that starts and cannot move is indistinguishable "
            "from a run that starts and will not, and you would spend the difference "
            "diagnosing the robot. Restart the driver with --allow-motion, with a clear area "
            "and an operator on the abort — or call start_run with no arm_motion, which is "
            "the default and runs the scene check that cannot command a leg.")

    check_choice("policy_mode", policy_mode, POLICY_MODES)
    check_choice("heading_servo", heading_servo, HEADING_SERVOS)
    check_extra_args(profile.extra_args, source=profile.source_path)
    seconds = clamp_seconds(seconds)

    argv = [profile.python, "-u", profile.script, "--package", profile.package]
    argv += list(profile.extra_args)
    # After extra_args, so that reading the command line top to bottom shows the
    # deployment's constants and then this driver's decisions. They cannot collide —
    # check_extra_args refuses a profile that spells any of them — but the order is what a
    # person checks, and last-wins is the rule they will assume.
    argv += ["--policy-mode", policy_mode]
    argv += servo_flags(profile, heading_servo)
    argv += ["--max-seconds", _seconds_text(seconds)]
    for flag, path in output_paths(profile, run_id).items():
        argv += [flag, path]
    if live:
        argv.append("--live")
    return argv


def output_paths(profile: RunProfile, run_id: str) -> dict:
    """``{flag: path}`` for the run's own evidence, or nothing at all.

    Named after the run rather than fixed, so two runs do not overwrite each other's
    telemetry — which is what a fixed ``--telemetry`` in a profile would do, silently, and
    the file you would go looking for afterwards is the one that got overwritten.

    ``--record`` is opt-in per profile: the recorder's codec check fails the whole run on a
    host without one, and a demo that will not start is worse than a demo that is not filmed.
    """
    if not profile.output_dir or not run_id:
        return {}
    stem = f"{profile.output_dir.rstrip('/')}/{run_id}"
    paths = {"--telemetry": f"{stem}.jsonl"}
    if profile.record:
        paths["--record"] = f"{stem}.mp4"
    return paths


def _seconds_text(seconds: float) -> str:
    """A number the far end's argparse will read back as the same float."""
    return f"{seconds:g}"


def pidfile_for(profile: RunProfile, run_id: str) -> str:
    return f"{profile.pidfile_dir.rstrip('/')}/mappo-dashboard-run-{run_id}.pid"


def launch_command(profile: RunProfile, argv, pidfile: str = "") -> list:
    """The OS argv this process should actually spawn.

    Two shapes, and the difference is which machine the legs are on.

    **Local** (``launch_prefix`` empty) — the run is a child of this process. It is spawned
    directly, with no shell in between, so a SIGTERM from :meth:`stop` reaches the run
    itself rather than a ``/bin/sh`` that would exit and leave it.

    **Remote** — the run is on the robot and this process holds an SSH client. ``ssh`` takes
    ONE remote command string, so the whole thing is rendered as a shell line with every
    element ``shlex.quote``-d. The line records the remote PID before ``exec``-ing the run,
    which is what makes a stop possible at all: signalling the local ``ssh`` closes a socket
    and the process on the far end never hears about it.
    """
    argv = [str(a) for a in argv]
    if not profile.launch_prefix:
        return argv
    if not pidfile:
        raise RunRefused("a remote run needs a pidfile: signalling the local ssh client "
                         "closes a socket and does not stop the robot")
    parts = []
    if profile.env_setup:
        parts.append(". " + shlex.quote(profile.env_setup))
    # AFTER the source, because that is the order the known-good wrapper uses and because
    # these are meant to replace what the source left behind rather than be replaced by it.
    parts.extend("export " + shlex.quote(pair) for pair in profile.env)
    parts.append("cd " + shlex.quote(profile.workdir))
    # $$ is the remote shell's pid, and `exec` replaces that shell with the run, so the
    # recorded pid stays the run's for its whole life.
    parts.append("echo $$ > " + shlex.quote(pidfile))
    parts.append("exec " + " ".join(shlex.quote(a) for a in argv))
    # `&&`, not `;`: a failed `cd` must not run the next line from whatever directory the
    # login shell happened to leave us in.
    return [*profile.launch_prefix, " && ".join(parts)]


def local_env(profile: RunProfile, base=None):
    """The environment for a LOCAL run, or ``None`` to inherit this process's.

    A remote run gets its variables from ``export`` lines inside the shell command; a local
    one has no shell, so they are handed to the spawn instead. Same source, two renderings,
    because there is no shell in the local path on purpose — see :func:`launch_command`.
    """
    if not profile.env:
        return None
    merged = dict(os.environ if base is None else base)
    for pair in profile.env:
        name, _, value = pair.partition("=")
        merged[name] = value
    return merged


def stop_command(profile: RunProfile, pidfile: str):
    """The OS argv that stops a REMOTE run, or ``None`` when there is nothing to send.

    ⛔ **SIGTERM, and there is no SIGKILL path in this module at all.** ``SAFETY.md`` §0 is
    the cardinal rule: a hard kill is the opposite of a stop, because the last command stays
    latched on a motor bus with nothing left to update or damp it. ``mappo_drive`` inherits
    ``visual_nav``'s teardown and the shared ``SafeStop``, both of which damp on SIGTERM and
    neither of which can run on SIGKILL. ``test_run_control`` asserts that no command this
    module builds contains ``-9`` or ``KILL``.

    ``|| true`` because a run that has already exited is the normal case for a stop pressed
    a moment late, and a non-zero exit there would be reported to the operator as a stop
    that failed.
    """
    if not profile.launch_prefix:
        return None                              # a local child is signalled, not scripted
    quoted = shlex.quote(pidfile)
    return [*profile.launch_prefix,
            f"if [ -s {quoted} ]; then kill -TERM \"$(cat {quoted})\" 2>/dev/null || true; fi"]


def describe(profile: RunProfile, allow_motion: bool = False) -> dict:
    """What ``get_capabilities`` advertises about run control on this robot.

    Two previews, because there are two commands and the page should be able to show either
    one BEFORE the press rather than in the reply afterwards.

    * ``command_preview`` is what ``start_run()`` with no arguments runs: the scene check,
      no ``--live``, available on every driver.
    * ``armed_command_preview`` is what ``start_run(arm_motion=true)`` runs, and it is
      ``None`` on a driver without ``--allow-motion`` — because on that driver there is no
      such command. Showing one would be the one line on the page telling the operator a
      gate is not there.
    """
    preview = build_run_argv(profile, seconds=DEFAULT_RUN_SECONDS, policy_mode="supervised",
                             heading_servo="off", live=False, allow_motion=True,
                             run_id="PREVIEW")
    armed = None
    if allow_motion:
        armed = build_run_argv(profile, seconds=DEFAULT_RUN_SECONDS,
                               policy_mode="supervised", heading_servo="off", live=True,
                               allow_motion=True, run_id="PREVIEW")
    return {
        "supported": True,
        "label": profile.label,
        "remote": bool(profile.launch_prefix),
        "launch_prefix": list(profile.launch_prefix),
        "workdir": profile.workdir,
        "script": profile.script,
        "python": profile.python,
        "package": profile.package,
        "output_dir": profile.output_dir or None,
        "records_video": bool(profile.record and profile.output_dir),
        "max_seconds": MAX_RUN_SECONDS,
        "default_seconds": DEFAULT_RUN_SECONDS,
        "policy_modes": list(POLICY_MODES),
        # What this TREE can be told, not what the flag supports. A legacy tree can only be
        # told 'off', and a page that offered the other two would be offering a refusal.
        "heading_servos": (["off"] if profile.heading_servo_flag == "legacy"
                           else list(HEADING_SERVOS)),
        "heading_servo_flag": profile.heading_servo_flag,
        "command_preview": preview,
        "armed_command_preview": armed,
        "motion_enabled": bool(allow_motion),
        "arms_with": "arm_motion",
        "named_commit": None,
        "tree_note": profile.tree_note or TREE_NOTE,
        "profile_path": profile.source_path or None,
    }


def unsupported(reason: str = "") -> dict:
    """The same shape as :func:`describe`, for a driver started with no run profile.

    A driver without a profile has to say so in the SAME field the page reads when there is
    one, or the page's only way to tell "this robot cannot start a run" from "this robot's
    capability payload is from an older driver" is the absence of a key — which is also what
    a transport that dropped it looks like.
    """
    return {
        "supported": False,
        "reason": reason or ("this driver was started without --run-profile, so it has no "
                             "machine, interpreter or working directory to start a run in"),
        "max_seconds": MAX_RUN_SECONDS,
        "policy_modes": list(POLICY_MODES),
        "heading_servos": list(HEADING_SERVOS),
        "named_commit": None,
        "armed_command_preview": None,
        "motion_enabled": False,
        "arms_with": "arm_motion",
        "tree_note": TREE_NOTE,
    }


@dataclass
class RunRecord:
    """One started run, in the shape ``get_status`` and the events report it.

    Deliberately holds the **rendered command** as well as the argv. The argv is what a
    person reads; the rendered command is what was actually handed to the operating system,
    including the SSH prefix and the shell line, and on a remote run those are not the same
    thing. Reporting only the argv would describe a local run that never happened.
    """

    run_id: str
    argv: list
    command: list
    live: bool
    policy_mode: str
    heading_servo: str
    seconds: float
    remote: bool
    pidfile: str = ""
    outputs: dict = field(default_factory=dict)
    started_at: float = 0.0
    pid = None
    #: Filled in when the run ends, so a finished run can still be reported.
    finished_reason: str = ""
    exit_code = None

    def snapshot(self, now: float, tail=()) -> dict:
        """What the page shows. ``elapsed_s`` is a DURATION, never a timestamp.

        Same rule as ``peer_pose``'s ``sample_age_s`` and for the same reason: the Go2 has
        no working RTC and its clock was measured 56 years behind on 2026-08-26, so a run's
        wall-clock start is not a fact two machines can share. A duration measured between
        two readings of one clock is.
        """
        return {
            "run_id": self.run_id,
            "live": self.live,
            "policy_mode": self.policy_mode,
            "heading_servo": self.heading_servo,
            "seconds": self.seconds,
            "elapsed_s": round(max(0.0, now - self.started_at), 2),
            "remote": self.remote,
            "pid": self.pid,
            "argv": list(self.argv),
            "command": list(self.command),
            "outputs": dict(self.outputs),
            "named_commit": None,
            "tail": list(tail),
        }


#: Runs started by THIS driver process, so two inside one second cannot collide.
_run_counter = itertools.count()


def new_run_id(clock=None) -> str:
    """A run id that is safe in a path, readable in a log, and unique.

    Stamped on the DRIVER's clock, which is the workstation's, and used only as a label and
    a filename — never subtracted from anything on the robot. The Go2's own clock was
    measured 56 years out on 2026-08-26, so a run id taken there would sort a demo's runs
    into 1970.

    **A counter and not only randomness.** The id names a telemetry file, and two runs
    sharing one is the second silently overwriting the first — which you find out about when
    you go looking for the first. Random suffixes make that *unlikely*: two bytes collide
    once in about four hundred tries at 200 runs, which is a flaky test and, eventually, a
    lost run. The counter makes it impossible within a process; the random half is there so
    that two drivers writing to one directory still do not collide.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock() if clock else time.time()))
    return f"{stamp}-{next(_run_counter) % 0x1000:03x}{os.urandom(2).hex()}"
