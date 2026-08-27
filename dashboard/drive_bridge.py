#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SDK-env worker: run ONE motion command, print a JSON result, exit.

⛔ **``robot-stack/SAFETY.md`` governs this file.** It commands legs.

## Why a subprocess at all

``device-connect-edge`` requires Python >= 3.11. The Go2's Jetson runs the Unitree SDK —
``unitree_sdk2py`` over CycloneDDS — on Python 3.8, and the Lite3's perception host runs ROS 2.
Neither environment can hold both. So the driver stays in a 3.11 env with no robot
dependencies at all, and reaches the robot by running this file in the SDK env: one command,
one JSON line on stdout, exit. That is the same two-env bridge the upstream Go2 Device
Connect driver uses, and this reuses its shape rather than inventing a second one.

The bridge is also a safety property, not just a packaging workaround. Every command runs in
a process that exits, so a driver that hangs, crashes or is killed cannot leave a velocity
latched — the worker's own ``finally`` stops the robot, and if the worker dies instead the
process is gone and the next command starts from a clean client.

**``pose-stream`` is the one command that does not exit, and it is allowed to because the
property above is about a LATCHED VELOCITY.** It reads the estimator and writes JSON lines;
it never calls ``set_velocity``, so there is nothing for its lifetime to leave latched. It
exists because a peer robot has to publish its pose at 10 Hz for another robot to avoid it,
and one subprocess per sample is not an implementation of 10 Hz — ``robot_driver`` budgets
``MAX_SECONDS + 25`` seconds for one ``status``, nearly all of it SDK import and DDS
discovery. See :func:`command_pose_stream`.

## Every button is a measurement

Commands are **duration-bounded and open-loop**, and the result reports what the robot
actually did: pose before, pose after, distance travelled, and the fraction of the commanded
speed that was delivered. Closed-loop distance control was the alternative and it was
rejected deliberately — it would hide exactly the number this repository has spent the most
time on. This Go2 delivers roughly 0.45 of what it is commanded when derated and 0.70 at
full forward command, and about 0.27 laterally (issue #42); **no Lite3 has produced any of
these numbers, or any number at all.** A "walk 1 m" button that quietly ran longer to get
there would have made the gait floor take another five runs to find. A button that says
"commanded 0.35 for 1.5 s, travelled 0.012 m" diagnoses itself.

## What refuses to run

* **Motion without ``--allow-motion``.** Status is always available; nothing moves unless
  the driver was started with motion enabled AND passes the flag through.
* **A speed below the platform's measured gait floor**, unless ``--force``. Below it a Go2
  stands still while every instrument insists it is fine (``avoidance.MIN_GAIT_COMMAND_M_S``
  and the table in ``deploy/README.md``). This is the single most expensive failure in the
  repository's history and the worker refuses to reproduce it silently.
* **Any motion on a platform whose gait has never been measured**, unless ``--force``. That
  is the Lite3 today: issue #13's measurements are all still open, and a borrowed floor is
  worse than no floor because it arrives with an air of authority. See
  :data:`GAIT_FLOORS`.
* **Reverse beyond a short bounded nudge.** The planner never samples reverse because the
  robot has no rear sensing, and issue #40 caught the policy commanding it. A back button
  exists because it was asked for; it is capped at :data:`MAX_REVERSE_SECONDS` and it says
  in its own result that nothing is watching behind the robot.

## Which interface a Lite3 is commanded through

``--locomotion-transport`` selects it, with the same three names and the same ``udp``
default the Lite3 navigator offers
(``robot-stack/deep_robotics/lite3/visual_nav/robot_bindings.py``) and the commissioning
harness offers (``.../lite3/commissioning/robot_link.py``). There was no such flag here
until issue #141: the loader built ``Lite3Locomotion`` with its ROS 2 default, so on a
Venture with no ROS 2 runtime **every** command — ``stop`` included — died at
``ModuleNotFoundError`` before it reached the robot, and the ``axis`` transport both
Ventures have actually walked on was unreachable from the dashboard entirely.

⚠️ **``axis`` throws the commanded magnitude away.** Past the profile's linear deadband
every command emits the same full-scale primitive, so a speed there is a DIRECTION and the
``delivered_fraction`` this worker computes is not a delivery ratio. That is argued in
``lite3_axis_locomotion.py`` and priced in ``robot_link.TRANSPORTS``; this file states it
in the result rather than restating the argument. ``udp`` carries the number to the wire
and has never been seen to move either Venture; ``axis`` has moved both.

No default port, host or topic name is spelled here. ``None`` means "whatever the transport
module itself defines", because the one thing this file must not do is carry a second copy
of a vendor constant — the same reason :data:`GAIT_FLOORS` does not carry a borrowed floor.

## What ``stop`` can promise, and what it cannot

``stop`` is the reason issue #141 was filed, and the promise is per-transport rather than
universal:

* **Go2, Lite3 ``udp``, and the bench double** — a zero velocity is commanded from THIS
  process and it is authoritative: the last velocity persists on the far side until the
  next one, so a zero from any process ends the walk. ``commanded_zero`` is ``true``.
* **Lite3 ``axis``** — it is not, and saying otherwise would be the worst kind of green
  tick. The axis stream is a thread inside the process that started it
  (:class:`AxisStreamSender`); a process that never started one has no setpoint to zero
  and ``stop()`` on it sends nothing at all. What ends an axis walk is ENDING THAT
  PROCESS, which ``robot_driver.stop`` does — SIGTERM — before it calls here. So this
  worker reports ``commanded_zero: false`` and says which lever actually moved. ⚠️ What
  happens on the robot once the datagrams stop is the vendor watchdog's business: the
  command TTL is capped below 250 ms *because* that is the documented watchdog, and
  **nobody has measured that watchdog on either Venture**.
* **A transport that will not load** — nothing reached the robot, and the result says so
  as :data:`CAUSE_TRANSPORT_UNAVAILABLE` rather than as a generic failure.

⚠️ **One limitation is not fixed here and is not hidden either.** On ``udp``,
``Lite3UdpLocomotion.stop()`` documents itself as "Never refuses: a stop must survive a
lost link" — but ``connect()`` waits 5 s for a state frame and raises ``Lite3LinkLost``
first, so a stop cannot be commanded to a robot whose telemetry is not arriving even
though the command socket would work. Measured 2026-08-27 with no robot present: 5.0 s,
then a refusal. Fixing it means a ``connect()`` that can open the command socket without
the state stream, which is a change in ``robot-stack/deep_robotics/lite3/locomotion/`` and
not in the dashboard. What this file does is make sure that stop still reports
``commanded_zero: false`` and says the zero was not sent, rather than a bare traceback.

## Three ways to fail, not two

``refused: true`` (a rule in this stack turned the command down) and ``refused: false``
(anything else) put "the operator asked for something we will not do" and "the Python on
this machine cannot reach the robot" in the same bucket, and issue #141 is what that costs:
a missing module renders as the robot misbehaving, and the day goes on the robot.
:data:`CAUSE_REFUSED` / :data:`CAUSE_TRANSPORT_UNAVAILABLE` / :data:`CAUSE_FAULT` are the
three, carried in ``cause``; ``refused`` stays beside them and keeps its old meaning.

Python 3.8, stdlib only — it has to import in the robot's SDK env.
``python3 test_drive_bridge.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

#: Commands that move the robot. Everything not in here is read-only and always permitted.
MOTION_COMMANDS = frozenset({"stand", "stand-down", "walk", "strafe", "turn"})

#: Refresh rate for a persistent velocity command. ``SportClient.Move`` has no dead-man
#: timeout — the last velocity persists until the next command — so a bounded nudge is a
#: loop that re-sends, not a single call and a sleep.
COMMAND_HZ = 10.0

#: Hard ceiling on any single nudge. A web button should never be able to start a walk that
#: outlives the operator's attention; anything longer is a job for ``mappo_drive.py``, which
#: has perception, a planner and a stall detector.
MAX_SECONDS = 5.0

#: Reverse is capped harder than forward, because nothing on either robot looks backwards.
MAX_REVERSE_SECONDS = 2.0

#: Default rate for ``pose-stream``. The control loop that consumes it runs at 10 Hz, and
#: a peer pose arriving slower than the loop that reads it is a loop reading the same
#: sample twice. Capped at :data:`MAX_POSE_STREAM_HZ` because the sample comes off DDS and
#: the peer's own control stack is on that bus too.
POSE_STREAM_HZ = 10.0
MAX_POSE_STREAM_HZ = 20.0

#: Measured gait floors. These are per-platform and per-axis, and the two axes genuinely
#: differ — see issue #42, which is about a calibration interface that assumed they did not.
#:
#: ⚠️ ``None`` means NOT MEASURED. It is not a default and it is not zero; it is the absence
#: of a number. The one thing this table must never do is hand one robot another robot's
#: measurement, because a floor is a property of a specific machine on a specific floor —
#: issue #13 opens by saying "do not copy values between units".
#:
#: **Go2 — both entries come from the same evening.** ``forward`` 0.35 is
#: ``avoidance.MIN_GAIT_COMMAND_M_S``, five runs at 0.21 m/s (no gait) against one at 0.35
#: (2.07 m in 9 s). ``lateral`` 0.20 is issue #42's table, measured 2026-08-19 with a forward
#: step as a same-session control: vy 0.15 travelled 0.010 m in 1.5 s (no gait), vy 0.20
#: travelled 0.087 / 0.076 / 0.080 m — walked, 3 of 3. Both are LOWEST-OBSERVED-TO-WORK
#: values rather than located thresholds: the true lateral floor lies in (0.15, 0.20], and
#: the forward one is contested by a run that sustained 0.295 m/s (issue #26). Erring high
#: refuses a little more than strictly necessary, which is the safe direction. ``yaw`` has
#: never been measured on either robot.
#:
#: **Lite3 — nothing has been measured, on any axis.** Neither event robot has moved under
#: this stack at all: ``robot-stack/deep_robotics/lite3/README.md`` records "gait floor |
#: required as ``--gait-floor`` | not measured", and issue #13 still carries "Lowest forward
#: command that sustains a gait" as an open box. This row read
#: ``{"forward": 0.35, "lateral": 0.20}`` until it was checked — the Go2's pair verbatim,
#: presented as a Lite3 measurement. See :func:`check_gait_floor` for what an all-unmeasured
#: platform now does with a motion command.
GAIT_FLOORS = {
    "go2":   {"forward": 0.35, "lateral": 0.20,  "yaw": None},
    "lite3": {"forward": None, "lateral": None, "yaw": None},
    "sim":   {"forward": 0.0,  "lateral": 0.0,  "yaw": 0.0},
}

#: Default speeds for a dashboard nudge, chosen to be at or above the Go2's floors — the only
#: floors anybody has. ``DEFAULT_VY`` 0.20 is both the Go2's measured lateral floor and the
#: envelope's lateral cap (``avoidance.Limits.max_vy``); those coincide, and that coincidence
#: is load-bearing in ``integration/mappo_drive.py``. On a Lite3 these defaults are refused,
#: which is the point: there is no Lite3 speed anybody can say is safe to press.
DEFAULT_VX = 0.35
DEFAULT_VY = 0.20
DEFAULT_WZ = 0.70

#: The Lite3 vendor interfaces this worker can command through, in the vocabulary the rest
#: of the Lite3 tree already uses. Not a second naming: ``robot_bindings._add_ros_arguments``
#: offers exactly ``{udp, axis, ros2}`` with ``udp`` as the default, and
#: ``commissioning/robot_link.TRANSPORTS`` offers ``{udp, axis}`` with the same default.
#:
#: The two facts beside each name are the ones a dashboard has to act on, and they are the
#: two that ``robot_link.Transport`` records for the same reason: ``preserves_magnitude``
#: decides whether a commanded speed means anything on the wire, and ``walked`` is whether
#: any Venture has been seen to move on this interface at all. They are NOT imported from
#: that table — this file is stdlib-only and has to import with no ``robot-stack`` beside
#: it, which is what ``--platform sim`` on a laptop is — so the WORDING here is short and
#: points at the table rather than paraphrasing its evidence.
LITE3_TRANSPORTS = {
    "udp":  {"preserves_magnitude": True,  "walked": False,
             "summary": "legacy complex-velocity UDP. The commanded number reaches the "
                        "wire; no Venture has been seen to MOVE on it (robot_link.TRANSPORTS)"},
    "axis": {"preserves_magnitude": False, "walked": True,
             "summary": "profile-gated moving-mode simple axes. Both Ventures have walked "
                        "on it, and its mapping is SIGN-ONLY: a commanded speed past the "
                        "profile's deadband is a direction, not a speed"},
    "ros2": {"preserves_magnitude": True,  "walked": False,
             "summary": "the Lite3_ROS bridge topics. Needs a ROS 2 runtime on the "
                        "perception host, which these two Ventures may not have"},
}

#: The default, and it is the siblings' default rather than this file's opinion. It was
#: effectively ``ros2`` before issue #141 — not chosen, just what ``Lite3Locomotion``
#: constructs with no factory — and that is the configuration in which ``stop`` died at an
#: import error on a robot with no ROS 2.
DEFAULT_LITE3_TRANSPORT = "udp"

#: ``cause``: three states, because "we will not" and "we cannot" and "it went wrong" are
#: three different messages to an operator and only one of them is about the robot.
CAUSE_REFUSED = "refused"
CAUSE_TRANSPORT_UNAVAILABLE = "transport_unavailable"
CAUSE_FAULT = "fault"


class BridgeError(RuntimeError):
    """A refusal, reported as JSON rather than as a traceback."""


class TransportUnavailable(RuntimeError):
    """The locomotion transport could not be LOADED, so nothing was asked of the robot.

    Separate from :class:`BridgeError` because the two are opposite diagnoses that used to
    arrive looking identical. A ``BridgeError`` means this stack turned the command down
    and the robot is fine. This means the Python environment the worker runs in cannot
    reach the robot at all — a missing ``ros2_twist_locomotion``, an absent
    ``unitree_sdk2py``, a ``robot-stack`` that was not deployed beside this file. Nothing
    was commanded, nothing was refused, and the robot has not been consulted.

    ⚠️ It is raised for an IMPORT failure only. A transport that imports and then cannot
    reach the machine — ``Lite3LinkLost``, a socket error, a dead DDS — is a real fault
    about the real link and is reported as one; calling that "unavailable" would put a
    silent robot and a missing package in the same bucket again.
    """


# ── platform backends ─────────────────────────────────────────────────────────────────────
def _stack_dir():
    """The deployed ``robot-stack`` directory.

    ``MAPPO_STACK_DIR`` wins; otherwise the path relative to this file in a checkout. Same
    resolution order the upstream Go2 worker uses, so a robot with both deployed does not
    need two different environment variables.
    """
    if os.environ.get("MAPPO_STACK_DIR"):
        return os.environ["MAPPO_STACK_DIR"]
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot-stack"))


def _load_go2(iface):
    sys.path.insert(0, os.path.join(_stack_dir(), "unitree", "go2", "locomotion"))
    # connect() is inside the try, not only the import. ⚠️ THIS IS WHERE THE FIRST FIX FOR
    # issue #141 WAS WRONG, and running the documented command is what caught it: the
    # binding a locomotion class composes is imported LAZILY, inside connect(), so
    # wrapping the module import alone classified a missing vendor SDK as a robot fault
    # exactly as before. Only ImportError is caught -- a transport that imports and then
    # cannot reach the machine is a real fault about the real link.
    try:
        from go2_locomotion import Go2Locomotion
        loco = Go2Locomotion(iface=iface)
        loco.connect()
    except ImportError as exc:
        raise TransportUnavailable(
            f"the Go2 locomotion binding could not be imported from {_stack_dir()} "
            f"({exc}). Nothing was commanded and the robot was not consulted. This is the "
            f"interpreter, not the robot: --bridge-python must be the SDK env's python, "
            f"the one that can import unitree_sdk2py, and robot-stack/ must be beside this "
            f"file or named by MAPPO_STACK_DIR.") from exc
    return loco


class Lite3Link:
    """Which Lite3 vendor interface to command through, and what that interface needs.

    Every field except ``transport`` defaults to ``None``, meaning "whatever the transport
    module itself defines". That is deliberate and it is the same rule :data:`GAIT_FLOORS`
    follows: a port number or a motion-host address copied into this file is a second copy
    of a vendor constant that nothing keeps in step, and the Lite3 row of ``GAIT_FLOORS``
    is what a copied constant looks like once it has drifted.
    """

    def __init__(self, transport=DEFAULT_LITE3_TRANSPORT, motion_host=None,
                 command_port=None, state_port=None, state_bind=None, axis_profile=None,
                 axis_local_port=None, cmd_vel_topic=None, odom_topic=None,
                 chosen=False):
        self.transport = transport
        self.motion_host = motion_host
        self.command_port = command_port
        self.state_port = state_port
        self.state_bind = state_bind
        self.axis_profile = axis_profile
        self.axis_local_port = axis_local_port
        self.cmd_vel_topic = cmd_vel_topic
        self.odom_topic = odom_topic
        #: Whether ``--locomotion-transport`` was actually typed. A flag that was not given
        #: cannot conflict with anything, and a flag that was must not be silently ignored
        #: on a platform that has no such interface — see :func:`check_lite3_link`.
        self.chosen = chosen

    def describe(self):
        """The transport row, for a result that has to say what it was driving through."""
        row = LITE3_TRANSPORTS[self.transport]
        return {"locomotion_transport": self.transport,
                "transport_preserves_magnitude": row["preserves_magnitude"],
                "transport_has_walked_a_venture": row["walked"],
                "transport_note": row["summary"]}

    def link_kwargs(self):
        """The UDP link keywords, omitting every one that was not given.

        Omitted rather than defaulted, so the transport module's own default is what
        applies and this file never has to know what it is.
        """
        pairs = (("motion_host", self.motion_host), ("command_port", self.command_port),
                 ("state_port", self.state_port), ("bind", self.state_bind))
        return {name: value for name, value in pairs if value is not None}


def _lite3_factory(link):
    """The ``implementation_factory`` for the selected transport, or ``None`` for ros2.

    ``None`` is the ros2 case and not an error: ``Lite3Locomotion``'s own default factory
    is the ROS 2 Twist binding, so leaving it alone is how that transport is selected.
    Same shape as ``robot_bindings.create_locomotion`` and ``robot_link._transport_factory``
    — the choice is contained here and ``Lite3Locomotion`` never learns what it composed.
    """
    if link.transport == "ros2":
        return None
    if link.transport == "udp":
        from deep_robotics.lite3.locomotion.lite3_udp_locomotion import udp_locomotion_factory
        return udp_locomotion_factory(**link.link_kwargs())
    from deep_robotics.lite3.locomotion.lite3_axis_locomotion import (
        AxisProfile,
        AxisProfileError,
        axis_locomotion_factory,
    )
    profile = None
    if link.axis_profile is not None:
        try:
            profile = AxisProfile.load(Path(link.axis_profile))
        except AxisProfileError as exc:
            # A profile this worker cannot read is a refusal, not a transport gap: the file
            # was named and it is wrong, which is something the operator can fix.
            raise BridgeError(f"cannot use axis profile {link.axis_profile}: {exc}") from None
    kwargs = link.link_kwargs()
    if link.axis_local_port is not None:
        kwargs["axis_local_port"] = link.axis_local_port
    # profile=None reaches here only on `stop` and `status`, which check_lite3_link permits
    # precisely because neither needs one -- Lite3AxisLocomotion raises AxisProfileError on
    # the first set_velocity and never on stop().
    return axis_locomotion_factory(axis_profile=profile, **kwargs)


def _load_lite3(operator_ready, link=None):
    link = Lite3Link() if link is None else link
    # The tree root, not the locomotion directory: lite3_udp_locomotion imports
    # `deep_robotics.lite3.commissioning.lite3_state_probe`, and lite3_axis_locomotion
    # imports its sibling by the same absolute path. That is the layout every other Lite3
    # entry point uses (robot_bindings.py, robot_link.py), so this uses it too rather than
    # putting the same file on sys.path twice under two names.
    stack = _stack_dir()
    if stack not in sys.path:
        sys.path.insert(0, stack)
    try:
        from deep_robotics.lite3.locomotion.lite3_locomotion import Lite3Locomotion
        factory = _lite3_factory(link)
        loco = _connect_lite3(Lite3Locomotion, factory, link, operator_ready)
    except ImportError as exc:
        raise TransportUnavailable(
            f"--locomotion-transport {link.transport} could not be loaded from {stack} "
            f"({exc}). Nothing was commanded and the robot was not consulted. "
            + ("The ROS 2 Twist binding needs a ROS 2 runtime and the shared robotkit on "
               "the interpreter that runs this worker; a Venture with no provisioned "
               "perception host has neither. --locomotion-transport udp and axis are "
               "stdlib sockets and need neither."
               if link.transport == "ros2" else
               "This transport is stdlib sockets, so a failure here is robot-stack/ not "
               "being beside this file: deploy it, or name it with MAPPO_STACK_DIR.")
        ) from exc
    return loco


def _connect_lite3(locomotion_class, factory, link, operator_ready):
    """Compose and connect. Split out only so the ``ImportError`` above can cover it.

    ⚠️ ``connect()`` HAS TO BE INSIDE THAT ``try``, and the first fix for issue #141 did not
    put it there. ``lite3_locomotion._ros2_locomotion`` imports the ROS 2 Twist binding
    LAZILY, and that module's own docstring says why: it is "composed rather than
    subclassed so this module remains importable on a workstation without ROS 2 … both are
    loaded only by :meth:`connect`". So ``import lite3_locomotion`` succeeds on a robot
    with no ROS 2 and the ``ModuleNotFoundError`` arrives one call later — a wrapper around
    the import alone classified it as a robot fault, exactly as before the fix. Running
    ``drive_bridge.py status --platform lite3 --locomotion-transport ros2`` is what caught
    that; reading the wrapper did not.
    """
    arguments = {"operator_ready": operator_ready}
    if factory is not None:
        arguments["implementation_factory"] = factory
    # Passed only when given, for the same reason the link ports are: /cmd_vel and
    # /leg_odom2 are Lite3Locomotion's own defaults and this file does not restate them.
    if link.cmd_vel_topic is not None:
        arguments["cmd_vel_topic"] = link.cmd_vel_topic
    if link.odom_topic is not None:
        arguments["odom_topic"] = link.odom_topic
    loco = locomotion_class(**arguments)
    loco.connect()
    return loco


class SimLocomotion:
    """A bench double that integrates the commanded velocity into a pose.

    NOT a physics simulation and it does not pretend to be one: it delivers exactly what it
    is commanded, so it cannot tell you anything about gait floors, delivery ratios or
    whether a real robot would walk. What it is for is exercising every path above it — the
    Device Connect mesh, the RPC schemas, the event stream, the refusals and the dashboard —
    without a robot in the room.

    It implements the same surface the two real bindings do, and
    ``test_drive_bridge.py`` asserts that by introspection against ``Go2Locomotion``, so a
    change to the real interface fails here rather than at the first live run.
    """

    def __init__(self):
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = (0.0, 0.0, 0.0)
        self._last = None
        self._standing = False

    def connect(self):
        self._last = time.monotonic()

    def _integrate(self):
        now = time.monotonic()
        if self._last is None:
            self._last = now
            return
        dt = now - self._last
        self._last = now
        vx, vy, wz = self._v
        # Body-frame velocity into an odom-frame pose, using the heading at the start of the
        # step. First-order and that is enough: the point is a pose that moves the way the
        # command says, not an integrator worth defending.
        self._x += (vx * math.cos(self._yaw) - vy * math.sin(self._yaw)) * dt
        self._y += (vx * math.sin(self._yaw) + vy * math.cos(self._yaw)) * dt
        self._yaw = _wrap(self._yaw + wz * dt)

    def set_velocity(self, vx, vy, vyaw):
        self._integrate()
        self._v = (vx, vy, vyaw)

    def stop(self):
        self._integrate()
        self._v = (0.0, 0.0, 0.0)

    def pose(self):
        self._integrate()
        return _Pose(self._x, self._y, self._yaw)

    def velocity(self):
        self._integrate()
        return self._v

    def stand(self):
        self._standing = True

    def stand_down(self):
        self.stop()
        self._standing = False

    def current_mode(self):
        return "sim"

    def shutdown(self):
        self.stop()


class _Pose:
    """The three fields this worker reads off a pose, for the sim backend."""

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _require_virtualenv(component):
    """Refuse a system-Python run before the vendor SDK import that would fail obscurely.

    Placed here rather than at module import so ``--platform sim`` never needs
    ``robot-stack`` on disk, and so the guard covers ``status`` too: ``status`` opens DDS
    and imports ``unitree_sdk2py``, so from the system Python it produces exactly the
    ``ModuleNotFoundError`` that an agent then reaches for ``pip install`` to silence.

    The refusal is raised as :class:`BridgeError` so it reaches the dashboard as
    ``refused: true`` with the whole message in ``error`` -- a bare ``SystemExit`` would
    print to stderr and exit with no JSON, and ``robot_driver`` would surface only the
    last 400 characters of it. It is echoed to stderr as well, for a person running this
    worker by hand who wants it laid out rather than JSON-escaped.
    """
    preflight_dir = os.path.join(_stack_dir(), "preflight")
    if preflight_dir not in sys.path:
        sys.path.insert(0, preflight_dir)
    try:
        from venv_guard import evaluate
    except ImportError as exc:
        raise BridgeError(
            f"cannot import the virtualenv guard from {_stack_dir()}/preflight "
            f"({exc}). This worker needs robot-stack/ beside it -- the same tree "
            "_load_go2 and _load_lite3 import their locomotion from. Set "
            "MAPPO_STACK_DIR if it is deployed elsewhere.") from exc
    decision = evaluate(component, reaching_hardware=True)
    if decision.refuse:
        print(decision.message, file=sys.stderr)
        raise BridgeError(decision.message)
    return decision


def load_platform(platform, iface="eth0", operator_ready=False, link=None):
    if platform == "go2":
        _require_virtualenv("drive_bridge --platform go2")
        return _load_go2(iface)
    if platform == "lite3":
        _require_virtualenv("drive_bridge --platform lite3")
        return _load_lite3(operator_ready, link)
    if platform == "sim":
        loco = SimLocomotion()
        loco.connect()
        return loco
    raise BridgeError(f"unknown platform {platform!r}; use go2, lite3 or sim")


def link_from_args(args):
    """The :class:`Lite3Link` ``--locomotion-transport`` and its companions describe."""
    chosen = getattr(args, "locomotion_transport", None)
    return Lite3Link(
        transport=chosen or DEFAULT_LITE3_TRANSPORT,
        motion_host=getattr(args, "motion_host", None),
        command_port=getattr(args, "command_port", None),
        state_port=getattr(args, "state_port", None),
        state_bind=getattr(args, "state_bind", None),
        axis_profile=getattr(args, "axis_profile", None),
        axis_local_port=getattr(args, "axis_local_port", None),
        cmd_vel_topic=getattr(args, "cmd_vel_topic", None),
        odom_topic=getattr(args, "odom_topic", None),
        chosen=chosen is not None,
    )


def check_lite3_link(backend, link, command):
    """Refuse a transport/option combination that cannot do what was asked. Pure.

    Called before anything connects, for the reason ``check_gait_floor`` is: these are
    decidable from the arguments, and opening a socket to deliver a command that was never
    going to run costs seconds on the robot's own host.

    Every rule here is the same shape — **an option that would be silently ignored is
    refused instead**. ``robot_link.load_axis_profile`` puts it best about the one that
    caught it first: passing a profile to a transport that ignores it "looks exactly like a
    profile that took effect".

    ⚠️ ``axis`` WITHOUT a profile is refused for motion and permitted for ``stop`` and
    ``status``, and that asymmetry is load-bearing rather than lenient.
    ``Lite3AxisLocomotion.set_velocity`` raises ``AxisProfileError`` without one, so a
    nudge is refused here instead of deep inside the transport — but ``stop()`` on that
    class only zeroes a setpoint and reads no primitive at all, and a stop that demanded a
    profile would be a stop with a prerequisite.
    """
    if link.chosen and backend != "lite3":
        raise BridgeError(
            f"--locomotion-transport {link.transport} selects a Lite3 vendor interface and "
            f"this worker is driving the {backend}. It would have been ignored, which "
            f"looks exactly like a transport that took effect. Drop the flag, or point "
            f"--backend/--platform at the lite3.")
    if backend != "lite3":
        return
    if link.transport not in LITE3_TRANSPORTS:
        raise BridgeError(f"unknown --locomotion-transport {link.transport!r}; expected one "
                          f"of " + ", ".join(sorted(LITE3_TRANSPORTS)))
    if link.axis_profile is not None and link.transport != "axis":
        raise BridgeError(
            f"--axis-profile was given but --locomotion-transport is {link.transport!r}, "
            f"which does not read one. Passing a profile to a transport that ignores it "
            f"looks exactly like a profile that took effect.")
    if link.transport != "ros2" and (link.cmd_vel_topic is not None
                                     or link.odom_topic is not None):
        raise BridgeError(
            f"--cmd-vel-topic/--odom-topic name ROS 2 topics and "
            f"--locomotion-transport {link.transport} has no ROS 2 node to publish them "
            f"on. They would be accepted and discarded.")
    if link.transport == "axis" and link.axis_profile is None and command in MOTION_COMMANDS:
        raise BridgeError(
            "--locomotion-transport axis requires --axis-profile for a command that moves "
            "the robot. The transport ships no raw axis value for any direction and will "
            "not invent one: a profile supplies each primitive together with the evidence "
            "reference behind it. (stop and status do not need one and do not ask.)")


# ── safety ────────────────────────────────────────────────────────────────────────────────
class _NullGuard:
    """Stands in for SafeStop where the robotkit is not deployed (a workstation sim)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def safe_stop_guard(loco, name="dashboard"):
    """The shared guaranteed-damp context manager, or a no-op where it is not deployed.

    ``arm_dc_robotkit.safe_stop.SafeStop`` turns SIGTERM/SIGINT and an unhandled exception
    into a damp. Its ``damp_fn`` here is ``loco.stop`` and NOT ``loco.damp``: on a Go2,
    ``damp()`` drops the robot into soft damping, which collapses it — correct for a robot
    that is already low or supported, and exactly wrong for one mid-stride. Stopping a
    velocity walk means commanding zero, which is what ``stop()`` does.

    This is what makes the driver's timeout path safe. The driver sends SIGTERM before
    SIGKILL precisely so this fires; a bare kill would leave ``SportClient.Move``'s last
    velocity latched, because that command has no dead-man timeout.

    When the robotkit is not importable the worker still stops the robot in its own
    ``finally``. The difference is only what happens on a signal — and a sim has no legs to
    leave moving.
    """
    try:
        from arm_dc_robotkit.safe_stop import SafeStop
    except ImportError:
        try:
            from safe_stop import SafeStop
        except ImportError:
            return _NullGuard()
    return SafeStop(loco.stop, name=name, verbose=False)


def _nothing_measured(platform):
    """True when NO axis on this platform has a measured gait floor.

    Derived from :data:`GAIT_FLOORS` rather than kept as a second list of platform names,
    so it cannot drift out of step with the row it describes: the first measurement to land
    on a Lite3 turns this off by itself, and nobody has to remember to edit two places.
    """
    floors = GAIT_FLOORS.get(platform, {})
    return bool(floors) and all(value is None for value in floors.values())


def check_gait_floor(platform, axis, speed, force=False):
    """Refuse a command measured not to produce a gait. Returns a warning string or None.

    Three states, not two, because "not measured" is not one thing:

    * **A measured floor.** Below it, refuse. This is the single most expensive failure in
      the repository's history and the worker will not reproduce it silently.
    * **One unmeasured axis on a robot that has walked** (the Go2's yaw). Warn, do not
      block: the operator should be TOLD on every press that this control's behaviour is
      unknown, but a robot with a proven gait is not a robot you refuse to turn.
    * **A robot where nothing has been measured at all** (the Lite3). Refuse. Nothing about
      this machine's gait is known, it has never moved under this stack, and its own
      navigator already fails closed the same way — ``lite3/visual_nav/robot_bindings.py``
      answers a live run with no ``--gait-floor`` with "REFUSING TO WALK: missing".
      A dashboard button is not a weaker authority than that, so it does not get a weaker
      rule. ``--force`` is the operator's documented way past it, as everywhere else here.
    """
    floor = GAIT_FLOORS.get(platform, {}).get(axis)
    if floor is None:
        if _nothing_measured(platform):
            if force:
                return (f"no {axis} gait floor exists for the {platform} — no axis on this "
                        "platform has ever been measured; forced")
            raise BridgeError(
                f"no gait floor has ever been measured on the {platform}: not on {axis}, "
                "and not on any other axis. This robot has never moved under this stack, so "
                "there is no speed here that is known to walk and none that is known not "
                "to. Measure it first (issue #13, and the evidence table in "
                "robot-stack/deep_robotics/lite3/README.md), or pass --force and watch it.")
        return (f"no {axis} gait floor has been measured on the {platform}; this command "
                "may produce no movement at all and that would not be a fault")
    if abs(speed) >= floor or floor == 0.0:
        return None
    if force:
        return (f"commanded {abs(speed):.3f} m/s is below the measured {axis} floor "
                f"{floor:.3f}; forced")
    raise BridgeError(
        f"{abs(speed):.3f} is below this {platform}'s measured {axis} gait floor of "
        f"{floor:.3f}. Below it the robot does not walk: it stands still with no fault and "
        "every instrument agreeing it is fine. Raise the speed or pass --force.")


# ── the nudge ─────────────────────────────────────────────────────────────────────────────
def _pose_tuple(loco):
    try:
        pose = loco.pose()
    except Exception as exc:
        return None, str(exc)
    return {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}, None


def run_nudge(loco, vx, vy, wz, seconds, sleep=time.sleep, clock=time.monotonic):
    """Hold a velocity for ``seconds``, refreshing it, then stop. Returns a measurement.

    The ``finally`` is the whole point: every exit path — normal, exception, or the SIGTERM
    the driver sends on a timeout — stops the robot. ``SportClient.Move`` persists until the
    next command, so a return without a stop is a robot that keeps walking.
    """
    before, _ = _pose_tuple(loco)
    period = 1.0 / COMMAND_HZ
    started = clock()
    ticks = 0
    try:
        while clock() - started < seconds:
            loco.set_velocity(vx, vy, wz)
            ticks += 1
            sleep(period)
    finally:
        # Suppressed, not handled: if the stop itself fails there is nothing further this
        # process can do about it, and the exception that brought us here (if any) is the
        # more useful one to propagate.
        with contextlib.suppress(Exception):
            loco.stop()
    elapsed = clock() - started
    after, _ = _pose_tuple(loco)

    travelled = None
    turned = None
    if before and after:
        travelled = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
        turned = _wrap(after["yaw"] - before["yaw"])

    commanded_speed = math.hypot(vx, vy)
    result = {
        "ok": True,
        "commanded": {"vx": vx, "vy": vy, "wz": wz},
        "seconds": round(elapsed, 3),
        "ticks": ticks,
        "pose_before": before,
        "pose_after": after,
        "travelled_m": None if travelled is None else round(travelled, 4),
        "turned_rad": None if turned is None else round(turned, 4),
        "turned_deg": None if turned is None else round(math.degrees(turned), 2),
    }
    # The delivery ratio is the number this repository keeps re-learning: what the robot did
    # over what it was told to do. Only meaningful when something was commanded and time
    # actually passed.
    if travelled is not None and commanded_speed > 0.0 and elapsed > 0.0:
        result["delivered_fraction"] = round(travelled / (commanded_speed * elapsed), 3)
    return result


# ── commands ──────────────────────────────────────────────────────────────────────────────
def command_status(loco, platform):
    pose, pose_error = _pose_tuple(loco)
    try:
        velocity = [float(v) for v in loco.velocity()]
    except Exception as exc:
        velocity, pose_error = None, str(exc)
    mode = None
    if hasattr(loco, "current_mode"):
        try:
            mode = loco.current_mode()
        except Exception as exc:
            mode = f"unreadable: {exc}"
    return {"ok": True, "platform": platform, "pose": pose, "velocity": velocity,
            "mode": mode, "error": pose_error}


def command_pose_stream(loco, platform, hz, out=None, sleep=time.sleep,
                        clock=time.monotonic, limit=None):
    """Print one pose line per sample, forever, so a peer can be avoided over the mesh.

    ⚠️ **This is the one command in this file that does not exit after one result**, and
    the exception is argued rather than assumed. The module's "one command, one JSON line,
    exit" rule is a SAFETY property: every command runs in a process that exits, so a
    driver that hangs or is killed cannot leave a velocity latched on the bus. That rule
    protects against a latched velocity. This command never sets one — it calls
    ``pose()`` and ``velocity()`` and nothing else — so there is no latch for its lifetime
    to extend, and it is deliberately NOT wrapped in ``safe_stop_guard``: stopping the
    robot on the way out of a READ would be a pose reader that stops a peer's legs, and
    the peer is very likely walking under the dashboard's own motion RPC while this runs.

    It has to be persistent because the alternative measured out. ``robot_driver`` polls
    ``status`` as a subprocess every 5 s and budgets ``MAX_SECONDS + 25`` for one, because
    process start, SDK import and DDS discovery are the slow part on a cold Jetson. A peer
    pose is needed at 10 Hz. One process per sample is not an implementation of that.

    ``mono_s`` is this process's own ``time.monotonic()`` at the moment the estimator was
    read. It is meaningless to anyone off this host and it is not meant to leave it: the
    driver, which is another process on the SAME machine, subtracts it from its own
    ``time.monotonic()`` to turn it into an AGE — a duration — and only the duration goes
    on the mesh. See ``integration/peer_source.py`` for why no timestamp can make that
    trip and a duration can.

    ``limit`` stops after that many samples and exists for the tests; a run leaves it None.
    """
    out = sys.stdout if out is None else out
    period = 1.0 / max(0.1, float(hz))
    emitted = 0
    while limit is None or emitted < limit:
        started = clock()
        pose, error = _pose_tuple(loco)
        try:
            velocity = [float(v) for v in loco.velocity()]
        except Exception as exc:
            velocity, error = None, str(exc)
        # A failed read is REPORTED, not skipped. A skipped sample is indistinguishable
        # from a network drop at the far end, and the far end's answer to a drop is to
        # stop the robot — which is right for a drop and an unhelpfully vague diagnosis
        # for an estimator that is throwing.
        line = {"ok": pose is not None and velocity is not None, "platform": platform,
                "mono_s": started, "pose": pose, "velocity": velocity, "error": error}
        try:
            out.write(json.dumps(line) + "\n")
            out.flush()
        except (BrokenPipeError, ValueError):
            # The driver has gone. Exiting here rather than raising: the parent is the
            # only consumer, and a traceback about a closed pipe would be the last thing
            # in its log about a shutdown it initiated.
            return {"ok": True, "streamed": emitted, "stopped": "parent closed stdout"}
        emitted += 1
        sleep(max(0.0, period - (clock() - started)))
    return {"ok": True, "streamed": emitted}


def command_lie_down(loco, platform):
    """Lie the robot down — and be honest that this means two different things.

    On a Go2 ``stand_down()`` issues ``StandDown`` and the robot lies down. On a Lite3 it
    only stops: posture there is operator-controlled through the vendor app, deliberately,
    because the ROS bridge path keeps the manufacturer's balance controller in charge. A
    dashboard that showed the same green tick for both would be telling the operator
    something untrue about where their robot is.
    """
    loco.stand_down()
    if platform == "lite3":
        return {"ok": True, "postured": False, "note":
                "the Lite3 was STOPPED, not laid down. Posture on this platform is "
                "operator-controlled through the vendor app; the ROS bridge exposes no "
                "lie-down command and this driver will not fake one."}
    return {"ok": True, "postured": True, "note": "StandDown issued"}


def command_stop(args, link):
    """Zero the velocity — and never claim to have done it when it was not done.

    ⛔ **This is the safety command in this file and issue #141 is about it.** It used to
    run inside ``dispatch`` after ``load_platform``, so on a Lite3 with no ROS 2 it died at
    ``ModuleNotFoundError`` before ``loco.stop()`` — a stop that depended on the thing it
    was stopping. It owns its own loading now, and the three outcomes are told apart in the
    reply rather than collapsed into one failure:

    * ``commanded_zero: true`` — a zero went out and it is authoritative.
    * ``commanded_zero: false`` with ``ok: true`` — the axis transport, where a process
      that did not start the stream has no setpoint to zero and ``stop()`` sends nothing.
      The note says which lever actually ends that walk, because a tick over nothing here
      is exactly the failure this repository keeps writing tests about.
    * ``ok: false`` with :data:`CAUSE_TRANSPORT_UNAVAILABLE` — the transport would not
      load, so **nothing reached the robot**, and the reply says that in those words
      rather than returning a generic error the page renders as a robot fault.

    Never gated by ``--allow-motion``: a stop you cannot issue because motion is disabled
    is the wrong affordance on a robot console.
    """
    backend = args.backend or args.platform
    described = link.describe() if backend == "lite3" else {}
    if backend == "lite3" and link.transport == "axis":
        # NOT a connection this worker declines to make out of caution -- it is a
        # connection that could not accomplish anything. AxisStreamSender lives in the
        # process that started it; a fresh process has `self._streamer is None`, so
        # Lite3AxisLocomotion.stop() returns having sent no datagram. Opening a socket and
        # waiting for the state stream first would add a way for the STOP path to fail,
        # in exchange for calling a method that is a no-op here.
        # ``stopped`` is False on purpose even though ``ok`` is True. ``ok`` means "the
        # stop did what this transport permits"; ``stopped`` and ``commanded_zero`` are
        # about what THIS WORKER put on the wire, which is nothing. No field on this reply
        # may claim an action that did not happen -- the note is where the explanation
        # goes, not a boolean.
        result = {"ok": True, "stopped": False, "commanded_zero": False,
                  "reached_robot": False, "platform": args.platform,
                  "note": "NO ZERO WAS SENT, and that is what this transport allows rather "
                          "than a failure. On --locomotion-transport axis the command "
                          "stream is a thread inside the process that started it, so a "
                          "separate process has no setpoint to zero. What ends an axis "
                          "walk is ENDING THAT PROCESS: robot_driver.stop() SIGTERMs the "
                          "motion worker before calling here, and the worker streams zeros "
                          "for 2 s on the way out. Once the datagrams stop it is the "
                          "vendor watchdog that zeroes the axes -- the command TTL is "
                          "capped below 250 ms because that is the documented watchdog, "
                          "and NOBODY HAS MEASURED IT ON EITHER VENTURE. If the robot is "
                          "still moving, use the physical abort."}
        result.update(described)
        return result
    try:
        loco = load_platform(backend, iface=args.iface,
                             operator_ready=args.operator_ready, link=link)
    except BridgeError:
        raise
    except TransportUnavailable as exc:
        result = {"ok": False, "cause": CAUSE_TRANSPORT_UNAVAILABLE, "refused": False,
                  "stopped": False, "commanded_zero": False, "reached_robot": False,
                  "platform": args.platform,
                  "error": "STOP DID NOT REACH THE ROBOT. " + str(exc) + " Nothing this "
                           "worker sends can reach the robot on this transport, which also "
                           "means no command from this dashboard has ever reached it on "
                           "this transport -- so this dashboard has not left a velocity "
                           "latched. It has also not stopped anything. If the robot is "
                           "moving, something else is driving it: USE THE PHYSICAL ABORT. "
                           "To command a real zero, restart the driver on a transport this "
                           "interpreter can load."}
        result.update(described)
        return result
    except Exception as exc:
        # A transport that LOADED and could not open. Still a fault about the real link --
        # the classification does not move -- but the reply keeps `commanded_zero`, so the
        # page renders "velocity NOT zeroed" rather than an error the operator has to read
        # to the end to learn that nothing was sent. The commonest one is `udp`'s connect
        # waiting 5 s for a state frame; see this module's docstring.
        result = {"ok": False, "cause": CAUSE_FAULT, "refused": False, "stopped": False,
                  "commanded_zero": False, "platform": args.platform,
                  "error": f"THE ZERO WAS NOT SENT: {type(exc).__name__}: {exc}"}
        result.update(described)
        return result
    try:
        loco.stop()
    finally:
        with contextlib.suppress(Exception):
            loco.shutdown()
    result = {"ok": True, "stopped": True, "commanded_zero": True, "reached_robot": True,
              "platform": args.platform}
    result.update(described)
    return result


def stop_guarantee(result, interrupted_motion=False, ended_run=False):
    """What a stop DID, and whether that adds up to a stop. Pure. ``(ok, note)``.

    Here rather than in ``robot_driver`` for the reason :func:`check_gait_floor` is here:
    this is the rule and the driver is the wiring. It also means the rule is exercised by a
    suite CI runs on both interpreters — ``test_robot_driver.py`` dies at
    ``ModuleNotFoundError`` on ``device_connect_edge``, which is not on PyPI before launch,
    so a safety rule that lived only there would be a rule no CI has ever run.

    Three levers end a walk and a stop pulls all three: the policy run, the in-flight nudge
    worker, and a zero velocity on the wire. The note names each and says which of them
    moved, because "stopped" on its own is a word an operator cannot check. ``ok`` is False
    when the transport would not load — that press did nothing at all and must not come
    back green — and True otherwise, including the ``axis`` case where no zero was sent
    because on that transport there was no zero to send and the worker was ended instead.
    """
    zeroed = bool(result.get("commanded_zero"))
    # Each clause says exactly what its boolean says and no more. "the worker was not
    # running" would be an inference: `_terminate_worker` also returns False for a worker
    # that had already exited, and for one whose terminate() raised.
    note = (("the policy run was ENDED" if ended_run else "no policy run was ended")
            + ("; the in-flight nudge worker was TERMINATED" if interrupted_motion
               else "; no nudge worker was terminated")
            + "; a zero velocity was " + ("COMMANDED" if zeroed else "NOT commanded")
            + ".")
    if result.get("cause") == CAUSE_TRANSPORT_UNAVAILABLE:
        return False, (note + " The transport would not load, so nothing this dashboard "
                       "sent could have reached the robot -- which also means this "
                       "dashboard cannot have left a velocity latched on it. It has not "
                       "stopped anything either. If the robot is moving, USE THE PHYSICAL "
                       "ABORT.")
    return bool(result.get("ok")), note


def dispatch(args):
    """Run one command and return its result dict."""
    if args.command in MOTION_COMMANDS and not args.allow_motion:
        raise BridgeError(
            f"{args.command} moves the robot and --allow-motion was not given. This is the "
            "same gate as --live: a dashboard that can walk a robot the moment it is "
            "discovered is a worse posture than the command line it replaces.")

    link = link_from_args(args)
    check_lite3_link(args.backend or args.platform, link, args.command)

    # BEFORE load_platform, and that placement is the whole of issue #141's first defect:
    # a stop that has to open the motion transport before it can command zero is a stop
    # that dies with the transport. See :func:`command_stop`.
    if args.command == "stop":
        return command_stop(args, link)

    # Plan the nudge BEFORE connecting. A speed below the gait floor is a refusal that can
    # be decided from a table, and connecting to DDS to deliver it would cost seconds on a
    # Jetson and put a client on the bus for a command that was never going to run.
    plan = plan_nudge(args) if args.command in ("walk", "strafe", "turn") else None

    # `--platform` is the RULES (gait floors, posture semantics); `--backend` is what is
    # actually driven. They differ only when simulating, and keeping them separate is what
    # makes a simulated Go2 refuse 0.21 m/s exactly like a real one. Collapsing them — which
    # the first version of --simulate did — silently gave a demo the bench double's floors of
    # zero, so the refusal that is this stack's most characteristic behaviour never fired.
    loco = load_platform(args.backend or args.platform, iface=args.iface,
                         operator_ready=args.operator_ready, link=link)
    try:
        if args.command == "status":
            result = command_status(loco, args.platform)
            if (args.backend or args.platform) == "lite3":
                result.update(link.describe())
            return result
        if args.command == "pose-stream":
            return command_pose_stream(loco, args.platform,
                                       min(args.hz, MAX_POSE_STREAM_HZ))

        # Everything past here moves the robot, so it runs inside the guaranteed-damp
        # guard. Entering it only for motion keeps a status poll from installing signal
        # handlers it has no use for.
        with safe_stop_guard(loco, name=args.command):
            if args.command == "stand":
                loco.stand()
                return {"ok": True, "standing": True}
            if args.command == "stand-down":
                return command_lie_down(loco, args.platform)
            vx, vy, wz, seconds, warning = plan
            result = run_nudge(loco, vx, vy, wz, seconds)
            result["platform"] = args.platform
            if (args.backend or args.platform) == "lite3":
                # ``transport_preserves_magnitude: false`` is what tells a reader that
                # ``delivered_fraction`` is not a delivery ratio here: the axis transport
                # threw the commanded number away before it reached the wire, so the ratio
                # describes the primitive rather than what the robot delivered. The ratio
                # is still reported -- deleting a measurement is worse than labelling it --
                # and one field says so rather than two that could drift apart.
                result.update(link.describe())
            if warning:
                result["warning"] = warning
            return result
    finally:
        with contextlib.suppress(Exception):
            loco.shutdown()


def plan_nudge(args):
    """Resolve one nudge to ``(vx, vy, wz, seconds, warning)``, or raise BridgeError.

    Pure: no robot, no clock. That is what lets the refusals be tested without a platform
    and decided without a connection.
    """
    seconds = min(args.seconds, MAX_SECONDS)
    if args.command == "walk":
        if args.vx < 0.0:
            # Reverse is not gait-floor checked. The floor is a measurement of the forward
            # gait and applying it here would be the axis conflation issue #42 is about;
            # what reverse needs is a shorter leash and a statement of what is not watching.
            return (args.vx, 0.0, 0.0, min(seconds, MAX_REVERSE_SECONDS),
                    "REVERSE. Neither platform has rear sensing and the planner never "
                    "samples this direction; nothing is watching behind the robot.")
        return (args.vx, 0.0, 0.0, seconds,
                check_gait_floor(args.platform, "forward", args.vx, args.force))
    if args.command == "strafe":
        return (0.0, args.vy, 0.0, seconds,
                check_gait_floor(args.platform, "lateral", args.vy, args.force))
    if args.command == "turn":
        return (0.0, 0.0, args.wz, seconds,
                check_gait_floor(args.platform, "yaw", args.wz, args.force))
    raise BridgeError(f"unknown command {args.command!r}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one bounded motion command on a quadruped and report what it did.")
    parser.add_argument("command", choices=sorted(
        MOTION_COMMANDS | {"status", "stop", "pose-stream"}))
    parser.add_argument("--platform", default="sim", choices=("go2", "lite3", "sim"),
                        help="Whose RULES apply: gait floors, posture semantics, warnings.")
    parser.add_argument("--backend", default="", choices=("", "go2", "lite3", "sim"),
                        help="What is actually driven. Defaults to --platform; set it to "
                             "'sim' to apply a real platform's rules to the bench double.")
    parser.add_argument("--iface", default=os.environ.get("GO2_DDS_IFACE", "eth0"),
                        help="DDS interface (Go2).")
    parser.add_argument("--operator-ready", action="store_true",
                        help="Lite3: the operator has confirmed STANDING + high-level "
                             "navigation mode on the vendor interface.")

    # THE SAME FLAG NAMES, THE SAME CHOICES AND THE SAME DEFAULT the Lite3 navigator and
    # the commissioning harness already use -- robot_bindings._add_ros_arguments and
    # robot_link.add_link_arguments. A dashboard that spelled this differently would be a
    # second vocabulary for one vendor interface, and the operator holding the runbook
    # would be reading about the other one.
    #
    # Every value defaults to None, meaning "the transport module's own default". Nothing
    # here restates a vendor port, host or topic; see :class:`Lite3Link`.
    #
    # NOT PLUMBED, deliberately: --axis-rate-hz, --axis-heartbeat-hz, --axis-command-ttl
    # and --axis-source-address. They are stream tuning with vendor minima that
    # AxisStreamSender validates on construction, and a dashboard nudge has no reason to
    # differ from the navigator on any of them. If one ever does, add it here rather than
    # inventing a dashboard-only name for it.
    link = parser.add_argument_group("Lite3 locomotion transport")
    link.add_argument("--locomotion-transport", default=None,
                      choices=sorted(LITE3_TRANSPORTS),
                      help="which vendor interface to command a Lite3 through. udp: the "
                           "high-level UDP interface directly, no ROS 2 (default); axis: "
                           "profile-gated moving-mode simple axes, the only one either "
                           "Venture has been seen to walk on; ros2: the Lite3_ROS bridge "
                           f"topics. Default {DEFAULT_LITE3_TRANSPORT}, which is the "
                           "navigator's default too.")
    link.add_argument("--motion-host", default=None,
                      help="Lite3 motion host address (udp and axis).")
    link.add_argument("--command-port", type=int, default=None,
                      help="motion host command port.")
    link.add_argument("--state-port", type=int, default=None,
                      help="local port the motion host streams state to; it must match "
                           "'ip'/'target_port' in ~/jy_exe/conf/network.toml.")
    link.add_argument("--state-bind", default=None,
                      help="local address for state telemetry.")
    link.add_argument("--axis-profile", default=None, metavar="PATH",
                      help="evidenced lite3-axis-profile JSON. Required by "
                           "--locomotion-transport axis for a command that MOVES the "
                           "robot, refused by the other transports, and not needed by "
                           "stop or status.")
    link.add_argument("--axis-local-port", type=int, default=None,
                      help="local port the axis stream sends from.")
    link.add_argument("--cmd-vel-topic", default=None,
                      help="Lite3_ROS Twist command topic (ros2 only).")
    link.add_argument("--odom-topic", default=None,
                      help="Lite3_ROS Odometry topic (ros2 only).")
    parser.add_argument("--allow-motion", action="store_true",
                        help="Permit commands that move the robot.")
    parser.add_argument("--force", action="store_true",
                        help="Command a speed below the measured gait floor anyway.")
    parser.add_argument("--vx", type=float, default=DEFAULT_VX,
                        help="m/s forward (negative = back).")
    parser.add_argument("--vy", type=float, default=DEFAULT_VY,
                        help="m/s left (negative = right).")
    parser.add_argument("--wz", type=float, default=DEFAULT_WZ,
                        help="rad/s CCW (negative = CW).")
    parser.add_argument("--seconds", type=float, default=1.5,
                        help=f"How long to hold the command, capped at {MAX_SECONDS}.")
    parser.add_argument("--hz", type=float, default=POSE_STREAM_HZ,
                        help=f"pose-stream sample rate, capped at {MAX_POSE_STREAM_HZ}.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except BridgeError as exc:
        result = {"ok": False, "refused": True, "cause": CAUSE_REFUSED, "error": str(exc)}
    except TransportUnavailable as exc:
        # ⚠️ `refused` stays FALSE here, because this was not a refusal -- but `cause` is
        # what a reader must branch on. Issue #141: with only the boolean, a dashboard
        # renders a missing Python module as the robot misbehaving, and an operator in
        # Shanghai spends the day on the robot instead of on the deployment.
        result = {"ok": False, "refused": False, "cause": CAUSE_TRANSPORT_UNAVAILABLE,
                  "reached_robot": False, "error": str(exc)}
    except Exception as exc:
        result = {"ok": False, "refused": False, "cause": CAUSE_FAULT,
                  "error": f"{type(exc).__name__}: {exc}"}
    # The result JSON is the LAST line of stdout. Everything the SDK prints on import goes
    # before it, which is why the driver reads backwards rather than parsing all of stdout.
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
