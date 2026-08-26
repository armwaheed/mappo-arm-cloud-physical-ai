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


class BridgeError(RuntimeError):
    """A refusal, reported as JSON rather than as a traceback."""


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
    from go2_locomotion import Go2Locomotion
    loco = Go2Locomotion(iface=iface)
    loco.connect()
    return loco


def _load_lite3(operator_ready):
    sys.path.insert(0, os.path.join(_stack_dir(), "deep_robotics", "lite3", "locomotion"))
    from lite3_locomotion import Lite3Locomotion
    loco = Lite3Locomotion(operator_ready=operator_ready)
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


def load_platform(platform, iface="eth0", operator_ready=False):
    if platform == "go2":
        return _load_go2(iface)
    if platform == "lite3":
        return _load_lite3(operator_ready)
    if platform == "sim":
        loco = SimLocomotion()
        loco.connect()
        return loco
    raise BridgeError(f"unknown platform {platform!r}; use go2, lite3 or sim")


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


def dispatch(args):
    """Run one command and return its result dict."""
    if args.command in MOTION_COMMANDS and not args.allow_motion:
        raise BridgeError(
            f"{args.command} moves the robot and --allow-motion was not given. This is the "
            "same gate as --live: a dashboard that can walk a robot the moment it is "
            "discovered is a worse posture than the command line it replaces.")

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
                         operator_ready=args.operator_ready)
    try:
        if args.command == "status":
            return command_status(loco, args.platform)
        if args.command == "pose-stream":
            return command_pose_stream(loco, args.platform,
                                       min(args.hz, MAX_POSE_STREAM_HZ))
        if args.command == "stop":
            loco.stop()
            return {"ok": True, "stopped": True}

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
        result = {"ok": False, "refused": True, "error": str(exc)}
    except Exception as exc:
        result = {"ok": False, "refused": False, "error": f"{type(exc).__name__}: {exc}"}
    # The result JSON is the LAST line of stdout. Everything the SDK prints on import goes
    # before it, which is why the driver reads backwards rather than parsing all of stdout.
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
