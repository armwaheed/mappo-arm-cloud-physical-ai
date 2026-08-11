#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Blended joint-space motion for the Unitree D1 arm.

The naive way to move a D1 is a funcode-1 command per joint, waiting for each to settle.
That produces exactly what you'd expect: stop-and-go, one joint at a time, visibly jerky.

``funcode 2`` sets all seven servos at once and its ``mode`` field selects a smoother:

    mode 0   "small smoothing of 10 Hz data"      — measured peak slew 66.7 deg/s
    mode 1   "large smoothing of trajectory use"  — measured peak slew 11.7 deg/s

⚠ **mode 1 is not the smooth mode. It is the SLOW mode.** Measured on unit
192.168.123.18 (2026-07-09), a 55 deg shoulder step: mode 1 crawls at 11.7 deg/s, mode 0
runs at 66.7 deg/s. mode 1 is meant for *sparse* waypoints you want the firmware to blend
slowly. mode 0 is documented as smoothing "10 Hz data" — which is exactly a 10 Hz stream.

So the smooth primitive is: **interpolate in joint space, ease it, and stream the
waypoints at the arm's 10 Hz control rate with mode 0.** We do the blending; the firmware
smooths within each 100 ms step. A raised cosine gives zero velocity at both endpoints,
so nothing lurches.

⚠ **Streaming waypoints does not mean the arm arrived.** The servos lag: at 16 deg/s the
shoulder was still 120 deg from its target when a 2.8 s stream ran out. :func:`stream`
therefore holds the final target and polls feedback until the arm converges. Pass
``read`` or you get an open-loop fire-and-forget, which is how a grasp gets attempted
mid-flight.

⚠ THE CLAMP TRAP
----------------
funcode 2 writes **every** joint, and the firmware clamps each to
:data:`d1_fk.COMMANDABLE_LIMITS_DEG`. If the arm is currently *outside* that envelope —
which happens whenever a human has hand-posed it, since the joints back-drive well past
their commandable range — then the very first frame you stream will yank that joint to the
limit. Measured 2026-07-09: the shoulder rests at -91.2 deg under gravity and a
hand-taught grasp sat at +96.7 deg; commanding -95 parks it at -90.3.

:func:`assert_commandable` refuses to stream from such a pose. Move the offending joint
with funcode 1 first, or re-teach inside the envelope.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

try:                                     # package import (unitree.go2.d1_arm.trajectory)
    from . import d1_fk as _d1_fk
except ImportError:                      # run directly, or sys.path'd to this directory
    import d1_fk as _d1_fk

COMMANDABLE_LIMITS_DEG = _d1_fk.COMMANDABLE_LIMITS_DEG

COMMAND_TOPIC = "rt/arm_Command"
FUNCODE_SET_ALL = 2
MODE_POINT = 0      # small smoothing
MODE_TRAJECTORY = 1  # large smoothing — blends successive waypoints

CONTROL_HZ = 10.0    # the D1's control cycle; streaming faster gains nothing

#: Measured peak joint slew, deg/s, 55 deg shoulder step, unit 192.168.123.18.
MEASURED_MAX_DEG_PER_S = {MODE_POINT: 66.7, MODE_TRAJECTORY: 11.7}

#: AVERAGE deg/s used to time a move. A raised cosine peaks at pi/2 = 1.57x the average,
#: so 35 avg -> 55 peak, comfortably inside the 66.7 the arm can actually do. Ask for more
#: and the arm simply falls behind the trajectory.
DEFAULT_MAX_DEG_PER_S = 35.0


def ease(t: float) -> float:
    """Raised cosine on [0,1]: zero velocity at both endpoints."""
    return 0.5 * (1.0 - math.cos(math.pi * min(max(t, 0.0), 1.0)))


def envelope_snap_mm(q7, kin=None) -> float:
    """How far the JAW jumps the instant this pose is first commanded.

    You cannot ease a joint back from outside the envelope: the firmware clamps every
    frame, so the first command — funcode 1 or 2, interpolated or not — drives that joint
    straight to its limit. Interpolation buys nothing. What matters is the size of the
    resulting Cartesian jump, which is what this returns.

    Two real cases on unit 192.168.123.18:
      * arm at rest, shoulder drooped to -91.2 deg  ->  a few mm. Harmless.
      * hand-taught grasp, shoulder at +96.7 deg    ->  ~64 mm. Would rake the target.
    """
    if kin is None:
        kin = _d1_fk.D1Kinematics()
    clamped = clamp_to_envelope(q7)
    return float(np.linalg.norm(kin.jaw_xyz(clamped[:6]) - kin.jaw_xyz(list(map(float, q7[:6]))))) * 1000.0


def assert_commandable(q7, max_snap_mm: float = 10.0, kin=None) -> None:
    """Raise if first-commanding this pose would jump the jaw more than ``max_snap_mm``.

    Guards on the physical consequence, not on a joint-angle tolerance: the joints
    routinely sit a fraction of a degree outside the envelope (gravity droop, back-drive)
    and that is fine. A hand-taught pose several degrees outside is not.
    """
    snap = envelope_snap_mm(q7, kin)
    if snap > max_snap_mm:
        bad = ", ".join(f"J{i}={q7[i]:+.2f} (limit +/-{COMMANDABLE_LIMITS_DEG[i]:.0f})"
                        for i in range(6)
                        if abs(q7[i]) > COMMANDABLE_LIMITS_DEG[i] + 1e-6)
        raise ValueError(
            f"commanding this pose would snap the jaw {snap:.0f} mm (limit {max_snap_mm:.0f} mm): "
            f"{bad}. The firmware clamps every frame, so this cannot be eased — re-teach "
            "the pose inside the envelope, or solve IK against COMMANDABLE_LIMITS_DEG.")


def clamp_to_envelope(q7, margin_deg: float = 0.0):
    """Project the 6 arm joints into the commandable envelope; leave the jaw alone.

    ``margin_deg`` pulls the target that far *inside* each limit. The envelope's edge is
    not a holdable set-point: commanded to 90.0, a loaded shoulder settles at 90.99 —
    still outside, so the next command snaps it again. Aim inside.
    """
    out = list(map(float, q7))
    for i in range(6):
        lim = max(0.0, COMMANDABLE_LIMITS_DEG[i] - margin_deg)
        out[i] = float(np.clip(out[i], -lim, lim))
    return out


def snap_into_envelope(publish, q7, hz: float = CONTROL_HZ, seq_start: int = 0,
                       settle_frames: int = 20, kin=None, margin_deg: float = 2.0) -> int:
    """Bring an out-of-envelope arm into its commandable range, and return the next seq.

    This SNAPS. It cannot do otherwise: the firmware clamps every frame, so the first
    command already carries the clamped target. Interpolating between an out-of-envelope
    start and its clamped self sends the same clamped value every frame. Check the cost
    with :func:`envelope_snap_mm` and decide before you call this.

    Being outside the envelope is the *normal* state after a teach or a power cycle —
    gravity parks the shoulder at -91.2 deg, and the joints back-drive several degrees
    past their limits by hand. The arm must already be ENABLED (funcode 5 ``{"mode": 1}``);
    discharged joints have zero torque and will not move.

    Note the envelope's edge is not a holdable set-point: commanded to 90.0, a loaded
    shoulder settles at 90.99 — still outside, so the next command snaps it again. Hence
    ``margin_deg``: aim a couple of degrees inside and let the droop land you in range.
    """
    target = clamp_to_envelope(q7, margin_deg=margin_deg)
    if all(abs(a - b) < 1e-6 for a, b in zip(target[:6], q7[:6])):
        return seq_start
    print(f"[trajectory] snapping into envelope (margin {margin_deg:.1f} deg): "
          f"jaw moves ~{envelope_snap_mm(q7, kin):.0f} mm")
    seq = seq_start
    for _ in range(settle_frames):
        publish(_frame(target, MODE_POINT, seq))
        seq += 1
        time.sleep(1.0 / hz)
    return seq


def plan_duration(q_from, q_to, max_deg_per_s: float = DEFAULT_MAX_DEG_PER_S,
                  min_s: float = 0.6, max_s: float = 8.0) -> float:
    """Seconds for the move, set by the joint that has furthest to travel."""
    span = max(abs(float(b) - float(a)) for a, b in zip(q_from, q_to))
    return float(np.clip(span / max_deg_per_s, min_s, max_s))


def interpolate(q_from, q_to, seconds: float, hz: float = CONTROL_HZ):
    """Yield eased joint-space waypoints (7 values each) at ``hz``."""
    n = max(2, int(round(seconds * hz)))
    a, b = np.asarray(q_from, float), np.asarray(q_to, float)
    for k in range(1, n + 1):
        yield a + (b - a) * ease(k / n)


def _frame(q7, mode: int, seq: int) -> str:
    data = {"mode": int(mode)}
    for i, v in enumerate(q7):
        data[f"angle{i}"] = round(float(v), 2)
    return json.dumps({"seq": seq, "address": 1, "funcode": FUNCODE_SET_ALL, "data": data})


def stream(publish, q_from, q_to, seconds: float | None = None, hz: float = CONTROL_HZ,
           mode: int = MODE_POINT, seq_start: int = 0,
           max_deg_per_s: float = DEFAULT_MAX_DEG_PER_S,
           read=None, tol_deg: float = 1.5, settle_s: float = 6.0) -> int:
    """Stream an eased trajectory from ``q_from`` to ``q_to``, then WAIT for arrival.

    ``publish`` takes one JSON string. ``q_from``/``q_to`` are 7 values (6 joints + jaw)
    in degrees. ``read`` returns the current 7 angles; without it this is open-loop and
    the caller cannot know the arm got there.

    Raises ``TimeoutError`` if the arm has not converged to within ``tol_deg`` on every
    arm joint after ``settle_s``. That is a real failure — acting on a pose the arm has
    not reached is how you close a gripper on empty air.
    """
    assert_commandable(q_from)
    seconds = plan_duration(q_from[:6], q_to[:6], max_deg_per_s) if seconds is None else seconds
    seq = seq_start
    dt = 1.0 / hz
    for wp in interpolate(q_from, q_to, seconds, hz):
        publish(_frame(wp, mode, seq))
        seq += 1
        time.sleep(dt)

    # Hold the target and let the servos catch up. They lag the trajectory badly.
    deadline = time.time() + settle_s
    while time.time() < deadline:
        publish(_frame(q_to, mode, seq))
        seq += 1
        time.sleep(dt)
        if read is not None:
            now = read()
            if max(abs(now[i] - q_to[i]) for i in range(6)) <= tol_deg:
                return seq
    if read is not None:
        now = read()
        err = max(abs(now[i] - q_to[i]) for i in range(6))
        if err > tol_deg:
            raise TimeoutError(
                f"arm did not reach the target in {seconds + settle_s:.1f}s "
                f"(worst joint off by {err:.1f} deg). Slew is ~"
                f"{MEASURED_MAX_DEG_PER_S[mode]:.0f} deg/s in mode {mode}.")
    return seq
