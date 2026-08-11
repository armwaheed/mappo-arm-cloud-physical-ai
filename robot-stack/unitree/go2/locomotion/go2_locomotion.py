# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 binding for the robot-agnostic locomotion layer.

:class:`Go2Locomotion` drives the Go2 through Unitree's high-level ``SportClient``
— the manufacturer's balance/gait controller (the Go2 analogue of the G1's
``LocoClient``) — and reads **measured** odometry from the ``rt/sportmodestate``
DDS topic (``SportModeState_``: ``position``, ``velocity``, ``yaw_speed``,
``imu_state.rpy``). No reinforcement-learning policy is pushed onto the legs;
walking is the vendor's velocity-command interface, so balance and gait are the
controller's responsibility. (The low-level RL deploy path is a separate module,
``unitree/go2/deploy``.)

WHAT'S DIFFERENT FROM THE G1 — and why this is not just a copy:

  * **Mode + lease.** A Go2 only accepts ``SportClient`` commands when its motion
    service is in a sport mode ("normal"/"ai") and no other client holds the
    lease. The single most common "my Go2 ignores velocity commands" cause is the
    wrong mode / a held lease. :meth:`connect` reports the live mode via
    ``MotionSwitcherClient`` and :meth:`ensure_sport_mode` / :meth:`release_control`
    make the mode explicit — the G1 has no equivalent.
  * **Persistent Move, no dead-man.** ``SportClient.Move`` commands a velocity that
    persists until the next command — there is no built-in timeout. So EVERY exit
    path must ``StopMove`` and callers MUST guard a walk in a ``try/finally`` that
    stops the robot (the closed-loop helpers in ``arm_dc_robotkit.locomotion`` already do). Pair
    this with the controller abort (``unitree/go2/controller``) and ``arm_dc_robotkit.safe_stop``.

Frames: body ``+x`` forward / ``+y`` left, planar pose in the estimator's odom
frame. The shared closed-loop helpers (``walk_to``, ``turn_to``, ``walk_forward``)
live in ``arm_dc_robotkit.locomotion``.

Safety: ``set_velocity`` / ``stand`` / ``recover`` move the legs. The caller is
responsible for a clear area, an operator on the e-stop, and adequate battery.
"""

from __future__ import annotations

import argparse
import math
import threading
import time

from arm_dc_robotkit.locomotion import LocomotionController, Pose

ODOM_TOPIC = "rt/sportmodestate"
# Same stall-gate DESIGN as the G1 (only a sustained near-total stop while commanded
# to move counts as blocked), but tuned tighter: the Go2 trot dips toward zero
# forward speed between steps far less than the G1's slow stepping gait, so a shorter
# grace (3.0 s vs the G1's 4.0 s) is enough without false-tripping.
STALL_SPEED_FRACTION = 0.10  # measured/commanded speed below this counts as stalled
STALL_GRACE_S = 3.0          # ...sustained for this long → blocked

_dds_ready = False


def _ensure_dds(iface: str) -> None:
    """Initialise the Cyclone DDS channel factory once per process."""
    global _dds_ready
    if _dds_ready:
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, iface)
    _dds_ready = True


class Go2Locomotion(LocomotionController):
    """Velocity-walk + measured-odometry control of a Unitree Go2."""

    def __init__(self, iface: str = "eth0", init_dds: bool = True,
                 enable_lease: bool = False) -> None:
        self._iface = iface
        self._init_dds = init_dds
        self._enable_lease = enable_lease
        self._client = None          # SportClient
        self._switcher = None        # MotionSwitcherClient
        self._sub = None
        self._lock = threading.Lock()
        self._state = None           # latest SportModeState_
        self._mode: str | None = None
        self._stall_since: float | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def connect(self) -> None:
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        if self._init_dds:
            _ensure_dds(self._iface)

        self._client = SportClient(enableLease=self._enable_lease)
        self._client.SetTimeout(5.0)
        self._client.Init()

        # Report the live motion mode — the lease/mode check the G1 doesn't need.
        self._switcher = MotionSwitcherClient()
        self._switcher.SetTimeout(5.0)
        self._switcher.Init()
        self._mode = self._read_mode()

        self._sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
        self._sub.Init(self._on_state, 10)

        deadline = time.monotonic() + 3.0
        while self._state is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._state is None:
            self.shutdown()  # don't leak the subscriber on the error path
            raise RuntimeError(f"no {ODOM_TOPIC} in 3 s — is the robot on?")
        p = self.pose()
        mode = self._mode or "(unknown)"
        print(f"[Go2Locomotion] odom live  mode={mode!r}  pose=({p.x:+.2f}, "
              f"{p.y:+.2f}, {math.degrees(p.yaw):+.1f}°)")
        if self._mode not in ("normal", "ai"):
            print(f"[Go2Locomotion] WARNING: motion mode is {mode!r}, not a sport "
                  f"mode — Move commands may be ignored. Call ensure_sport_mode().")

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg

    def _read_mode(self) -> str | None:
        """Current motion-service mode name ('normal'/'ai'/…) or None if unavailable."""
        try:
            code, data = self._switcher.CheckMode()
        except Exception as exc:
            print(f"[Go2Locomotion] CheckMode failed: {exc!r}")
            return None
        if code != 0 or not isinstance(data, dict):
            return None
        return data.get("name") or None

    def shutdown(self) -> None:
        try:
            self.stop()  # never let cleanup raise (may run from connect()'s error path)
        except Exception:
            pass
        if self._sub is not None:
            try:
                self._sub.Close()
            except Exception:
                pass
            self._sub = None

    # ── Mode / lease (Go2-specific) ─────────────────────────────────────────
    def current_mode(self) -> str | None:
        """Re-read and return the live motion-service mode name."""
        self._mode = self._read_mode()
        return self._mode

    def ensure_sport_mode(self, mode: str = "normal") -> str | None:
        """Select ``mode`` ('normal' or 'ai') if not already active, so ``Move``
        is accepted. NOTE: switching modes reconfigures the controller and can make
        the robot shift/stand — only call it with a clear area + e-stop ready."""
        if self._read_mode() == mode:
            return mode
        print(f"[Go2Locomotion] switching motion mode -> {mode!r}")
        self._switcher.SelectMode(mode)
        time.sleep(1.0)  # let the controller come up before commanding it
        self._mode = self._read_mode()
        return self._mode

    def release_control(self) -> None:
        """Release the sport mode/lease — required before low-level (rt/lowcmd)
        control (see unitree/go2/deploy). High-level Move stops working after this
        until a sport mode is re-selected."""
        if self._switcher is not None:
            self._switcher.ReleaseMode()
            self._mode = None

    # ── Posture verbs ───────────────────────────────────────────────────────
    def stand(self) -> None:
        """Enter a balanced stand, ready to walk (``BalanceStand``). The robot must
        already be up (use :meth:`recover` from lying/fallen first)."""
        self._client.BalanceStand()

    def recover(self) -> None:
        """Recover to a stand from lying/fallen (``RecoveryStand``)."""
        self._client.RecoveryStand()

    def stand_down(self) -> None:
        """Lie the robot down (``StandDown``)."""
        self._client.StandDown()

    def damp(self) -> None:
        """Drop to soft damping (``Damp``). Collapses the robot — only when it is
        supported or already low; never mid-stride."""
        self._client.Damp()

    def speed_level(self, level: int) -> None:
        """Set the gait speed level (``-1`` slow … ``1`` fast; SDK-defined)."""
        self._client.SpeedLevel(level)

    @property
    def client(self):
        """The raw ``SportClient`` for the full gait menu (StaticWalk, TrotRun,
        ClassicWalk, obstacle modes, tricks). Kept out of this interface so the
        robot-agnostic surface stays small; reach through here for Go2 specials."""
        return self._client

    # ── Primitives ──────────────────────────────────────────────────────────
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        # SportClient.Move commands a velocity that PERSISTS until the next command
        # (no dead-man). The closed loop refreshes it ~10×/s; every exit path must
        # StopMove and callers must guard the walk in a finally that stops the robot.
        self._client.Move(vx, vy, vyaw)

    def stop(self) -> None:
        if self._client is not None:
            self._client.StopMove()

    def pose(self) -> Pose:
        with self._lock:
            state = self._state
        if state is None:
            return Pose(0.0, 0.0, 0.0)
        pos = state.position
        yaw = float(state.imu_state.rpy[2])
        return Pose(float(pos[0]), float(pos[1]), yaw)

    def velocity(self) -> tuple[float, float, float]:
        """Measured body-frame velocity ``(vx, vy, vyaw)`` from the estimator."""
        with self._lock:
            state = self._state
        if state is None:
            return (0.0, 0.0, 0.0)
        return (float(state.velocity[0]), float(state.velocity[1]),
                float(state.yaw_speed))

    def is_blocked(self, commanded_vx: float) -> str | None:
        """Stalled if commanded to move but the measured speed stays near zero."""
        if commanded_vx <= 0.05:
            self._stall_since = None
            return None
        vx, vy, _ = self.velocity()
        moving = math.hypot(vx, vy) >= STALL_SPEED_FRACTION * commanded_vx
        if moving:
            self._stall_since = None
            return None
        now = time.monotonic()
        if self._stall_since is None:
            self._stall_since = now
        elif now - self._stall_since >= STALL_GRACE_S:
            self._stall_since = None
            return "stalled"
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 locomotion diagnostics.")
    ap.add_argument("--iface", default="eth0", help="DDS network interface")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="how long to stream the measured pose (read-only)")
    ap.add_argument("--forward", type=float, default=0.0,
                    help="DANGER: walk this many metres forward, then stop")
    args = ap.parse_args()

    loco = Go2Locomotion(iface=args.iface)
    loco.connect()
    try:
        if args.forward > 0.0:
            reply = input(f"About to WALK {args.forward:.2f} m forward — the legs "
                          f"will move. Clear area + e-stop ready? [type 'walk']: ")
            if reply.strip().lower() != "walk":
                print("aborted.")
                return
            loco.ensure_sport_mode("normal")
            loco.stand()
            result = loco.walk_forward(args.forward)
            print(f"[walk_forward] result={result!r}  pose={loco.pose()}")
            return

        print(f"streaming measured pose for {args.seconds:.0f}s (no motion)…")
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            p = loco.pose()
            vx, vy, vyaw = loco.velocity()
            print(f"  pose=({p.x:+.2f}, {p.y:+.2f}, {math.degrees(p.yaw):+6.1f}°)  "
                  f"vel=({vx:+.2f}, {vy:+.2f}, {vyaw:+.2f})")
            time.sleep(0.5)
    finally:
        loco.stop()
        loco.shutdown()


if __name__ == "__main__":
    main()
