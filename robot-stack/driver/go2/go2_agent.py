#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The Unitree Go2 EDU (+ D1 arm) as an MHS device.

Registers the physical Go2 on the MHS fabric so an orchestrator (or a human agent)
can drive it: walk to a goal, stand/sit, move the D1 arm, read status — with
`arrived` / `blocked` / `arm_moved` events on the fabric event stream.

  get_status()                 -> motion mode + measured pose
  walk_forward(distance)       -> closed-loop walk forward, then stop
  walk_to(x, y)                -> closed-loop walk to an odom-frame point
  stand() / stand_down() / stop()
  arm_reset() / arm_move(joint_angles, gripper)

ARCHITECTURE — why this delegates to a subprocess:
  The Go2 control code drives `unitree_sdk2py` over CycloneDDS and lives in the robot's SDK env
  (Python 3.8 on the Go2 Jetson). MHS wants Python >=3.10, so this driver keeps its own env free of
  the DDS stack and drives the robot by invoking `go2_drive.py` in the SDK env — the two-env bridge (see
  the bootstrap-mhs-env skill). It reuses the verified control in `unitree/go2` (Go2Locomotion, D1Arm)
  and adds no new robot logic.

SAFETY: motion RPCs move the robot. They are gated behind `allow_motion=True` (the worker also requires
`--allow-motion`), so a default Go2 MHS device is STATUS-ONLY. Enable motion only with a clear area, an
operator on the controller abort ([`unitree/go2/controller`]) and the e-stop, and adequate battery.

Run ON the robot:
  python go2_agent.py --creds /path/to/<robot>.creds.json [--allow-motion]
Status-only smoke test (no fabric):
  python go2_agent.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import subprocess

from arm_mhs_robotkit.mhs_sidecar import (
    HAVE_MHS, DeviceDriver, rpc, emit, DeviceIdentity, DeviceStatus, run_device_from_creds,
    DEFAULT_NATS_URL,
)

log = logging.getLogger("go2-agent")

# Defaults for delegating control to the SDK env on the robot.
DEFAULT_DRIVE_PY = "/home/unitree/robotics-connect-go2/bin/python"
DEFAULT_DRIVE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_drive.py")


def drive_blocking(subcommand: str, args, drive_py: str, drive_script: str, dds_iface: str,
                   allow_motion: bool, *, gripper=None, timeout: float = 60.0) -> dict:
    """Run one go2_drive.py command in the SDK env; return its parsed JSON result."""
    cmd = [drive_py, drive_script, subcommand]
    cmd += [str(a) for a in (args or [])]
    if gripper is not None:
        cmd += ["--gripper", str(gripper)]
    cmd += ["--iface", dds_iface]
    if allow_motion:
        cmd.append("--allow-motion")
    env = dict(os.environ, GO2_DDS_IFACE=dds_iface)
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        log.warning("go2_drive %s failed to start: %r", subcommand, exc)
        return {"ok": False, "error": repr(exc)}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # SIGTERM first so the worker's SafeStop stops the robot, THEN SIGKILL — a
        # bare timeout SIGKILL would bypass the safe-stop while the robot is moving.
        proc.terminate()
        try:
            proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        log.warning("go2_drive %s timed out after %ss; terminated", subcommand, timeout)
        return {"ok": False, "error": f"go2_drive {subcommand} timed out after {timeout}s"}
    for line in reversed((stdout or "").strip().splitlines()):  # result JSON is the last line
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {"ok": False, "error": f"go2_drive rc={proc.returncode}, no JSON; "
                                  f"stderr={(stderr or '')[-300:]!r}"}


class Go2AgentDriver(DeviceDriver):
    """The Unitree Go2 EDU (+ D1 arm) as an MHS device."""

    device_type = "unitree_go2"

    def __init__(self, name: str = "Go2 EDU", drive_py: str = DEFAULT_DRIVE_PY,
                 drive_script: str = DEFAULT_DRIVE_SCRIPT, dds_iface: str = "eth0",
                 allow_motion: bool = False):
        if HAVE_MHS:
            super().__init__()
        self.name = name
        self.drive_py = drive_py
        self.drive_script = drive_script
        self.dds_iface = dds_iface
        self.allow_motion = allow_motion

    @property
    def identity(self) -> "DeviceIdentity":
        return DeviceIdentity(
            device_type="unitree_go2",
            manufacturer="Unitree",
            model=f"Go2 EDU ({self.name})" if self.name and self.name != "Go2 EDU" else "Go2 EDU",
            description=(f"{self.name} — a Unitree Go2 quadruped (12-DOF) with a Unitree D1 back arm. "
                        f"Walks to a goal, stands/sits, and manipulates with the arm, over MHS. "
                        f"Motion is {'ENABLED' if self.allow_motion else 'disabled (status-only)'}."),
        )

    @property
    def status(self) -> "DeviceStatus":
        return DeviceStatus(availability="available")

    @emit()
    async def arrived(self, x: float, y: float):
        """The robot reached a walk goal."""

    @emit()
    async def blocked(self, reason: str):
        """A walk stopped early (stalled / aborted / timeout)."""

    @emit()
    async def arm_moved(self, sent_angles):
        """The D1 arm executed a move (the clamped angles actually sent)."""

    # ── control RPCs ─────────────────────────────────────────────────────────
    async def _drive(self, subcommand, args=None, *, gripper=None) -> dict:
        loop = asyncio.get_running_loop()
        # run_in_executor forwards only positional args; drive_blocking takes `gripper`
        # keyword-only, so bind it with partial rather than passing it positionally.
        call = functools.partial(
            drive_blocking, subcommand, args, self.drive_py, self.drive_script,
            self.dds_iface, self.allow_motion, gripper=gripper)
        return await loop.run_in_executor(None, call)

    @rpc()
    async def get_status(self) -> dict:
        """Motion mode + measured pose (read-only; always allowed)."""
        return await self._drive("status")

    @rpc()
    async def stop(self) -> dict:
        """Stop the robot (zero velocity, hold stance). Always allowed."""
        return await self._drive("stop")

    @rpc()
    async def stand(self) -> dict:
        """Enter a balanced stand, ready to walk."""
        return await self._drive("stand")

    @rpc()
    async def stand_down(self) -> dict:
        """Lie the robot down."""
        return await self._drive("stand-down")

    @rpc()
    async def walk_forward(self, distance: float) -> dict:
        """Walk `distance` metres forward (closed-loop), then stop."""
        result = await self._drive("walk-forward", [distance])
        await self._emit_walk_result(result)
        return result

    @rpc()
    async def walk_to(self, x: float, y: float) -> dict:
        """Walk to odom-frame point (x, y) (closed-loop), then stop."""
        result = await self._drive("walk-to", [x, y])
        await self._emit_walk_result(result)
        return result

    @rpc()
    async def arm_reset(self) -> dict:
        """Return the D1 arm to its zero posture."""
        return await self._drive("arm-reset")

    @rpc()
    async def arm_move(self, joint_angles, gripper: "float | None" = None) -> dict:
        """Move the D1 arm: `joint_angles` = 6 joint targets (rad); optional jaw `gripper` (rad)."""
        result = await self._drive("arm-move", list(joint_angles), gripper=gripper)
        if result.get("ok"):
            await self.arm_moved(sent_angles=result.get("sent_angles"))
        return result

    async def _emit_walk_result(self, result: dict) -> None:
        if not isinstance(result, dict):
            return
        if result.get("ok"):
            pose = result.get("pose") or {}
            await self.arrived(x=pose.get("x", 0.0), y=pose.get("y", 0.0))
        elif result.get("result"):
            await self.blocked(reason=str(result.get("result")))


# ──────────────────────────────────────────────────────────────────────────────────────────────
def _run_fabric(args) -> None:
    driver = Go2AgentDriver(name=args.name, drive_py=args.drive_python,
                            drive_script=args.drive_script, dds_iface=args.dds_iface,
                            allow_motion=args.allow_motion)

    def _on_ready(_mounted) -> None:
        log.info("Go2 agent '%s' live on the MHS fabric; drive via %s; motion=%s",
                 args.name, args.drive_python, "ENABLED" if args.allow_motion else "disabled")

    run_device_from_creds(driver, args.creds, device_id=args.device_id, broker=args.nats_url,
                          on_ready=_on_ready)


def main() -> None:
    ap = argparse.ArgumentParser(description="The Unitree Go2 EDU (+ D1 arm) as an MHS device.")
    ap.add_argument("--creds", help="NATS creds JSON (registers the robot on the dashboard fabric).")
    ap.add_argument("--device-id", default=None, help="Override device id (default: from creds).")
    ap.add_argument("--nats-url", default=None, help="Override NATS url.")
    ap.add_argument("--name", default=os.environ.get("GO2_AGENT_NAME", "Go2 EDU"),
                    help="Robot display name.")
    ap.add_argument("--drive-python", default=DEFAULT_DRIVE_PY, help="Python in the SDK env.")
    ap.add_argument("--drive-script", default=DEFAULT_DRIVE_SCRIPT, help="Path to go2_drive.py.")
    ap.add_argument("--dds-iface", default=os.environ.get("GO2_DDS_IFACE", "eth0"), help="DDS interface.")
    ap.add_argument("--allow-motion", action="store_true",
                    help="Enable motion RPCs (default: status-only). Requires a clear area + e-stop.")
    ap.add_argument("--self-test", action="store_true", help="Status-only smoke test and exit (no fabric).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-9s  %(levelname)-7s  %(message)s")

    if args.self_test:
        result = drive_blocking("status", [], args.drive_python, args.drive_script,
                                args.dds_iface, allow_motion=False)
        print(json.dumps(result, indent=2))
        return
    if not args.creds:
        ap.error("--creds is required (or use --self-test for a status-only check).")
    try:
        _run_fabric(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
