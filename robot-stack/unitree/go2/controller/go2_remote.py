# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 handheld controller — read buttons/sticks; trip an abort on a press.

The remote rides in ``LowState_.wireless_remote`` (a 40-byte ``xRockerBtnDataStruct``):
a little-endian uint16 button bitmask at bytes ``[2:4]`` followed by four float
axes. This is the SAME struct the G1 uses — only the ``LowState_`` IDL differs
(``unitree_go`` for the Go2 vs ``unitree_hg`` for the G1) — so the parser and the
abort latch are shared behaviour, verified button-by-button on the G1.

SAFETY — this is NOT an emergency stop. The controller's firmware-level damping
and the robot's physical power/e-stop work even if this process is hung or DDS has
stalled; they are the real stop, and you should keep them in reach. Use this only
for a *clean software abort* of an autonomous routine: it stops motion and holds
the stance (via :meth:`LocomotionController.set_abort_source`), it does not damp
the robot to the floor.
"""

from __future__ import annotations

import argparse
import struct
import threading
import time

# Unitree remote KeyMap — bit index = position in this tuple.
BUTTONS = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
           "A", "B", "X", "Y", "up", "right", "down", "left")


def parse_buttons(wireless_remote) -> int:
    """Button bitmask from the 40-byte ``wireless_remote`` struct (0 = none)."""
    raw = bytes(wireless_remote)
    return struct.unpack_from("<H", raw, 2)[0] if len(raw) >= 4 else 0


def parse_sticks(wireless_remote) -> dict:
    """The four analog axes (each in ~[-1, 1]) from the struct's float fields."""
    raw = bytes(wireless_remote)
    if len(raw) < 24:
        return {"lx": 0.0, "rx": 0.0, "ry": 0.0, "ly": 0.0}
    lx, rx, ry = struct.unpack_from("<fff", raw, 4)
    ly = struct.unpack_from("<f", raw, 20)[0]
    return {"lx": lx, "rx": rx, "ry": ry, "ly": ly}


def button_names(mask: int) -> list[str]:
    return [name for i, name in enumerate(BUTTONS) if mask & (1 << i)]


_dds_ready = False


def _ensure_dds(iface: str) -> None:
    global _dds_ready
    if _dds_ready:
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, iface)
    _dds_ready = True


class Go2Remote:
    """Watches the handheld controller and latches an abort on ANY button press.

    Arms only once the buttons are seen released, so a button held at start-up
    does not trip it immediately; then :meth:`aborted` latches True on the first
    press. Wire :meth:`aborted` into ``LocomotionController.set_abort_source`` and
    poll it between routine steps. Pass ``init_dds=False`` if another component
    already initialised the DDS channel factory in this process.
    """

    def __init__(self, iface: str = "eth0", init_dds: bool = True) -> None:
        self._iface = iface
        self._init_dds = init_dds
        self._sub = None
        self._lock = threading.Lock()
        self._mask = 0
        self._sticks = {"lx": 0.0, "rx": 0.0, "ry": 0.0, "ly": 0.0}
        self._armed = False
        self._tripped = False
        self._have_frame = False

    def connect(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        if self._init_dds:
            _ensure_dds(self._iface)
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)

        deadline = time.monotonic() + 3.0
        while not self._have_frame and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._have_frame:
            self.shutdown()  # don't leak the subscriber on the error path
            raise RuntimeError("no rt/lowstate in 3 s — is the robot on?")

    def _on_state(self, msg) -> None:
        mask = parse_buttons(msg.wireless_remote)
        sticks = parse_sticks(msg.wireless_remote)
        with self._lock:
            self._mask = mask
            self._sticks = sticks
            self._have_frame = True
            if not self._armed:
                self._armed = (mask == 0)          # arm once buttons are released
            elif mask != 0:
                self._tripped = True               # latch on the first press

    # ── Reads ───────────────────────────────────────────────────────────────
    def aborted(self) -> bool:
        with self._lock:
            return self._tripped

    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def pressed(self) -> list[str]:
        with self._lock:
            return button_names(self._mask)

    def sticks(self) -> dict:
        with self._lock:
            return dict(self._sticks)

    def reset(self) -> None:
        """Clear the latch and re-disarm (next clean release re-arms)."""
        with self._lock:
            self._tripped = False
            self._armed = False

    def wait_until_armed(self, timeout_s: float = 5.0) -> bool:
        """Block until the controller is armed (buttons released). Call before
        starting a routine so a residual press doesn't trip it instantly."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.armed():
                return True
            time.sleep(0.02)
        return self.armed()

    def shutdown(self) -> None:
        if self._sub is not None:
            try:
                self._sub.Close()
            except Exception:
                pass
            self._sub = None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Go2 controller read-only dry test (no motion). Press buttons "
                    "to confirm the abort watcher sees them.")
    ap.add_argument("--iface", default="eth0", help="DDS network interface")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    remote = Go2Remote(iface=args.iface)
    remote.connect()
    print("READ-ONLY — no motion. Press any button; expect 'ABORTED=True' to latch.")
    print(f"armed={remote.armed()} (waiting for a clean button release to arm)")

    end = time.monotonic() + args.seconds
    last = None
    while time.monotonic() < end:
        snap = (tuple(remote.pressed()), remote.armed(), remote.aborted())
        if snap != last:
            names, armed, aborted = snap
            print(f"  buttons={list(names) or '—'}  armed={armed}  ABORTED={aborted}")
            last = snap
        time.sleep(0.05)
    remote.shutdown()


if __name__ == "__main__":
    main()
