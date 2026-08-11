# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 onboard-LiDAR perception → body frame, cropped to the room.

The Go2 carries a built-in **L1 4D dome LiDAR** on its head and runs its OWN
LiDAR-SLAM, so — unlike the G1 — we don't need an off-board SLAM to perceive or
navigate: the point cloud, a robot-frame height map, and an odom/pose estimate
arrive on ``rt/utlidar/*`` (and ``rt/uslam/*``). This module turns that cloud into
clean body-frame geometry for the :class:`~navigation.Navigator`.

ROOM CROP — why it matters here. The dome LiDAR sees straight through an **open
door**, so a raw scan taken in a room with a doorway is polluted by returns from
the corridor/next room beyond it. Those far, sparse points wreck an occupancy grid
(phantom free space and stray obstacles outside the walls). So the headline
operation is :func:`crop_to_room`: clip the cloud to the room's rectangular
footprint before any mapping. Set the bounds to the wall lines; give the door side
the wall's coordinate, not the far return's.

The geometry (:func:`crop_to_room`, :func:`drop_self_returns`,
:func:`room_occupancy`) is pure numpy and unit-tested with synthetic clouds; the
DDS acquisition (:class:`Go2LidarSight`) is a thin layer on top.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"
# Dome housing self-returns cluster at the sensor origin; drop inside this radius.
SELF_RETURN_RADIUS_M = 0.10


@dataclass
class RoomBounds:
    """Axis-aligned room footprint in the frame the cloud is expressed in (metres).

    ``+x`` forward, ``+y`` left (REP-103 body frame) unless the cloud has been
    transformed into a world/odom frame. The ``door_note`` records which edge is an
    open doorway — the edge whose bound must be the WALL line so scans don't leak
    through it (see the module docstring).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = -0.10
    z_max: float = 2.20
    door_note: str = ""

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)


def crop_to_room(cloud: np.ndarray, bounds: RoomBounds) -> np.ndarray:
    """Keep only the points inside ``bounds`` — clips scans that leak past a doorway.

    ``cloud`` is ``(N, 3)`` XYZ. Returns the ``(M, 3)`` subset within the box.
    """
    cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
    if cloud.size == 0:
        return cloud
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    inside = ((x >= bounds.x_min) & (x <= bounds.x_max) &
              (y >= bounds.y_min) & (y <= bounds.y_max) &
              (z >= bounds.z_min) & (z <= bounds.z_max))
    return cloud[inside]


def drop_self_returns(cloud: np.ndarray, radius_m: float = SELF_RETURN_RADIUS_M) -> np.ndarray:
    """Drop points within ``radius_m`` of the sensor origin (dome self-returns)."""
    cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
    if cloud.size == 0:
        return cloud
    r2 = cloud[:, 0] ** 2 + cloud[:, 1] ** 2 + cloud[:, 2] ** 2
    return cloud[r2 > radius_m * radius_m]


def room_occupancy(cloud: np.ndarray, bounds: RoomBounds, cell_m: float = 0.05,
                   z_floor: float = 0.08, z_ceiling: float = 1.8) -> np.ndarray:
    """Top-down occupancy grid over ``bounds`` from a body/world-frame cloud.

    Points in the height band ``[z_floor, z_ceiling]`` mark obstacle cells (the
    floor and ceiling are excluded so they don't fill the grid). Returns a boolean
    array indexed ``[ix, iy]`` (True = occupied). For A* planning, feed the cropped
    cloud to ``arm_mhs_robotkit.navigation`` instead — this grid is for a quick top-down view.
    """
    cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
    nx = max(1, int(math.ceil((bounds.x_max - bounds.x_min) / cell_m)))
    ny = max(1, int(math.ceil((bounds.y_max - bounds.y_min) / cell_m)))
    grid = np.zeros((nx, ny), dtype=bool)
    if cloud.size == 0:
        return grid
    band = cloud[(cloud[:, 2] >= z_floor) & (cloud[:, 2] <= z_ceiling)]
    if band.size == 0:
        return grid
    ix = np.floor((band[:, 0] - bounds.x_min) / cell_m).astype(int)
    iy = np.floor((band[:, 1] - bounds.y_min) / cell_m).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[ix[ok], iy[ok]] = True
    return grid


PF_FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Decode a ``sensor_msgs/PointCloud2``-shaped message into an ``(N, 3)`` array.

    Reads the ``x``/``y``/``z`` fields by their declared offsets, honouring
    ``is_bigendian``, so it works for any point step / extra fields (intensity, ring,
    time). Requires ``x``/``y``/``z`` to be ``FLOAT32`` (the Go2 cloud is) and raises
    on any other datatype rather than silently misreading. Kept dependency-light (no
    ros2 ``point_cloud2`` helper) so it runs from the SDK env.
    """
    fields = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"} <= set(fields):
        raise ValueError("PointCloud2 has no x/y/z fields")
    for axis in ("x", "y", "z"):
        dtype = getattr(fields[axis], "datatype", PF_FLOAT32)
        if dtype != PF_FLOAT32:
            raise ValueError(f"PointCloud2 field {axis!r} is datatype {dtype}, not FLOAT32 — "
                             f"pointcloud2_to_xyz only decodes float32 x/y/z")
    f32 = np.dtype(">f4") if getattr(msg, "is_bigendian", False) else np.dtype("<f4")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(msg.point_step)
    n = len(raw) // step if step else 0
    if n == 0:
        return np.empty((0, 3), dtype=float)
    raw = raw[: n * step].reshape(n, step)
    out = np.empty((n, 3), dtype=np.float32)
    for i, axis in enumerate(("x", "y", "z")):
        off = int(fields[axis].offset)
        out[:, i] = raw[:, off:off + 4].copy().view(f32).reshape(-1)
    finite = np.isfinite(out).all(axis=1)
    return out[finite].astype(float)


class Go2LidarSight:
    """Acquire the Go2's onboard LiDAR cloud and return clean body-frame geometry.

    ``cloud_type`` injects the ``PointCloud2`` IDL (from unitree_sdk2py / a Go2 ROS2
    bridge); by default it is resolved lazily and, if unavailable, :meth:`connect`
    explains what to pass. The geometry functions above work on any ``(N, 3)`` cloud
    regardless of how it was acquired.
    """

    def __init__(self, iface: str = "eth0", bounds: RoomBounds | None = None,
                 cloud_topic: str = CLOUD_TOPIC, init_dds: bool = True,
                 cloud_type=None) -> None:
        self._iface = iface
        self._bounds = bounds
        self._cloud_topic = cloud_topic
        self._init_dds = init_dds
        self._cloud_type = cloud_type
        self._sub = None
        self._lock = threading.Lock()
        self._raw = None

    def connect(self) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

        cloud_type = self._cloud_type or _resolve_pointcloud2()
        if cloud_type is None:
            raise RuntimeError(
                "no PointCloud2 IDL found — pass cloud_type=<PointCloud2_ class> from "
                "unitree_sdk2py or your Go2 ROS2 message package.")
        self._cloud_type = cloud_type

        if self._init_dds:
            ChannelFactoryInitialize(0, self._iface)
        self._sub = ChannelSubscriber(self._cloud_topic, cloud_type)
        self._sub.Init(self._on_cloud, 5)

        deadline = time.monotonic() + 3.0
        while self._raw is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._raw is None:
            self.shutdown()
            raise RuntimeError(f"no {self._cloud_topic} in 3 s — is the LiDAR service up?")

    def _on_cloud(self, msg) -> None:
        with self._lock:
            self._raw = msg

    def get_cloud(self) -> np.ndarray:
        """The latest raw cloud as ``(N, 3)`` XYZ (no crop)."""
        with self._lock:
            raw = self._raw
        return pointcloud2_to_xyz(raw) if raw is not None else np.empty((0, 3))

    def cropped_cloud(self) -> np.ndarray:
        """The latest cloud, self-returns dropped and cropped to the room bounds.

        Requires ``bounds`` — without a room footprint we can't clip the doorway
        leak, so this raises rather than return a polluted cloud.
        """
        if self._bounds is None:
            raise RuntimeError("set RoomBounds to crop the doorway leak (see module docstring)")
        return crop_to_room(drop_self_returns(self.get_cloud()), self._bounds)

    def shutdown(self) -> None:
        if self._sub is not None:
            try:
                self._sub.Close()
            except Exception:
                pass
            self._sub = None


def _resolve_pointcloud2():
    for modpath, name in (("unitree_sdk2py.idl.sensor_msgs.msg.dds_", "PointCloud2_"),
                          ("sensor_msgs.msg.dds_", "PointCloud2_")):
        try:
            return getattr(importlib.import_module(modpath), name)
        except Exception:
            pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 LiDAR read-only diagnostics (no motion).")
    ap.add_argument("--iface", default="eth0", help="DDS network interface")
    ap.add_argument("--room", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="room bounds (m) to crop the doorway leak, e.g. --room -2 3 -2 2")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    bounds = RoomBounds(*args.room) if args.room else None
    sight = Go2LidarSight(iface=args.iface, bounds=bounds)
    sight.connect()
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            raw = sight.get_cloud()
            if bounds is not None:
                kept = sight.cropped_cloud()
                print(f"  raw={len(raw):6d} pts   after room-crop={len(kept):6d} pts "
                      f"({len(raw) - len(kept)} clipped, incl. doorway leak)")
            else:
                print(f"  raw={len(raw):6d} pts   (pass --room to crop the doorway leak)")
            time.sleep(0.5)
    finally:
        sight.shutdown()


if __name__ == "__main__":
    main()
