#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Go2 LiDAR geometry — room crop, self-returns, occupancy.

Synthetic clouds only; no DDS, no robot. The headline case is a "doorway leak":
points outside the room bounds that :func:`crop_to_room` must remove.

Run: ``python3 test_go2_lidar_sight.py``
"""
from __future__ import annotations

import os
import struct
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_lidar_sight import (  # noqa: E402
    RoomBounds, crop_to_room, drop_self_returns, room_occupancy, pointcloud2_to_xyz,
)

ROOM = RoomBounds(x_min=-2.0, x_max=3.0, y_min=-2.0, y_max=2.0, door_note="south (+x) doorway")


def test_crop_removes_doorway_leak():
    # 4 points inside the room, plus a cluster 5 m past the +x wall (through the door)
    inside = np.array([[0, 0, 0.5], [2.5, 1.5, 1.0], [-1.5, -1.5, 0.2], [1.0, 0.0, 0.9]])
    leak = np.array([[8.0, 0.0, 0.5], [10.0, -1.0, 0.3]])  # corridor beyond the door
    cloud = np.vstack([inside, leak])
    kept = crop_to_room(cloud, ROOM)
    assert len(kept) == len(inside), f"expected {len(inside)} kept, got {len(kept)}"
    assert kept[:, 0].max() <= ROOM.x_max


def test_crop_respects_z_band():
    cloud = np.array([[0, 0, 0.5], [0, 0, 5.0], [0, 0, -1.0]])  # one in band, two out
    assert len(crop_to_room(cloud, ROOM)) == 1


def test_crop_empty_cloud():
    assert crop_to_room(np.empty((0, 3)), ROOM).shape == (0, 3)


def test_drop_self_returns():
    cloud = np.array([[0.01, 0.0, 0.0], [0.05, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kept = drop_self_returns(cloud, radius_m=0.10)
    assert len(kept) == 1 and abs(kept[0, 0] - 0.5) < 1e-9


def test_room_occupancy_marks_walls_not_floor():
    # a wall of points at x=2.9 (in band) + a sheet of floor points (z≈0)
    wall = np.array([[2.9, y, 1.0] for y in np.linspace(-1.5, 1.5, 20)])
    floor = np.array([[x, y, 0.0] for x in np.linspace(-1, 2, 10) for y in np.linspace(-1, 1, 10)])
    grid = room_occupancy(np.vstack([wall, floor]), ROOM, cell_m=0.1)
    assert grid.any(), "wall points should occupy cells"
    # the floor (z≈0 < z_floor=0.08) must not fill the grid
    assert grid.sum() < 40, "floor points must be excluded from occupancy"


def test_pointcloud2_to_xyz_decodes_offsets():
    # 2 points, point_step 16 (x,y,z float32 + 4 pad bytes), x@0 y@4 z@8
    pts = [(1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)]
    data = b"".join(struct.pack("<fffi", x, y, z, 0) for x, y, z in pts)
    fields = [SimpleNamespace(name=n, offset=o) for n, o in (("x", 0), ("y", 4), ("z", 8))]
    msg = SimpleNamespace(fields=fields, point_step=16, data=data)
    xyz = pointcloud2_to_xyz(msg)
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz, np.array(pts))


def test_pointcloud2_drops_nonfinite():
    data = struct.pack("<fffi", float("nan"), 0.0, 0.0, 0) + struct.pack("<fffi", 1.0, 1.0, 1.0, 0)
    fields = [SimpleNamespace(name=n, offset=o) for n, o in (("x", 0), ("y", 4), ("z", 8))]
    msg = SimpleNamespace(fields=fields, point_step=16, data=data)
    assert pointcloud2_to_xyz(msg).shape == (1, 3)  # the NaN point is dropped


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"go2_lidar_sight: {len(tests)}/{len(tests)} passed")
