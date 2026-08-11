# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture Go2 LiDAR reference scans into ``images/`` (run ON the robot).

Saves three PNGs that become the descriptor's LiDAR ``reference_media`` — the
"what a calibrated Go2 scan looks like" standard:

  * ``scene_topdown.jpg`` — top-down (XY) scatter, RAW vs room-CROPPED side by side,
    so the doorway leak (points beyond the walls) is visible and shown removed.
  * ``lidar_near_xz.jpg``  — side (XZ) scatter of the cropped cloud (floor→ceiling).
  * ``range_hist.jpg``     — histogram of point ranges, raw vs cropped.

Read-only: subscribes to the cloud, never commands motion. Needs ``matplotlib`` on
the robot (``pip install matplotlib``). Crop to the room first — a scan that leaks
through an open door is not a useful reference (see the module README).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import go2_lidar_sight as gls


def _scatter(ax, cloud, title, lim=None):
    if len(cloud):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.5, c=cloud[:, 2], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x fwd (m)")
    ax.set_ylabel("y left (m)")
    ax.set_aspect("equal", "box")
    if lim:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture Go2 LiDAR reference scans (read-only).")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--room", type=float, nargs=4, required=True,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="room bounds (m) — the doorway edge should be the WALL line")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "images"))
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to let the cloud settle")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    room = gls.RoomBounds(*args.room, door_note="captured scene")

    sight = gls.Go2LidarSight(iface=args.iface, bounds=room)
    sight.connect()
    try:
        time.sleep(args.settle)
        raw = sight.get_cloud()
        cropped = gls.crop_to_room(gls.drop_self_returns(raw), room)
        print(f"raw={len(raw)} pts   cropped={len(cropped)} pts   "
              f"({len(raw) - len(cropped)} clipped, incl. doorway leak)")

        lim = (room.x_min - 1, room.x_max + 1, room.y_min - 1, room.y_max + 1)

        # 1) top-down: raw vs cropped
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5))
        _scatter(a0, raw, f"RAW ({len(raw)} pts) — leaks past the door", lim)
        for edge in (room.x_min, room.x_max):
            a0.axvline(edge, color="r", lw=0.8, ls="--")
        for edge in (room.y_min, room.y_max):
            a0.axhline(edge, color="r", lw=0.8, ls="--")
        _scatter(a1, cropped, f"ROOM-CROPPED ({len(cropped)} pts)", lim)
        fig.tight_layout()
        fig.savefig(outdir / "scene_topdown.jpg", dpi=110)
        plt.close(fig)

        # 2) side view (XZ) of the cropped cloud
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(cropped):
            ax.scatter(cropped[:, 0], cropped[:, 2], s=0.5, c=cropped[:, 2], cmap="viridis")
        ax.set_title("cropped cloud — side (XZ)")
        ax.set_xlabel("x fwd (m)")
        ax.set_ylabel("z up (m)")
        fig.tight_layout()
        fig.savefig(outdir / "lidar_near_xz.jpg", dpi=110)
        plt.close(fig)

        # 3) range histogram, raw vs cropped
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(raw):
            ax.hist(np.linalg.norm(raw, axis=1), bins=80, alpha=0.5, label="raw")
        if len(cropped):
            ax.hist(np.linalg.norm(cropped, axis=1), bins=80, alpha=0.7, label="room-cropped")
        ax.set_title("point range histogram")
        ax.set_xlabel("range (m)")
        ax.set_ylabel("points")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "range_hist.jpg", dpi=110)
        plt.close(fig)

        print(f"wrote scene_topdown.jpg, lidar_near_xz.jpg, range_hist.jpg to {outdir}")
    finally:
        sight.shutdown()


if __name__ == "__main__":
    main()
