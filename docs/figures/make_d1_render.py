#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The D1 arm, drawn from this repository's own URDF -- motors, joints, and the DOF.

The figure for Appendix B of the whitepaper. It replaces a Mermaid stick chain, which
named the joints but showed neither where the motors sit nor which way each one turns.

WHAT THIS IS, AND WHAT IT IS NOT. Every joint origin, `rpy`, axis and limit drawn here is
read from `robot-stack/unitree/go2/d1_arm/urdf/d1_description.urdf` at draw time, so no
number in the picture is retyped. That file is kinematics only -- it carries no `visual`,
`collision`, `inertial` or mesh element, because Unitree's D1 STL meshes are not
redistributed here and no vendor asset is fetched by this script. So the LINK GEOMETRY IS
PRIMITIVE: each link is a tube between two joint origins, and the barrel on each revolute
joint is drawn along that joint's own axis, which is what makes the degree of freedom
visible. It is not CAD, it is not a photograph, and it must not be captioned as either.
The pose is illustrative and every angle in it is inside the URDF's own `<limit>`.

Joint numbering is the one the code uses, because getting it wrong is an off-by-one on a
real arm: the URDF names are `Joint1..Joint7_2`, the wire and `d1_fk.COMMANDABLE_LIMITS_DEG`
count `J0..J5` from the base yaw, and `angle6` is the jaw and not part of the 6-joint
chain. Both appear on every callout.

Regenerate with:

    python3 docs/figures/make_d1_render.py

`--check` re-derives every joint origin, axis and limit from the URDF and exits non-zero
if any has moved from the value the committed PNG was drawn with, so a stale figure is a
failing command rather than a picture nobody re-ran.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
URDF = REPO / "robot-stack" / "unitree" / "go2" / "d1_arm" / "urdf" / "d1_description.urdf"
OUT = HERE / "d1-arm-render.png"

# Palette, taken verbatim from make_robot_profile.py so this figure sits beside the three
# robot profiles in the same document without changing register. Its comment there reads
# "One figure variant, dark, opaque, legible under both GitHub themes."
BG = "#161a1f"
TEXT = "#e8eef4"
MUTED = "#94a3b2"
ACCENT = "#6fc9eb"
PANEL = "#1f2630"
EDGE = "#38424e"
LINK = "#8b949e"
JOINT = "#6fc9eb"
GRIP = "#f0883e"

# The pose the committed PNG is drawn in: not the zero pose, which is a straight stick,
# and not an extreme. Every value is inside that joint's own URDF limit, asserted below.
POSE = {
    "Joint1": 0.55,
    "Joint2": -0.70,
    "Joint3": 0.95,
    "Joint4": 0.30,
    "Joint5": -0.55,
    "Joint6": 0.40,
    "Joint7_1": 0.022,
    "Joint7_2": -0.022,
}

# Firmware index, human name, and the callout's offset in points from the projected joint
# origin. The firmware counts J0..J5 from the base yaw while the URDF counts Joint1..6, so
# every callout carries both -- see the module docstring.
CALLOUTS = {
    "Joint1": ("J0", "base yaw", 135.0, (-190, -78)),
    "Joint2": ("J1", "shoulder", 90.0, (-215, -4)),
    "Joint3": ("J2", "elbow", 90.0, (-190, 62)),
    "Joint4": ("J3", "elbow roll", 135.0, (120, 96)),
    "Joint5": ("J4", "wrist pitch", 90.0, (205, 62)),
    "Joint6": ("J5", "wrist roll", 135.0, (215, -34)),
}

# What the committed PNG was drawn with. --check re-derives these from the URDF and fails
# on any disagreement; the drawing reads the URDF too, so the picture and this table
# cannot drift apart without the check saying so.
EXPECTED = {
    "Joint1": ("revolute", (0.0, 0.0, 0.0533), (0.0, 0.0, 1.0), (-2.35, 2.35)),
    "Joint2": ("revolute", (0.0, 0.028, 0.0563), (0.0, 0.0, -1.0), (-1.57, 1.57)),
    "Joint3": ("revolute", (0.0, 0.2693, 0.0009), (0.0, 0.0, -1.0), (-1.57, 1.57)),
    "Joint4": ("revolute", (0.0577, 0.042, -0.0275), (0.0, 0.0, 1.0), (-2.35, 2.35)),
    "Joint5": ("revolute", (-0.0001, -0.0237, 0.14018), (0.0, 0.0, -1.0), (-1.57, 1.57)),
    "Joint6": ("revolute", (0.0825, -0.0010782, -0.023822), (0.0, 0.0, -1.0), (-2.35, 2.35)),
    "Joint7_1": ("prismatic", (-0.0056012, -0.029636, 0.0706), (0.0, 0.0, -1.0), (0.0, 0.03)),
    "Joint7_2": ("prismatic", (-0.0056388, 0.02964, 0.0706), (0.0, 0.0, 1.0), (-0.03, 0.0)),
}

# The three numbers Appendix B is about, and where each is checkable. The line count is
# 1,114 lines of robot-stack/unitree/go2/d1_arm/ plus 297 arm-specific lines of
# visual_nav/safety.py, which is the appendix's own table.
FOOTER = "3.15 kg on the hind legs  ·  70 °C abort / 55 °C warn  ·  ~1,411 lines of guard code"


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, applied Z then Y then X."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues, about an axis that the URDF does not promise is normalised."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def read_joints(urdf: Path) -> list[dict]:
    joints = []
    for j in ET.parse(urdf).getroot().findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": _vec(origin, "xyz"),
            "rpy": _vec(origin, "rpy"),
            "axis": _vec(axis, "xyz", "0 0 1"),
            "lower": float(limit.get("lower", 0.0)) if limit is not None else 0.0,
            "upper": float(limit.get("upper", 0.0)) if limit is not None else 0.0,
        })
    return joints


def _vec(node, attr: str, default: str = "0 0 0") -> np.ndarray:
    raw = default if node is None else node.get(attr, default)
    return np.array([float(v) for v in raw.split()])


def forward_kinematics(joints: list[dict]) -> dict[str, np.ndarray]:
    """Walk the URDF's parent/child tree, not the file order.

    The last two joints BOTH hang off `Link6` -- they are the two jaw fingers, not a chain
    of one behind the other. Accumulating transforms down the joint list instead of down
    the tree puts the second finger on the end of the first, which is a picture of an arm
    this repository does not describe.
    """
    by_parent: dict[str, list[dict]] = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    frames = {"base_link": np.eye(4)}
    stack = ["base_link"]
    while stack:
        link = stack.pop()
        for j in by_parent.get(link, []):
            local = np.eye(4)
            local[:3, :3] = rpy_matrix(*j["rpy"])
            local[:3, 3] = j["xyz"]
            t = frames[link] @ local
            q = POSE.get(j["name"], 0.0)
            move = np.eye(4)
            if j["type"] == "revolute":
                move[:3, :3] = axis_rotation(j["axis"], q)
            else:
                move[:3, 3] = j["axis"] / np.linalg.norm(j["axis"]) * q
            frames[j["child"]] = t @ move
            stack.append(j["child"])
    return frames


def tube(p0, p1, radius: float, facets: int = 24) -> list:
    """A closed cylinder between two points. The URDF has no mesh; this is the stand-in."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return []
    d = d / length
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    th = np.linspace(0, 2 * np.pi, facets)
    ring0 = p0 + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    ring1 = p1 + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    quads = [[ring0[i], ring0[(i + 1) % facets], ring1[(i + 1) % facets], ring1[i]]
             for i in range(facets)]
    quads.append(list(ring0))
    quads.append(list(ring1))
    return quads


def check(joints: list[dict]) -> int:
    """Re-derive the figure's numbers from the URDF. Non-zero means the PNG is stale."""
    seen = {j["name"]: j for j in joints}
    status = 0
    if set(seen) != set(EXPECTED):
        print(f"  FAIL  the URDF's joint set changed: {sorted(set(seen) ^ set(EXPECTED))}")
        return 1
    for name, (kind, xyz, axis, limits) in EXPECTED.items():
        j = seen[name]
        bad = []
        if j["type"] != kind:
            bad.append(f"type {j['type']}, drawn as {kind}")
        if not np.allclose(j["xyz"], xyz, atol=1e-9):
            bad.append(f"origin {np.round(j['xyz'], 7)}, drawn at {xyz}")
        if not np.allclose(j["axis"], axis, atol=1e-9):
            bad.append(f"axis {j['axis']}, drawn as {axis}")
        if not np.allclose((j["lower"], j["upper"]), limits, atol=1e-9):
            bad.append(f"limits ({j['lower']}, {j['upper']}), drawn as {limits}")
        if bad:
            status = 1
            print(f"  FAIL  {name:<9} {'; '.join(bad)}")
    for name, q in POSE.items():
        j = seen[name]
        if not j["lower"] - 1e-9 <= q <= j["upper"] + 1e-9:
            status = 1
            print(f"  FAIL  {name:<9} pose {q:.3f} is outside "
                  f"[{j['lower']}, {j['upper']}]")
    if status == 0:
        print("  ok  8 joints: origin, axis and limit unchanged, and the pose is inside every one")
    return status


def axis_label(axis: np.ndarray) -> str:
    letter = "XYZ"[int(np.argmax(np.abs(axis)))]
    sign = "+" if axis[int(np.argmax(np.abs(axis)))] > 0 else "-"
    return f"{sign}{letter}"


def draw(joints: list[dict], out: Path) -> None:
    frames = forward_kinematics(joints)
    by_name = {j["name"]: frames[j["child"]] for j in joints}
    by_name["base_link"] = frames["base_link"]
    pts = np.array([t[:3, 3] for t in frames.values()])
    light = LightSource(azdeg=225, altdeg=48)

    fig = plt.figure(figsize=(19.2, 8.6), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    # One tube per joint, from its parent link's origin to its child's: the jaw therefore
    # branches at Link6 in the picture the way it branches in the file.
    for j in joints:
        finger = j["type"] == "prismatic"
        ax.add_collection3d(Poly3DCollection(
            tube(frames[j["parent"]][:3, 3], frames[j["child"]][:3, 3],
                 0.012 if finger else 0.019),
            facecolors=GRIP if finger else LINK, shade=True, lightsource=light))

    # A short barrel on each revolute joint, along that joint's own axis. This is the only
    # part of the picture that shows a degree of freedom rather than a position.
    for j in joints:
        if j["type"] != "revolute":
            continue
        t = frames[j["child"]]
        a = t[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
        c = t[:3, 3]
        ax.add_collection3d(Poly3DCollection(
            tube(c - a * 0.021, c + a * 0.021, 0.029), facecolors=JOINT, shade=True,
            lightsource=light))

    ax.set_box_aspect((1, 1, 0.78))
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-64)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    span = (hi - lo).max() * 0.34
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)

    fig.canvas.draw()
    by_urdf_name = {j["name"]: j for j in joints}
    for name, (wire, human, commandable_deg, offset) in CALLOUTS.items():
        j = by_urdf_name[name]
        lines = [
            f"revolute · axis {axis_label(j['axis'])} · ±{j['upper']:.2f} rad mechanical",
            f"commandable ±{commandable_deg:g}° (d1_fk)",
        ]
        if name == "Joint3":
            lines.append(f"+{j['xyz'][1]:.4f} m — the long link")
        _annotate(ax, by_name[name], f"{name} · {wire}  {human}", lines, offset)

    jaw = by_urdf_name["Joint7_1"]
    _annotate(
        ax, by_name["Joint7_1"], "Joint7_1 / Joint7_2 · jaw",
        [f"2 fingers · PRISMATIC · axis {axis_label(jaw['axis'])} · 0 to {jaw['upper']:.2f} m",
         "the only non-revolute pair; angle6 on the wire,",
         "and not part of the 6-joint chain"],
        (130, -145))

    fig.text(0.010, 0.985, "Unitree D1 arm — 6 revolute joints + a 2-finger prismatic jaw",
             color=TEXT, fontsize=17, weight="bold", va="top", family="DejaVu Sans")
    fig.text(0.010, 0.938,
             "Rendered from this repository's own d1_description.urdf. The URDF carries no "
             "meshes, so link geometry is\nprimitive tubes and the barrels are drawn along "
             "each joint's real axis — but every origin, axis and limit\nbelow is the "
             "file's. Not CAD. The pose is illustrative and inside every limit.",
             color=MUTED, fontsize=10.5, linespacing=1.55, va="top", family="DejaVu Sans")
    fig.text(0.010, 0.028, FOOTER, color=GRIP, fontsize=11.5, weight="bold", va="bottom",
             family="DejaVu Sans")

    fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight", pad_inches=0.28)


def _annotate(ax, transform: np.ndarray, title: str, lines: list[str], offset) -> None:
    x, y, _ = proj3d.proj_transform(*transform[:3, 3], ax.get_proj())
    head, tail = title.split("  ", 1) if "  " in title else (title, "")
    bold = head.replace(" ", "\\ ").replace("_", "\\_")
    body = "\n".join(lines)
    ax.annotate(
        f"$\\bf{{{bold}}}$  {tail}\n{body}",
        xy=(x, y), xycoords="data", xytext=offset, textcoords="offset points",
        # A 3D projection puts some joints outside the 2D axes box, and matplotlib's
        # default is to silently DROP a data-anchored annotation that lands there. The
        # elbow callout disappeared exactly that way, with no warning and no error.
        annotation_clip=False,
        color=TEXT, fontsize=10.5, linespacing=1.45, family="DejaVu Sans",
        ha="left" if offset[0] > 0 else "right", va="center",
        bbox={"boxstyle": "round,pad=0.5", "fc": PANEL, "ec": EDGE, "lw": 1.0, "alpha": 0.95},
        arrowprops={"arrowstyle": "-", "color": ACCENT, "lw": 1.2, "shrinkA": 6,
                    "shrinkB": 9, "alpha": 0.85, "connectionstyle": "arc3,rad=0.12"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true",
                        help="re-derive the joint table from the URDF; do not draw")
    args = parser.parse_args()

    joints = read_joints(args.urdf)
    if args.check:
        return check(joints)
    if check(joints) != 0:
        print("refusing to draw a figure the URDF no longer supports", file=sys.stderr)
        return 1
    draw(joints, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
