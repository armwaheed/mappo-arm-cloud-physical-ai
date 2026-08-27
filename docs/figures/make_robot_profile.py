#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an annotated profile figure of a quadruped from its vendor URDF.

NO VENDOR GEOMETRY IS COMMITTED TO THIS REPOSITORY. There is no URDF, no STL and no
Collada file anywhere under this tree, and this script will not put one here. It reads
the vendor descriptions out of a checkout you make yourself, somewhere outside the
repository, and writes only a PNG. That is a licensing requirement, not a preference:
the meshes are BSD-3-Clause works of their vendors and are theirs to distribute.

FETCH THE MESHES FIRST. Pick any scratch directory -- $ASSETS below -- outside this
repository, and run exactly these commands:

    mkdir -p "$ASSETS" && cd "$ASSETS"

    # Unitree Go2 and Go2-W. BSD-3-Clause,
    # (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
    # A blobless sparse clone: the two descriptions are ~110 MB of .dae, the whole
    # repository is several times that.
    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/unitreerobotics/unitree_ros.git
    cd unitree_ros
    git sparse-checkout set robots/go2_description robots/go2w_description
    git sparse-checkout add LICENSE      # sparse mode excludes it, and it is the
    cd ..                                # only place the copyright line is stated

    # Deep Robotics Lite3. BSD-3-Clause, (c) 2024, DeepRoboticsLab.
    # .gitattributes is present but empty -- nothing here is git-lfs, a plain clone
    # gets real STL bytes rather than pointer files.
    git clone --depth 1 https://github.com/DeepRoboticsLab/deep_robotics_model.git

THEN RENDER. One command per figure; each writes one PNG and touches nothing else:

    python3 make_robot_profile.py --preset go2 \\
        --urdf "$ASSETS/unitree_ros/robots/go2_description/urdf/go2_description.urdf" \\
        --out go2-walk-profile.png

    python3 make_robot_profile.py --preset go2w \\
        --urdf "$ASSETS/unitree_ros/robots/go2w_description/urdf/go2w_description.urdf" \\
        --out go2-wheel-profile.png

    python3 make_robot_profile.py --preset lite3 \\
        --urdf "$ASSETS/deep_robotics_model/Lite3/urdf/Lite3.urdf" \\
        --out lite3-profile.png

WHAT IT IS. A software rasteriser in numpy, about as much renderer as a figure needs
and no more: no OpenGL, no matplotlib, no trimesh, no scene graph library. It parses
the URDF with xml.etree, walks the kinematic tree, loads each link's visual mesh
(binary STL, ascii STL, and Collada .dae including the visual-scene node transforms
that Unitree's exporter leaves the geometry sitting under), projects through a
perspective camera, z-buffers, and flat-shades with one key light, a cool fill and a
rim term. It then draws callout labels whose leader lines are anchored to PROJECTED
LINK POSITIONS -- a callout names a link and an offset in that link's own frame, and
the arrow lands wherever the camera puts it. No hand-guessed pixel coordinates, so
changing --azimuth does not silently detach every label from what it points at.

THE POSE IS NOT THE ZERO POSE, AND SAYING SO MATTERS. --zero-pose walks the tree with
every joint at zero, which is what the URDF describes and what this script defaults to
with no preset. It is also a pose neither robot can hold: Go2's calf joint is limited
to [-2.7227, -0.8378] and Lite3's knee to [0.524, 2.792], so zero is OUTSIDE the limit
on eight of twelve joints between them, and the render is a straight-legged animal
standing on stilts. Each preset therefore carries a stance -- a set of joint angles,
printed by --print-pose, every one of them inside the vendor's own <limit> -- and the
figures in this directory are rendered in that stance, not at zero.

Colours come from the mesh where the mesh has them (the .dae files carry per-material
lambert diffuse, which is why the Go2 renders light grey with black trim) and from the
URDF <material> otherwise; near-black vendor materials are lifted off the background
floor so they read as geometry rather than as a hole. The background is opaque
#161A1F with light text, one variant per robot, so the figure reads the same in
GitHub's light and dark themes.

Lint: cd docs/figures && ruff check .
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------------
# Palette. One figure variant, dark, opaque, legible under both GitHub themes.
# ---------------------------------------------------------------------------------

BG = (0x16, 0x1A, 0x1F)
BG_GLOW = (0x24, 0x2B, 0x34)
GROUND = (0x0D, 0x10, 0x14)
TEXT = (0xE8, 0xEE, 0xF4)
MUTED = (0x94, 0xA3, 0xB2)
DIM = (0x69, 0x76, 0x84)
ACCENT = (0x6F, 0xC9, 0xEB)

FONT_CANDIDATES = (
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


# ---------------------------------------------------------------------------------
# Small linear algebra. Everything is a 4x4 homogeneous transform in metres.
# ---------------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, i.e. Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def make_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues, returned as a 4x4 so it composes with the joint origins."""
    a = normalize(np.asarray(axis, dtype=np.float64))
    if not np.isfinite(a).all() or float(np.linalg.norm(a)) < 1e-9:
        return np.eye(4)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    r = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    out = np.eye(4)
    out[:3, :3] = r
    return out


def parse_floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * count
    vals = [float(x) for x in text.replace(",", " ").split()]
    while len(vals) < count:
        vals.append(default)
    return vals[:count]


# ---------------------------------------------------------------------------------
# Mesh container and loaders.
# ---------------------------------------------------------------------------------


@dataclass
class Mesh:
    """Triangle soup with one flat colour per triangle."""

    verts: np.ndarray
    tris: np.ndarray
    colors: np.ndarray

    @staticmethod
    def empty() -> Mesh:
        return Mesh(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    def is_empty(self) -> bool:
        return self.tris.shape[0] == 0


def merge_meshes(parts: Sequence[Mesh]) -> Mesh:
    parts = [p for p in parts if not p.is_empty()]
    if not parts:
        return Mesh.empty()
    verts, tris, colors, base = [], [], [], 0
    for p in parts:
        verts.append(p.verts)
        tris.append(p.tris + base)
        colors.append(p.colors)
        base += p.verts.shape[0]
    return Mesh(np.vstack(verts), np.vstack(tris), np.vstack(colors))


def load_stl(path: Path) -> Mesh:
    """Binary or ascii STL. The format is told apart by arithmetic, not by the word
    `solid`: several vendor exporters write a binary file whose 80-byte header starts
    with an ascii banner (go2w's calf.stl begins `COLOR=`), so sniffing the first five
    characters gets it wrong."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * count == len(data) and count > 0:
            dt = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
            rec = np.frombuffer(data, dtype=dt, count=count, offset=84)
            verts = rec["v"].reshape(-1, 3).astype(np.float64)
            tris = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
            return Mesh(verts, tris, np.ones((count, 3), dtype=np.float32))
    text = data.decode("utf-8", errors="replace")
    nums = re.findall(
        r"vertex\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)", text
    )
    if not nums:
        raise ValueError(f"{path}: not a readable STL")
    verts = np.array(nums, dtype=np.float64)
    count = verts.shape[0] // 3
    verts = verts[: count * 3]
    tris = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
    return Mesh(verts, tris, np.ones((count, 3), dtype=np.float32))


# --- Collada -----------------------------------------------------------------------


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _child(el: ET.Element | None, name: str) -> ET.Element | None:
    if el is None:
        return None
    for c in el:
        if _tag(c) == name:
            return c
    return None


def _children(el: ET.Element | None, name: str) -> list[ET.Element]:
    if el is None:
        return []
    return [c for c in el if _tag(c) == name]


def _numbers(el: ET.Element | None, dtype: Any) -> np.ndarray:
    if el is None or not el.text:
        return np.zeros(0, dtype=dtype)
    return np.fromstring(el.text, dtype=dtype, sep=" ")


def _dae_effect_colors(root: ET.Element) -> dict[str, np.ndarray]:
    """effect id -> diffuse rgb. Only the COMMON profile, only a constant colour: the
    vendor .dae files are Blender lambert exports with no textures."""
    out: dict[str, np.ndarray] = {}
    for lib in _children(root, "library_effects"):
        for eff in _children(lib, "effect"):
            eid = eff.get("id")
            if not eid:
                continue
            tech = _child(_child(eff, "profile_COMMON"), "technique")
            if tech is None:
                continue
            for kind in ("lambert", "phong", "blinn", "constant"):
                shader = _child(tech, kind)
                col = _child(_child(shader, "diffuse"), "color")
                if col is None:
                    col = _child(_child(shader, "emission"), "color")
                if col is not None and col.text:
                    vals = parse_floats(col.text, 4, 1.0)
                    out[eid] = np.array(vals[:3], dtype=np.float32)
                    break
    return out


def _dae_material_colors(root: ET.Element) -> dict[str, np.ndarray]:
    effects = _dae_effect_colors(root)
    out: dict[str, np.ndarray] = {}
    for lib in _children(root, "library_materials"):
        for mat in _children(lib, "material"):
            mid = mat.get("id")
            inst = _child(mat, "instance_effect")
            if mid is None or inst is None:
                continue
            url = (inst.get("url") or "").lstrip("#")
            if url in effects:
                out[mid] = effects[url]
    return out


def _dae_sources(mesh_el: ET.Element) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for src in _children(mesh_el, "source"):
        sid = src.get("id")
        arr = _child(src, "float_array")
        if sid is None or arr is None:
            continue
        vals = _numbers(arr, np.float64)
        acc = _child(_child(src, "technique_common"), "accessor")
        stride = int(acc.get("stride", "3")) if acc is not None else 3
        stride = max(stride, 1)
        usable = (vals.shape[0] // stride) * stride
        out[sid] = vals[:usable].reshape(-1, stride)
    return out


def _dae_primitive_indices(prim: ET.Element) -> tuple[str | None, np.ndarray]:
    """Return (VERTEX source id, (M,3) triangle index array) for one <triangles>,
    <polylist> or <polygons> element, fan-triangulating where it has to."""
    vertex_src, vertex_off, stride = None, 0, 1
    for inp in _children(prim, "input"):
        off = int(inp.get("offset", "0"))
        stride = max(stride, off + 1)
        if inp.get("semantic") == "VERTEX":
            vertex_src = (inp.get("source") or "").lstrip("#")
            vertex_off = off
    p_el = _child(prim, "p")
    if p_el is None:
        p_els = _children(prim, "p")
        if not p_els:
            return vertex_src, np.zeros((0, 3), dtype=np.int64)
        idx_parts = []
        for pe in p_els:
            face = _numbers(pe, np.int64)[vertex_off::stride]
            idx_parts.extend(
                [face[0], face[i], face[i + 1]] for i in range(1, face.shape[0] - 1)
            )
        return vertex_src, np.array(idx_parts, dtype=np.int64).reshape(-1, 3)
    flat = _numbers(p_el, np.int64)
    usable = (flat.shape[0] // stride) * stride
    idx = flat[:usable].reshape(-1, stride)[:, vertex_off]
    if _tag(prim) == "triangles":
        return vertex_src, idx[: (idx.shape[0] // 3) * 3].reshape(-1, 3)
    vcount = _numbers(_child(prim, "vcount"), np.int64)
    if vcount.shape[0] == 0:
        return vertex_src, idx[: (idx.shape[0] // 3) * 3].reshape(-1, 3)
    faces, cursor = [], 0
    for n in vcount.tolist():
        poly = idx[cursor : cursor + n]
        cursor += n
        faces.extend([poly[0], poly[i], poly[i + 1]] for i in range(1, n - 1))
    if not faces:
        return vertex_src, np.zeros((0, 3), dtype=np.int64)
    return vertex_src, np.array(faces, dtype=np.int64)


def _dae_geometries(
    root: ET.Element, mat_colors: dict[str, np.ndarray]
) -> dict[str, list[tuple[np.ndarray, np.ndarray, str | None]]]:
    """geometry id -> list of (positions, triangles, bound material symbol)."""
    geos: dict[str, list[tuple[np.ndarray, np.ndarray, str | None]]] = {}
    del mat_colors  # colours are resolved later, once bind_material is known
    for lib in _children(root, "library_geometries"):
        for geo in _children(lib, "geometry"):
            gid = geo.get("id")
            mesh_el = _child(geo, "mesh")
            if gid is None or mesh_el is None:
                continue
            sources = _dae_sources(mesh_el)
            vert_map: dict[str, str] = {}
            for v in _children(mesh_el, "vertices"):
                vid = v.get("id")
                for inp in _children(v, "input"):
                    if inp.get("semantic") == "POSITION" and vid is not None:
                        vert_map[vid] = (inp.get("source") or "").lstrip("#")
            parts: list[tuple[np.ndarray, np.ndarray, str | None]] = []
            for prim in mesh_el:
                if _tag(prim) not in ("triangles", "polylist", "polygons", "tristrips"):
                    continue
                vsrc, idx = _dae_primitive_indices(prim)
                if vsrc is None or idx.shape[0] == 0:
                    continue
                pos_id = vert_map.get(vsrc, vsrc)
                pos = sources.get(pos_id)
                if pos is None or pos.shape[0] == 0:
                    continue
                parts.append((pos[:, :3], idx, prim.get("material")))
            if parts:
                geos[gid] = parts
    return geos


def _dae_node_transform(node: ET.Element) -> np.ndarray:
    """Compose one node's own transform elements, in document order."""
    t = np.eye(4)
    for el in node:
        kind = _tag(el)
        if kind == "matrix":
            vals = _numbers(el, np.float64)
            if vals.shape[0] >= 16:
                t = t @ vals[:16].reshape(4, 4)
        elif kind == "translate":
            vals = parse_floats(el.text, 3)
            m = np.eye(4)
            m[:3, 3] = vals
            t = t @ m
        elif kind == "rotate":
            vals = parse_floats(el.text, 4)
            t = t @ axis_angle_to_matrix(np.array(vals[:3]), math.radians(vals[3]))
        elif kind == "scale":
            vals = parse_floats(el.text, 3, 1.0)
            m = np.eye(4)
            m[0, 0], m[1, 1], m[2, 2] = vals
            t = t @ m
    return t


def _dae_up_axis_fix(up: str) -> np.ndarray:
    if up == "Y_UP":
        return axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), math.pi / 2)
    if up == "X_UP":
        return axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    return np.eye(4)


def load_dae(path: Path) -> Mesh:
    """Collada, far enough for a vendor URDF: <float_array> positions, the <p> index
    stream under <triangles>/<polylist>, the <unit meter=...> scale, the asset up axis,
    the <visual_scene> node transforms, and the per-material lambert diffuse."""
    root = ET.parse(str(path)).getroot()
    asset = _child(root, "asset")
    unit_el = _child(asset, "unit")
    unit = float(unit_el.get("meter", "1")) if unit_el is not None else 1.0
    up_el = _child(asset, "up_axis")
    up = (up_el.text or "Z_UP").strip() if up_el is not None else "Z_UP"

    mat_colors = _dae_material_colors(root)
    geos = _dae_geometries(root, mat_colors)
    if not geos:
        return Mesh.empty()

    fix = _dae_up_axis_fix(up)
    fix[:3, :3] *= unit
    parts: list[Mesh] = []

    def emit(gid: str, world: np.ndarray, binding: dict[str, str]) -> None:
        for pos, idx, symbol in geos.get(gid, []):
            hom = np.hstack([pos, np.ones((pos.shape[0], 1))])
            verts = (hom @ world.T)[:, :3]
            target = binding.get(symbol or "", symbol or "")
            rgb = mat_colors.get(target)
            if rgb is None:
                rgb = np.array([0.72, 0.74, 0.78], dtype=np.float32)
            colors = np.tile(rgb.astype(np.float32), (idx.shape[0], 1))
            tris = idx
            if np.linalg.det(world[:3, :3]) < 0:
                tris = idx[:, ::-1]
            parts.append(Mesh(verts, tris.astype(np.int64), colors))

    def walk(node: ET.Element, parent: np.ndarray) -> None:
        world = parent @ _dae_node_transform(node)
        for inst in _children(node, "instance_geometry"):
            gid = (inst.get("url") or "").lstrip("#")
            binding: dict[str, str] = {}
            tech = _child(_child(inst, "bind_material"), "technique_common")
            for im in _children(tech, "instance_material"):
                sym = im.get("symbol")
                tgt = (im.get("target") or "").lstrip("#")
                if sym:
                    binding[sym] = tgt
            emit(gid, world, binding)
        for sub in _children(node, "node"):
            walk(sub, world)

    seen_before = len(parts)
    for lib in _children(root, "library_visual_scenes"):
        for scene in _children(lib, "visual_scene"):
            for node in _children(scene, "node"):
                walk(node, fix)
    if len(parts) == seen_before:
        for gid in geos:
            emit(gid, fix, {})
    return merge_meshes(parts)


MESH_CACHE: dict[str, Mesh] = {}


def load_mesh(path: Path) -> Mesh:
    key = str(path)
    if key not in MESH_CACHE:
        suffix = path.suffix.lower()
        if suffix == ".dae":
            MESH_CACHE[key] = load_dae(path)
        elif suffix == ".stl":
            MESH_CACHE[key] = load_stl(path)
        else:
            raise ValueError(f"{path}: unsupported mesh format")
    return MESH_CACHE[key]


# ---------------------------------------------------------------------------------
# URDF.
# ---------------------------------------------------------------------------------


@dataclass
class Visual:
    origin: np.ndarray
    mesh_path: Path | None = None
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    color: np.ndarray | None = None
    primitive: tuple[str, tuple[float, ...]] | None = None


@dataclass
class Link:
    name: str
    visuals: list[Visual] = field(default_factory=list)
    collisions: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: list[Joint]
    urdf_dir: Path


def resolve_mesh_path(raw: str, urdf_dir: Path) -> Path | None:
    """`package://pkg/rest`, `file://...` or a plain relative path. Packages are found
    by walking up from the URDF's own directory, which is how a vendor checkout is
    laid out; nothing is copied and nothing is searched outside that checkout."""
    text = raw.strip()
    if text.startswith("file://"):
        text = text[len("file://") :]
    if text.startswith("package://") or text.startswith("model://"):
        rest = text.split("://", 1)[1]
        pkg, _, tail = rest.partition("/")
        here = urdf_dir.resolve()
        for anc in [here, *here.parents]:
            if anc.name == pkg and (anc / tail).exists():
                return anc / tail
            cand = anc / pkg / tail
            if cand.exists():
                return cand
        return None
    cand = (urdf_dir / text).resolve()
    return cand if cand.exists() else None


def _parse_visual(el: ET.Element, urdf_dir: Path) -> Visual | None:
    org = _child(el, "origin")
    xyz = parse_floats(org.get("xyz") if org is not None else None, 3)
    rpy = parse_floats(org.get("rpy") if org is not None else None, 3)
    origin = make_transform(xyz, rpy)
    geom = _child(el, "geometry")
    if geom is None:
        return None
    color = None
    mat = _child(el, "material")
    col_el = _child(mat, "color")
    if col_el is not None:
        rgba = parse_floats(col_el.get("rgba"), 4, 1.0)
        color = np.array(rgba[:3], dtype=np.float32)
    mesh_el = _child(geom, "mesh")
    if mesh_el is not None:
        path = resolve_mesh_path(mesh_el.get("filename", ""), urdf_dir)
        scale = np.array(parse_floats(mesh_el.get("scale"), 3, 1.0), dtype=np.float64)
        if path is None:
            return None
        return Visual(origin=origin, mesh_path=path, scale=scale, color=color)
    for kind, attrs in (("box", ("size",)), ("cylinder", ("radius", "length")),
                        ("sphere", ("radius",))):
        prim_el = _child(geom, kind)
        if prim_el is None:
            continue
        vals: list[float] = []
        for a in attrs:
            vals.extend(parse_floats(prim_el.get(a), 3 if a == "size" else 1, 1.0))
        return Visual(origin=origin, color=color, primitive=(kind, tuple(vals)))
    return None


def parse_urdf(path: Path) -> Robot:
    root = ET.parse(str(path)).getroot()
    urdf_dir = path.parent
    links: dict[str, Link] = {}
    for le in _children(root, "link"):
        name = le.get("name")
        if not name:
            continue
        link = Link(name=name)
        for ve in _children(le, "visual"):
            vis = _parse_visual(ve, urdf_dir)
            if vis is not None:
                link.visuals.append(vis)
        for ce in _children(le, "collision"):
            vis = _parse_visual(ce, urdf_dir)
            if vis is not None:
                link.collisions.append(vis)
        links[name] = link
    joints: list[Joint] = []
    for je in _children(root, "joint"):
        parent = _child(je, "parent")
        child = _child(je, "child")
        if parent is None or child is None:
            continue
        org = _child(je, "origin")
        xyz = parse_floats(org.get("xyz") if org is not None else None, 3)
        rpy = parse_floats(org.get("rpy") if org is not None else None, 3)
        axis_el = _child(je, "axis")
        axis = np.array(
            parse_floats(axis_el.get("xyz") if axis_el is not None else None, 3), dtype=np.float64
        )
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.array([0.0, 0.0, 1.0])
        joints.append(
            Joint(
                name=je.get("name", ""),
                jtype=je.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin=make_transform(xyz, rpy),
                axis=axis,
            )
        )
    return Robot(name=root.get("name", path.stem), links=links, joints=joints, urdf_dir=urdf_dir)


def forward_kinematics(robot: Robot, pose: dict[str, float]) -> dict[str, np.ndarray]:
    """Link name -> 4x4 world transform. Unposed joints sit at zero, which is the
    URDF's own zero pose."""
    by_parent: dict[str, list[Joint]] = {}
    children = set()
    for j in robot.joints:
        by_parent.setdefault(j.parent, []).append(j)
        children.add(j.child)
    roots = [n for n in robot.links if n not in children]
    frames: dict[str, np.ndarray] = {}
    stack = [(r, np.eye(4)) for r in roots]
    while stack:
        name, world = stack.pop()
        if name in frames:
            continue
        frames[name] = world
        for j in by_parent.get(name, []):
            local = j.origin
            if j.jtype in ("revolute", "continuous"):
                local = local @ axis_angle_to_matrix(j.axis, pose.get(j.name, 0.0))
            elif j.jtype == "prismatic":
                slide = np.eye(4)
                slide[:3, 3] = normalize(j.axis) * pose.get(j.name, 0.0)
                local = local @ slide
            stack.append((j.child, world @ local))
    return frames


def primitive_mesh(kind: str, vals: tuple[float, ...]) -> Mesh:
    """Boxes, cylinders and spheres, for the collision-geometry fallback."""
    if kind == "box":
        sx, sy, sz = (v / 2.0 for v in vals[:3])
        corners = np.array(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
            dtype=np.float64,
        )
        faces = [
            (0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7), (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ]
        tris = np.array(faces, dtype=np.int64)
    elif kind == "cylinder":
        radius, length = vals[0], vals[1]
        seg = 28
        ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
        bottom = np.hstack([ring, np.full((seg, 1), -length / 2.0)])
        top = np.hstack([ring, np.full((seg, 1), length / 2.0)])
        corners = np.vstack([bottom, top, [[0, 0, -length / 2.0], [0, 0, length / 2.0]]])
        faces = []
        for i in range(seg):
            j = (i + 1) % seg
            faces.append((i, j, seg + j))
            faces.append((i, seg + j, seg + i))
            faces.append((2 * seg, j, i))
            faces.append((2 * seg + 1, seg + i, seg + j))
        tris = np.array(faces, dtype=np.int64)
    else:
        radius = vals[0]
        rows, cols = 12, 20
        pts, faces = [], []
        for r in range(rows + 1):
            theta = math.pi * r / rows
            for c in range(cols):
                phi = 2 * math.pi * c / cols
                pts.append([
                    radius * math.sin(theta) * math.cos(phi),
                    radius * math.sin(theta) * math.sin(phi),
                    radius * math.cos(theta),
                ])
        for r in range(rows):
            for c in range(cols):
                a = r * cols + c
                b = r * cols + (c + 1) % cols
                faces.append((a, b, a + cols))
                faces.append((b, b + cols, a + cols))
        corners = np.array(pts, dtype=np.float64)
        tris = np.array(faces, dtype=np.int64)
    return Mesh(corners, tris, np.ones((tris.shape[0], 3), dtype=np.float32))


def build_scene(robot: Robot, frames: dict[str, np.ndarray], use_collision: bool) -> Mesh:
    parts: list[Mesh] = []
    for name, link in robot.links.items():
        world = frames.get(name)
        if world is None:
            continue
        sources = link.collisions if use_collision else link.visuals
        for vis in sources:
            if vis.mesh_path is not None:
                mesh = load_mesh(vis.mesh_path)
                verts = mesh.verts * vis.scale
                tris = mesh.tris
                if float(np.prod(vis.scale)) < 0:
                    tris = tris[:, ::-1]
                colors = mesh.colors
                if vis.color is not None and use_collision:
                    colors = np.tile(vis.color, (tris.shape[0], 1))
            elif vis.primitive is not None:
                mesh = primitive_mesh(*vis.primitive)
                verts, tris = mesh.verts, mesh.tris
                base = vis.color if vis.color is not None else np.array(
                    [0.62, 0.66, 0.72], dtype=np.float32
                )
                colors = np.tile(base, (tris.shape[0], 1))
            else:
                continue
            full = world @ vis.origin
            hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
            parts.append(Mesh((hom @ full.T)[:, :3], tris, colors.astype(np.float32)))
    return merge_meshes(parts)


# ---------------------------------------------------------------------------------
# Camera and rasteriser.
# ---------------------------------------------------------------------------------


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    fov_deg: float

    def basis(self) -> np.ndarray:
        fwd = normalize(self.target - self.eye)
        up_world = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fwd, up_world))) > 0.999:
            up_world = np.array([0.0, 1.0, 0.0])
        right = normalize(np.cross(fwd, up_world))
        up = np.cross(right, fwd)
        return np.stack([right, up, -fwd])

    def project(self, verts: np.ndarray, width: int, height: int) -> tuple[np.ndarray, ...]:
        cam = (verts - self.eye) @ self.basis().T
        depth = np.maximum(-cam[:, 2], 1e-6)
        focal = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        aspect = width / float(height)
        ndc_x = focal * cam[:, 0] / depth / aspect
        ndc_y = focal * cam[:, 1] / depth
        px = (ndc_x + 1.0) * 0.5 * width
        py = (1.0 - ndc_y) * 0.5 * height
        return px, py, depth, cam[:, 2]


def fit_camera(
    verts: np.ndarray,
    azimuth: float,
    elevation: float,
    fov: float,
    width: int,
    height: int,
    stage: tuple[float, float, float, float],
    zoom: float,
) -> Camera:
    """Place the eye on the requested bearing, then solve for a distance and a target
    that put the whole robot inside `stage` -- the rectangle left over once the callout
    columns and the title band are reserved."""
    sample = verts[:: max(1, verts.shape[0] // 20000)]
    centre = 0.5 * (sample.min(axis=0) + sample.max(axis=0))
    az, el = math.radians(azimuth), math.radians(elevation)
    direction = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    span = float(np.linalg.norm(sample.max(axis=0) - sample.min(axis=0)))
    dist = span * 3.0
    target = centre.copy()
    x0, y0, x1, y1 = stage
    for _ in range(24):
        cam = Camera(eye=target + dist * direction, target=target, fov_deg=fov)
        px, py, _, _ = cam.project(sample, width, height)
        bw = max(px.max() - px.min(), 1e-6)
        bh = max(py.max() - py.min(), 1e-6)
        scale = min((x1 - x0) / bw, (y1 - y0) / bh) * zoom
        dist /= scale
        cam = Camera(eye=target + dist * direction, target=target, fov_deg=fov)
        px, py, _, _ = cam.project(sample, width, height)
        dx = 0.5 * (x0 + x1) - 0.5 * (px.min() + px.max())
        dy = 0.5 * (y0 + y1) - 0.5 * (py.min() + py.max())
        basis = cam.basis()
        focal = 1.0 / math.tan(math.radians(fov) / 2.0)
        world_per_px_x = (2.0 / width) * (width / float(height)) * dist / focal
        world_per_px_y = (2.0 / height) * dist / focal
        target = target - basis[0] * dx * world_per_px_x + basis[1] * dy * world_per_px_y
    return Camera(eye=target + dist * direction, target=target, fov_deg=fov)


def shade(mesh: Mesh, cam: Camera) -> np.ndarray:
    """Flat Lambert: one key light fixed to the camera, a cool fill from below-right,
    a rim term to lift the silhouette off a dark background, and a floor under the
    base colour so a near-black vendor material still reads as geometry."""
    v0 = mesh.verts[mesh.tris[:, 0]]
    v1 = mesh.verts[mesh.tris[:, 1]]
    v2 = mesh.verts[mesh.tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    basis = cam.basis()
    centroid = (v0 + v1 + v2) / 3.0
    view = centroid - cam.eye
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    facing = np.sum(normals * view, axis=1, keepdims=True)
    normals = np.where(facing > 0, -normals, normals)

    key = basis.T @ normalize(np.array([-0.42, 0.58, 0.70]))
    fill = basis.T @ normalize(np.array([0.75, -0.35, 0.28]))
    ndl = np.clip(normals @ key, 0.0, 1.0)[:, None]
    ndf = np.clip(normals @ fill, 0.0, 1.0)[:, None]
    rim = np.clip(1.0 - np.abs(np.sum(normals * view, axis=1)), 0.0, 1.0)[:, None] ** 3.0

    base = 0.13 + 0.87 * np.clip(mesh.colors.astype(np.float64), 0.0, 1.0)
    fill_tint = np.array([0.55, 0.72, 1.00])
    rim_tint = np.array([0.42, 0.70, 0.94])
    rgb = base * (0.15 + 0.78 * ndl) + base * 0.26 * ndf * fill_tint + 0.30 * rim * rim_tint
    return np.clip(rgb, 0.0, 1.0)


def rasterise(
    mesh: Mesh, cam: Camera, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffered scanline fill. Returns (rgb, coverage mask, depth)."""
    px, py, depth, zcam = cam.project(mesh.verts, width, height)
    rgb = shade(mesh, cam)

    tris = mesh.tris
    tx = px[tris]
    ty = py[tris]
    td = depth[tris]
    in_front = (zcam[tris] < -1e-4).all(axis=1)
    area2 = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (
        ty[:, 1] - ty[:, 0]
    )
    xmin = np.floor(tx.min(axis=1)).astype(np.int64)
    xmax = np.ceil(tx.max(axis=1)).astype(np.int64)
    ymin = np.floor(ty.min(axis=1)).astype(np.int64)
    ymax = np.ceil(ty.max(axis=1)).astype(np.int64)
    keep = (
        in_front
        & (np.abs(area2) > 1e-12)
        & (xmax >= 0)
        & (ymax >= 0)
        & (xmin < width)
        & (ymin < height)
    )
    order = np.flatnonzero(keep)

    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    image = np.zeros((height, width, 3), dtype=np.float32)
    cover = np.zeros((height, width), dtype=bool)

    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, 0, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, 0, height - 1)
    inv_d = 1.0 / td

    for t in order.tolist():
        ix0, ix1, iy0, iy1 = xmin[t], xmax[t], ymin[t], ymax[t]
        if ix0 > ix1 or iy0 > iy1:
            continue
        x0, x1, x2 = tx[t]
        y0, y1, y2 = ty[t]
        i0, i1, i2 = inv_d[t]
        if ix0 == ix1 and iy0 == iy1:
            z = 3.0 / (i0 + i1 + i2)
            if z < zbuf[iy0, ix0]:
                zbuf[iy0, ix0] = z
                image[iy0, ix0] = rgb[t]
                cover[iy0, ix0] = True
            continue
        xs = np.arange(ix0, ix1 + 1, dtype=np.float64) + 0.5
        ys = (np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5)[:, None]
        w0 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        w1 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
        w2 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        a = area2[t]
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if a > 0 else (
            (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        )
        if not inside.any():
            continue
        recip = (w0 * i0 + w1 * i1 + w2 * i2) / a
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(recip > 1e-12, 1.0 / recip, np.inf)
        win = inside & (z < zbuf[iy0 : iy1 + 1, ix0 : ix1 + 1])
        if not win.any():
            continue
        zbuf[iy0 : iy1 + 1, ix0 : ix1 + 1][win] = z[win]
        image[iy0 : iy1 + 1, ix0 : ix1 + 1][win] = rgb[t]
        cover[iy0 : iy1 + 1, ix0 : ix1 + 1][win] = True
    return image, cover, zbuf


def background(width: int, height: int, focus: tuple[float, float], radius: float) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs - focus[0]) / max(radius, 1.0)
    dy = (ys - focus[1]) / max(radius * 0.62, 1.0)
    glow = np.clip(1.0 - (dx * dx + dy * dy), 0.0, 1.0) ** 1.5
    base = np.array(BG, dtype=np.float64) / 255.0
    tint = np.array(BG_GLOW, dtype=np.float64) / 255.0
    return base[None, None, :] + (tint - base)[None, None, :] * glow[:, :, None]


def contact_shadow(
    canvas: np.ndarray, centre: tuple[float, float], rx: float, ry: float
) -> np.ndarray:
    height, width = canvas.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs - centre[0]) / max(rx, 1.0)
    dy = (ys - centre[1]) / max(ry, 1.0)
    alpha = (np.clip(1.0 - (dx * dx + dy * dy), 0.0, 1.0) ** 1.6) * 0.62
    ink = np.array(GROUND, dtype=np.float64) / 255.0
    return canvas * (1.0 - alpha[:, :, None]) + ink[None, None, :] * alpha[:, :, None]


# ---------------------------------------------------------------------------------
# Callouts.
# ---------------------------------------------------------------------------------


@dataclass
class Callout:
    text: str
    link: str
    xyz: tuple[float, float, float]
    side: str
    y: float


def load_fonts(size_body: int, size_title: int, size_small: int):
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            return (
                ImageFont.truetype(regular, size_body),
                ImageFont.truetype(bold, size_title),
                ImageFont.truetype(regular, size_small),
                True,
            )
    try:
        import matplotlib

        ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        regular, bold = ttf / "DejaVuSans.ttf", ttf / "DejaVuSans-Bold.ttf"
        if regular.exists() and bold.exists():
            return (
                ImageFont.truetype(str(regular), size_body),
                ImageFont.truetype(str(bold), size_title),
                ImageFont.truetype(str(regular), size_small),
                True,
            )
    except ImportError:
        pass
    default = ImageFont.load_default()
    return default, default, default, False


def _greedy_wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> list[str]:
    """Greedy wrap, then squeeze the column until one more line would appear. Greedy
    alone leaves a widow -- `12` or `m/s` alone on the last line of a right-aligned
    label -- which is the one typographic thing a reader notices in a figure."""
    lines = _greedy_wrap(draw, text, font, max_width)
    if len(lines) < 2:
        return lines
    lo, hi, best = max_width * 0.40, max_width, lines
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        trial = _greedy_wrap(draw, text, font, mid)
        if len(trial) <= len(lines):
            best, hi = trial, mid
        else:
            lo = mid
    return best


def stroke_line(draw: ImageDraw.ImageDraw, pts, colour, halo, wide: int, thin: int) -> None:
    draw.line(pts, fill=halo, width=wide, joint="curve")
    draw.line(pts, fill=colour, width=thin, joint="curve")


def draw_callouts(
    img: Image.Image,
    callouts: Sequence[Callout],
    anchors: Sequence[tuple[float, float]],
    column: float,
    margin: int,
    font,
) -> None:
    draw = ImageDraw.Draw(img)
    width, height = img.size
    line_h = int(font.size * 1.34)
    halo = tuple(int(c * 0.42) for c in BG)
    for callout, (ax, ay) in zip(callouts, anchors):
        lines = wrap(draw, callout.text, font, column)
        block_h = line_h * len(lines)
        top = callout.y * height - block_h / 2.0
        top = min(max(top, margin + 4), height - margin - block_h)
        mid = top + block_h / 2.0
        left_side = callout.side == "left"
        if left_side:
            text_x = margin + column
            elbow_x = margin + column + 30
            align = "right"
        else:
            text_x = width - margin - column
            elbow_x = width - margin - column - 30
            align = "left"
        stroke_line(
            draw,
            [(text_x + (12 if not left_side else -12), mid), (elbow_x, mid), (ax, ay)],
            ACCENT,
            halo,
            5,
            2,
        )
        draw.ellipse([ax - 8, ay - 8, ax + 8, ay + 8], outline=ACCENT, width=2)
        draw.ellipse([ax - 3.5, ay - 3.5, ax + 3.5, ay + 3.5], fill=ACCENT)
        for i, line in enumerate(lines):
            ly = top + i * line_h
            draw.text(
                (text_x, ly),
                line,
                font=font,
                fill=TEXT,
                anchor="la" if align == "left" else "ra",
                stroke_width=3,
                stroke_fill=halo,
            )


# ---------------------------------------------------------------------------------
# Per-robot presets. Every joint angle below is inside the vendor URDF's own <limit>.
# ---------------------------------------------------------------------------------


def _go2_pose(walking: bool = False) -> dict[str, float]:
    """Unitree's leg convention: thigh and calf both turn about +y, so positive thigh
    swings the knee backwards and the calf then folds the shank forwards. Thigh 0.90 /
    calf -1.80 is Unitree's own symmetric stand: equal 0.213 m links, so the foot lands
    directly under its own hip, 0.265 m below it. `walking` lifts the FR/RL diagonal
    into swing -- 0.047 m of ground clearance, 0.099 m ahead of the stance foot -- so
    the figure reads as a trot. Every value is inside the URDF <limit>:
    thigh [-1.5708, 3.4907], calf [-2.7227, -0.83776], hip [-1.0472, 1.0472]."""
    stance = (0.90, -1.80)
    swing = (0.55, -1.95)
    out: dict[str, float] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        thigh, calf = swing if walking and leg in ("FR", "RL") else stance
        out[f"{leg}_hip_joint"] = 0.02 if leg in ("FL", "RL") else -0.02
        out[f"{leg}_thigh_joint"] = thigh
        out[f"{leg}_calf_joint"] = calf
    return out


def _go2w_pose() -> dict[str, float]:
    """Go2-W's calf runs 0.2264 m to the wheel axle rather than 0.213 m to a point
    foot, and the wheel adds its own radius underneath, so the Go2 stand leaves it on
    tiptoe. Thigh 0.95 / calf -1.8215 solves 0.213 sin a = 0.2264 sin b, which puts
    each axle under its own hip at 0.270 m."""
    out: dict[str, float] = {}
    for leg in ("FL", "FR", "RL", "RR"):
        out[f"{leg}_hip_joint"] = 0.02 if leg in ("FL", "RL") else -0.02
        out[f"{leg}_thigh_joint"] = 0.95
        out[f"{leg}_calf_joint"] = -1.8215
    return out


PRESETS: dict[str, dict[str, Any]] = {
    "go2": {
        "title": "Unitree Go2 — the platform the RGB-only stack was measured on",
        "footer": "No LiDAR and no depth camera are used by this stack, whether or not "
                  "the unit carries them.",
        "credit": "Rendered from unitree_ros/robots/go2_description (BSD-3-Clause, "
                  "© 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. “Unitree Robotics”). "
                  "Vendor meshes are not redistributed in this repository.",
        "azimuth": -34.0,
        "elevation": 14.0,
        "fov": 24.0,
        "zoom": 1.0,
        "pose": _go2_pose(walking=True),
        # Within a column the labels are ordered by the SCREEN HEIGHT of the thing they
        # point at, which is what keeps the leaders from crossing each other.
        "callouts": [
            {"text": "Dorsal mount — D1 arm bolts here; runs use --no-latch-arm",
             "link": "base", "xyz": [0.02, 0.0, 0.075], "side": "right", "y": 0.17},
            {"text": "Front RGB camera — the only sensor this stack uses",
             "link": "front_camera", "xyz": [0.0, 0.0, 0.0], "side": "right", "y": 0.40},
            {"text": "Onboard compute (Jetson, Ubuntu 20.04 / Python 3.8)",
             "link": "base", "xyz": [0.02, -0.065, -0.02], "side": "right", "y": 0.66},
            {"text": "Ethernet tether during live runs",
             "link": "base", "xyz": [-0.20, -0.02, 0.04], "side": "left", "y": 0.22},
            {"text": "12 joint actuators, 3 per leg (hip / thigh / calf)",
             "link": "RR_thigh", "xyz": [0.0, -0.03, -0.12], "side": "left", "y": 0.68},
        ],
    },
    "go2w": {
        "title": "Unitree Go2-W — the peer robot the detector was trained to see",
        "footer": "The wheels are the difference that matters to the detector: same trunk, "
                  "same silhouette from the front, four wheels instead of four point feet.",
        "credit": "Rendered from unitree_ros/robots/go2w_description (BSD-3-Clause, "
                  "© 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. “Unitree Robotics”). "
                  "Vendor meshes are not redistributed in this repository.",
        "azimuth": -34.0,
        "elevation": 14.0,
        "fov": 24.0,
        "zoom": 1.0,
        "pose": _go2w_pose(),
        "callouts": [
            {"text": "Publishes its own pose at 10 Hz over the device mesh — "
                     "no detector needed",
             "link": "base", "xyz": [0.02, 0.0, 0.075], "side": "right", "y": 0.17},
            {"text": "1,903 hand-labelled frames of this robot, class `go2wheel`",
             "link": "base", "xyz": [0.26, -0.045, -0.01], "side": "right", "y": 0.42},
            {"text": "0.70 x 0.31 m footprint; modelled to the policy as a 0.40 m disc "
                     "(the half-diagonal)",
             "link": "base", "xyz": [-0.14, -0.06, 0.02], "side": "left", "y": 0.24},
            {"text": "Wheel actuator at each calf — 16 DOF, not 12",
             "link": "RR_foot", "xyz": [0.0, -0.02, 0.0], "side": "left", "y": 0.70},
        ],
    },
    "lite3": {
        "title": "Deep Robotics Lite3 Venture — RGB only, no depth camera, no LiDAR",
        "footer": "Rendered from the vendor's stock Lite3 description; the Venture units in "
                  "this work carry no head LiDAR or depth module.",
        "credit": "Rendered from deep_robotics_model/Lite3 (BSD-3-Clause, "
                  "© 2024, DeepRoboticsLab). "
                  "Vendor meshes are not redistributed in this repository.",
        "azimuth": -34.0,
        "elevation": 14.0,
        "fov": 24.0,
        "zoom": 1.0,
        "pose": {
            f"{leg}_HipX_joint": 0.0 for leg in ("FL", "FR", "HL", "HR")
        },
        "callouts": [
            {"text": "RGB camera — 1280 x 720; focal length NOT yet measured "
                     "on this platform",
             "link": "TORSO", "xyz": [0.26, -0.03, 0.05], "side": "right", "y": 0.20},
            {"text": "No arm, no LiDAR, no depth camera on this configuration",
             "link": "TORSO", "xyz": [0.24, -0.055, -0.005], "side": "right", "y": 0.44},
            {"text": "High-level UDP axis interface; the transport is sign-only and "
                     "discards commanded magnitude",
             "link": "TORSO", "xyz": [-0.11, -0.06, 0.01], "side": "left", "y": 0.22},
            {"text": "12 joint actuators; measured gait floor 0.30 m/s",
             "link": "HR_THIGH", "xyz": [0.0, -0.03, -0.12], "side": "left", "y": 0.68},
        ],
    },
}

PRESETS["lite3"]["pose"].update(
    {
        "FL_HipY_joint": -0.78, "FL_Knee_joint": 1.514,
        "FR_HipY_joint": -0.78, "FR_Knee_joint": 1.514,
        "HL_HipY_joint": -0.78, "HL_Knee_joint": 1.514,
        "HR_HipY_joint": -0.78, "HR_Knee_joint": 1.514,
    }
)


# ---------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------


def anchor_points(
    robot: Robot,
    frames: dict[str, np.ndarray],
    callouts: Sequence[Callout],
    cam: Camera,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Project each callout's link-frame anchor. This is the whole point of the
    exercise: the label points at a link, not at a pixel someone eyeballed once."""
    pts = []
    for callout in callouts:
        frame = frames.get(callout.link)
        if frame is None:
            known = ", ".join(sorted(robot.links)[:12])
            raise SystemExit(
                f"callout anchor link {callout.link!r} is not in this URDF (have: {known} ...)"
            )
        world = frame @ np.array([*callout.xyz, 1.0])
        pts.append(world[:3])
    px, py, _, _ = cam.project(np.array(pts), width, height)
    return list(zip(px.tolist(), py.tolist()))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Render an annotated quadruped profile from a vendor URDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--urdf", required=True, type=Path, help="path to the vendor URDF")
    ap.add_argument("--out", required=True, type=Path, help="output PNG")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="built-in robot preset")
    ap.add_argument("--callouts", type=Path, help="JSON overriding the preset")
    ap.add_argument("--width", type=int, default=2200)
    ap.add_argument("--height", type=int, default=850)
    ap.add_argument("--azimuth", type=float, help="degrees; 0 looks from straight ahead")
    ap.add_argument("--elevation", type=float, help="degrees above the horizon")
    ap.add_argument("--fov", type=float, help="vertical field of view, degrees")
    ap.add_argument("--zoom", type=float, help="fill fraction of the stage rectangle")
    ap.add_argument("--supersample", type=int, default=2, help="render scale before LANCZOS")
    ap.add_argument("--zero-pose", action="store_true", help="ignore the preset stance")
    ap.add_argument("--collision", action="store_true", help="render <collision> primitives")
    ap.add_argument("--print-pose", action="store_true", help="print the stance and exit")
    return ap


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(PRESETS.get(args.preset, {})) if args.preset else {}
    if args.callouts:
        cfg.update(json.loads(args.callouts.read_text()))
    for key in ("azimuth", "elevation", "fov", "zoom"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    cfg.setdefault("azimuth", -34.0)
    cfg.setdefault("elevation", 14.0)
    cfg.setdefault("fov", 24.0)
    cfg.setdefault("zoom", 0.94)
    cfg.setdefault("title", "")
    cfg.setdefault("footer", "")
    cfg.setdefault("credit", "")
    cfg.setdefault("pose", {})
    cfg.setdefault("callouts", [])
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = resolve_config(args)
    pose: dict[str, float] = {} if args.zero_pose else dict(cfg["pose"])

    robot = parse_urdf(args.urdf)
    frames = forward_kinematics(robot, pose)
    if args.print_pose:
        for name in sorted(pose):
            print(f"{name:<24} {pose[name]:+.4f}")
        return 0

    scene = build_scene(robot, frames, use_collision=args.collision)
    if scene.is_empty():
        raise SystemExit(
            f"no visual geometry loaded from {args.urdf} -- check the mesh checkout"
        )

    width, height = args.width, args.height
    scale = max(1, args.supersample)
    margin = round(width * 0.018)
    column = width * 0.195
    stage = (
        margin + column + width * 0.045,
        height * 0.135,
        width - margin - column - width * 0.045,
        height * 0.885,
    )
    cam = fit_camera(
        scene.verts, cfg["azimuth"], cfg["elevation"], cfg["fov"], width, height, stage,
        float(cfg["zoom"]),
    )

    big_w, big_h = width * scale, height * scale
    rgb, cover, _ = rasterise(scene, cam, big_w, big_h)

    ys, xs = np.nonzero(cover)
    if xs.size == 0:
        raise SystemExit("nothing projected inside the frame; try --zoom or --fov")
    cx = 0.5 * (float(xs.min()) + float(xs.max()))
    proj_w = float(xs.max() - xs.min())
    cy = 0.5 * (float(ys.min()) + float(ys.max()))
    canvas = background(big_w, big_h, (cx, cy), proj_w * 0.85)
    canvas = contact_shadow(
        canvas, (cx, float(ys.max()) + proj_w * 0.012), proj_w * 0.60, proj_w * 0.055
    )
    canvas = np.where(cover[:, :, None], rgb.astype(np.float64), canvas)

    img = Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))
    if scale > 1:
        img = img.resize((width, height), Image.LANCZOS)

    body_size = max(13, round(height * 0.0245))
    title_size = max(16, round(height * 0.040))
    small_size = max(11, round(height * 0.0195))
    font_body, font_title, font_small, real_font = load_fonts(body_size, title_size, small_size)
    if not real_font:
        print("WARNING: no TrueType font found; falling back to Pillow's bitmap font",
              file=sys.stderr)

    callouts = [
        Callout(
            text=c["text"], link=c["link"], xyz=tuple(c.get("xyz", (0.0, 0.0, 0.0))),
            side=c.get("side", "right"), y=float(c.get("y", 0.5)),
        )
        for c in cfg["callouts"]
    ]
    if callouts:
        anchors = anchor_points(robot, frames, callouts, cam, width, height)
        draw_callouts(img, callouts, anchors, column, margin, font_body)

    draw = ImageDraw.Draw(img)
    if cfg["title"]:
        draw.text((margin, int(height * 0.045)), cfg["title"], font=font_title, fill=TEXT,
                  anchor="lm")
    if cfg["footer"]:
        draw.text((margin, height - margin - small_size * 1.1), cfg["footer"], font=font_small,
                  fill=MUTED, anchor="lm")
    if cfg["credit"]:
        for i, line in enumerate(wrap(draw, cfg["credit"], font_small, width * 0.42)):
            draw.text(
                (width - margin, height - margin - small_size * 1.1 - (1 - i) * small_size * 1.25),
                line, font=font_small, fill=DIM, anchor="rm",
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, format="PNG", optimize=True)
    print(
        f"{args.out}  {width} x {height}  {args.out.stat().st_size:,} bytes  "
        f"{scene.tris.shape[0]:,} triangles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
