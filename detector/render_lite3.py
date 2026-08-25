#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthesise labelled training frames of a Lite3, for the peer detector.

The Lite3 geometry is Deep Robotics' own, from ``DeepRoboticsLab/deep_robotics_model``
(BSD-3-Clause) — the vendor's MJCF and meshes, not photographs of their website. That
matters twice over: the licence permits redistribution with attribution, and a mesh can
be posed and ranged, which a marketing photo cannot.

## Why composite onto REAL frames instead of rendering a room

A rendered office would put a rendered robot on rendered carpet under rendered light, and
the detector would learn the renderer. Here only the ROBOT is synthetic: it is pasted
into frames from the robot's own camera, so the background carries the real lens
geometry, the real blown-out corridor window, the real sensor noise and the real JPEG
artefacts. The domain gap is then confined to one object instead of the whole image, and
the labels are still free — the segmentation buffer gives an exact mask.

## Scale is derived, never assumed

A sprite is rendered at a known distance through a known focal length, so its pixel
extent fixes a PHYSICAL extent: ``metres = pixels * render_distance / render_focal``.
Re-projecting that through the deployment camera at a sampled range gives the pixel size
the real sensor would see: ``pixels = metres * camera_focal / range``. Nothing here needs
the robot's published dimensions, so nothing here can disagree with them.

⚠️ ``--focal-px`` IS A DEPLOYMENT PARAMETER, NOT A CONSTANT. It defaults to the Go2
Walk's measured 1290.2 px because that is the camera these backgrounds came from. A Lite3
Venture has a different lens, and every apparent size scales linearly on this number:
render at the wrong focal length and the detector learns a scale prior that is wrong by
exactly that ratio. Calibrate the deployment unit and pass its value.

## What this deliberately does NOT model

The pasted sprite is rendered through a PINHOLE camera at a narrow field of view, while
the real camera is an equidistant fisheye. Over a small sprite that difference is slight,
but a robot filling the frame near the edge is genuinely distorted in a way this does not
reproduce. Treat the close-range end of the distribution as the weakest part of the set,
and keep recorded frames for it. The background is unaffected — it was imaged by the real
lens.

Usage:

    render_lite3.py --backgrounds FRAMES_DIR --out DATASET_DIR --count 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

#: Measured focal length of the Go2 Walk's front camera, pixels. See the module
#: docstring: this is the camera the BACKGROUNDS were shot with, and it is wrong for any
#: other unit.
DEFAULT_FOCAL_PX = 1290.2

#: Vertical field of view the sprite is rendered through, degrees. Narrow on purpose —
#: perspective distortion within the sprite grows with it, and the sprite is later scaled
#: rather than re-rendered, so a wide render would bake in a distortion the target range
#: does not have.
RENDER_FOVY_DEG = 20.0

#: Square edge of the offscreen render buffer, pixels. Fixes the render's focal length
#: together with :data:`RENDER_FOVY_DEG`, so both must reach :func:`compose`.
RENDER_HEIGHT_PX = 720

#: Distance the sprite is rendered from, metres. Only sets the working RESOLUTION — the
#: physical extent it implies is what carries through — so it is close, to give the
#: close-range samples real pixels instead of upscaled ones.
RENDER_DISTANCE_M = 1.5

#: Range band to sample, metres. The lower bound is inside the planner's hard gap and the
#: upper is past where a 0.4 m object is worth ranging at all on this sensor.
RANGE_M = (0.4, 4.0)

#: Elevation band of the camera relative to the robot's centre, degrees. The deployment
#: camera sits at ~0.32 m and the robot's body is ~0.30 m up, so the view is close to
#: level and can tilt slightly either way as the robot pitches.
ELEVATION_DEG = (-12.0, 18.0)

#: Fraction of samples deliberately pushed off the frame edge. Truncation is the
#: condition that matters — a peer close enough to avoid is usually clipped — and a set
#: without it teaches the detector that robots are always whole.
TRUNCATED_FRACTION = 0.35


@dataclass(frozen=True)
class DeploymentCamera:
    """The camera the frames will be DEPLOYED against, not the one that renders.

    The two numbers travel together because they answer one question between them —
    how big a robot at range r looks, and which row its feet land on — and passing them
    separately is how a set ends up rendered for one unit and shipped to another.
    """

    focal_px: float
    height_m: float


def robot_geoms(model: mujoco.MjModel) -> list[int]:
    """Geom ids belonging to the ROBOT, i.e. not to the world.

    The MJCF ships a ``floor`` plane in the worldbody, and a segmentation mask of
    "anything the renderer drew" is therefore mostly floor: measured 506,149 of 518,400
    pixels on a 720x720 render. Cropping to that mask boxes the checkerboard and pastes a
    slab of synthetic ground into every frame — labels that are confidently, uselessly
    wrong, and the kind that survive a training run because the loss goes down anyway.
    """
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]


def _sprite(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
            view: tuple[float, float], geoms: list[int]
            ) -> tuple[np.ndarray, np.ndarray] | None:
    """``(bgr, alpha)`` for the robot alone, cropped tight, or ``None`` if it missed.

    The alpha comes from the segmentation buffer rather than a colour key, so a grey
    robot against a grey background still cuts out exactly.
    """
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[1] if model.nbody > 1 else (0.0, 0.0, 0.3)
    camera.distance = RENDER_DISTANCE_M
    camera.azimuth, camera.elevation = view

    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera)
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera)
    seg = renderer.render()
    # Segmentation returns (objid, objtype). Restrict to the robot's own geoms — see
    # `robot_geoms` for what including the world costs.
    mask = (np.isin(seg[:, :, 0], geoms)
            & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
    if mask.sum() < 200:
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR), mask[y0:y1, x0:x1]


def _recolour(model: mujoco.MjModel, geoms: list[int], rng: random.Random) -> None:
    """Tint the robot's geoms toward real Lite3 greys.

    The vendor meshes carry no material, so MuJoCo default-shades them a saturated blue
    that no Lite3 has ever been. Left alone, the strongest and cheapest cue in the whole
    synthetic set is a colour the deployment scene never contains — the detector learns
    it, scores well on held-out synthetic frames, and finds nothing on the robot.
    """
    for geom in geoms:
        value = rng.uniform(0.18, 0.72)
        model.geom_rgba[geom] = (value, value * rng.uniform(0.96, 1.04),
                                 value * rng.uniform(0.96, 1.06), 1.0)


def _randomise_pose(model: mujoco.MjModel, data: mujoco.MjData, rng: random.Random
                    ) -> None:
    """Perturb the joint angles around the model's default stance.

    Small deliberately: this is a standing or slowly-walking peer, not a robot mid-leap,
    and a set full of implausible poses spends capacity on configurations the demo will
    never present.
    """
    for joint in range(model.njnt):
        address = model.jnt_qposadr[joint]
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = model.jnt_range[joint]
        span = (hi - lo) if hi > lo else 0.6
        data.qpos[address] = float(np.clip(data.qpos[address]
                                           + rng.gauss(0.0, 0.08 * span), lo, hi))
    mujoco.mj_forward(model, data)


def _light(sprite: np.ndarray, rng: random.Random) -> np.ndarray:
    """Jitter brightness, contrast and colour temperature of the sprite.

    The renderer's lighting is not the corridor's. This does not make it so, but it stops
    the detector keying on a single constant appearance, which is the failure mode that
    makes a synthetic set score well on itself and nowhere else.
    """
    out = sprite.astype(np.float32)
    out *= rng.uniform(0.55, 1.35)                                  # exposure
    out = (out - out.mean()) * rng.uniform(0.75, 1.25) + out.mean()  # contrast
    for channel, gain in enumerate((rng.uniform(0.92, 1.08),) * 3):
        out[:, :, channel] *= gain * rng.uniform(0.96, 1.04)         # colour cast
    return np.clip(out, 0, 255).astype(np.uint8)


def fisheye_tangential_stretch(column: float, row: float, width: int, height: int,
                               focal_px: float) -> float:
    """The ``theta / sin(theta)`` factor an equidistant fisheye applies off-axis.

    The sprite is rendered ON-AXIS — the render camera looks straight at the robot — and
    then placed using ``r = f * theta``, so its position and its RADIAL extent are already
    what the fisheye would produce. One term is still missing.

    For an object at off-axis angle ``theta``, an equidistant lens images a physical
    height ``h`` at range ``R`` across ``f * h / R`` pixels radially, but a physical width
    ``w`` across ``f * (w / R) * (theta / sin theta)`` pixels tangentially: the azimuthal
    extent subtended by ``w`` is ``(w / R) / sin theta``, and it lands on a circle of
    radius ``f * theta``. The ratio is the stretch returned here.

    It is small where it matters and large where it does not — 2.1% at 20 degrees off
    axis, 13.3% at the corner of this frame — because a peer close enough to avoid is
    roughly ahead. Applied because it is exact and costs nothing, not because it is the
    dominant term: unmodelled contact shadows, mismatched lighting direction, and the
    absent specular highlights on what is a brushed-metal robot are all larger.
    """
    radius = math.hypot(column - width / 2.0, row - height / 2.0)
    theta = radius / focal_px
    return 1.0 if theta < 1e-6 else theta / math.sin(theta)


def compose(background: np.ndarray, sprite: np.ndarray, mask: np.ndarray,
            camera: DeploymentCamera, rng: random.Random
            ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Paste one scaled sprite; returns the frame and its CLIPPED box, or ``None``.

    The box is clipped to the frame because that is what an annotator would draw and what
    the detector must regress. The sprite's own extent decides the physical size, so a
    robot seen end-on is correctly narrower than one seen broadside — the error a single
    published width would introduce.

    THE ROBOT IS STOOD ON THE FLOOR, not pasted at a free height. Range and image row are
    not independent for an object resting on the ground: at range ``r`` a camera ``h``
    above the floor sees the contact point ``focal * atan(h / r)`` pixels below the
    optical axis. Sampling the row freely produces robots near the ceiling and robots
    below the floor, and a detector trained on those learns that apparent size carries no
    information about position — discarding the one prior that makes a monocular range
    estimate believable in the first place.
    """
    height, width = background.shape[:2]
    # The renderer's focal length, in pixels of the FULL render buffer — not of the
    # cropped sprite, which is why RENDER_HEIGHT_PX and not sprite.shape[0].
    render_focal_px = 0.5 * RENDER_HEIGHT_PX / np.tan(np.radians(RENDER_FOVY_DEG) / 2.0)

    range_m = rng.uniform(*RANGE_M)
    physical_h = sprite.shape[0] * RENDER_DISTANCE_M / render_focal_px
    target_h = round(physical_h * camera.focal_px / range_m)
    target_w = round(target_h * sprite.shape[1] / sprite.shape[0])
    if target_h < 12 or target_w < 12 or target_h > height * 3:
        return None

    # Where the floor at this range lands, in image rows. Jittered by a couple of degrees
    # for the trunk pitch a walking quadruped actually carries.
    feet_row = height / 2.0 + camera.focal_px * (
        np.arctan2(camera.height_m, range_m) + np.radians(rng.gauss(0.0, 2.0)))
    y = round(feet_row - target_h)
    if rng.random() < TRUNCATED_FRACTION:
        x = rng.randint(-target_w // 2, width - target_w // 2)
    else:
        x = rng.randint(0, max(0, width - target_w))

    # Now that the sprite's position is known, apply the lens' tangential stretch. It
    # depends on WHERE the sprite sits, so it cannot be folded into the scale above.
    stretch = fisheye_tangential_stretch(x + target_w / 2.0, y + target_h / 2.0,
                                         width, height, camera.focal_px)
    # Tangential means perpendicular to the radius from the principal point. Resolving
    # that exactly needs a rotation; at these magnitudes the dominant component is
    # horizontal for a sprite standing on the floor near the vertical centre, so the
    # stretch is applied to width. Documented rather than hidden: the residual is the
    # vertical component, under 2% over the band where a peer is worth avoiding.
    target_w = max(8, round(target_w * stretch))
    scaled = _light(cv2.resize(sprite, (target_w, target_h),
                               interpolation=cv2.INTER_AREA), rng)
    scaled_mask = cv2.resize(mask, (target_w, target_h),
                             interpolation=cv2.INTER_NEAREST)

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + target_w, width), min(y + target_h, height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    patch_mask = scaled_mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(bool)
    if patch_mask.sum() < 64:
        return None
    frame = background.copy()
    region = frame[y0:y1, x0:x1]
    region[patch_mask] = scaled[y0 - y:y1 - y, x0 - x:x1 - x][patch_mask]
    frame[y0:y1, x0:x1] = region

    # Blur the composite to the motion blur the recorded runs actually carry — the
    # detector's hardest real frames are the ones where the peer is closest and the
    # exposure is 200-380 ms.
    if rng.random() < 0.4:
        k = rng.choice((3, 5))
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return frame, (x0, y0, x1, y1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--mjcf", type=Path, required=True,
                        help="Lite3.xml from DeepRoboticsLab/deep_robotics_model")
    parser.add_argument("--backgrounds", type=Path, required=True,
                        help="directory of PEER-FREE frames from the deployment camera")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--label", default="lite3")
    parser.add_argument("--focal-px", type=float, default=DEFAULT_FOCAL_PX,
                        help="focal length of the DEPLOYMENT camera — see the module "
                             "docstring, every apparent size scales on it")
    parser.add_argument("--camera-height-m", type=float, default=0.32,
                        help="deployment camera's optical centre above the floor. Sets "
                             "where a robot's feet land in the frame at a given range")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    backgrounds = sorted(p for p in args.backgrounds.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not backgrounds:
        parser.error(f"no images in {args.backgrounds}")

    # The vendored MJCF caps its offscreen framebuffer at 640x480, which is a property
    # of Deep Robotics' asset and not ours to edit — a re-pull would revert it and the
    # failure is a hard error at renderer construction. Raise it on the loaded spec
    # instead, which leaves the file on disk untouched.
    spec = mujoco.MjSpec.from_file(str(args.mjcf))
    spec.visual.global_.offwidth = RENDER_HEIGHT_PX
    spec.visual.global_.offheight = RENDER_HEIGHT_PX
    model = spec.compile()
    data = mujoco.MjData(model)
    geoms = robot_geoms(model)
    camera = DeploymentCamera(focal_px=args.focal_px, height_m=args.camera_height_m)
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT_PX, width=RENDER_HEIGHT_PX)
    rng = random.Random(args.seed)

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    records, attempts = [], 0
    while len(records) < args.count and attempts < args.count * 6:
        attempts += 1
        _randomise_pose(model, data, rng)
        _recolour(model, geoms, rng)
        made = _sprite(renderer, model, data,
                       view=(rng.uniform(0.0, 360.0), rng.uniform(*ELEVATION_DEG)),
                       geoms=geoms)
        if made is None:
            continue
        background = cv2.imread(str(rng.choice(backgrounds)))
        if background is None:
            continue
        placed = compose(background, made[0], made[1], camera, rng)
        if placed is None:
            continue
        frame, box = placed
        name = f"{args.label}_{len(records):06d}.jpg"
        cv2.imwrite(str(images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"image": f"images/{name}", "label": args.label, "box": list(box)})

    (args.out / "annotations.json").write_text(json.dumps(
        {"label": args.label, "focal_px": args.focal_px,
         "camera_height_m": args.camera_height_m,
         "asset": "DeepRoboticsLab/deep_robotics_model (BSD-3-Clause)",
         "records": records}, indent=1))
    print(f"{len(records)} frames -> {args.out}  ({attempts} attempts)")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
