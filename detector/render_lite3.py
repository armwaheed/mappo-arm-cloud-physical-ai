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

## A BACKGROUND IS A PICTURE PLUS THE CAMERA THAT TOOK IT

The sprite is stood on the floor, and the floor it is stood on is the one in the
background frame. Placing it needs that frame's focal length, camera height and camera
pitch — see :meth:`BackgroundGeometry.ground_row`. Those are properties of the FRAME, not
global constants, and a background shot from a different height, with a different lens, or
with the robot in a different posture breaks the placement silently: the composite still
looks like a picture of a corridor, the box is still tight on the sprite, and the robot is
simply standing at the wrong depth. Nothing downstream can detect it.

So geometry travels WITH the frames. ``--backgrounds`` may be given more than once, and
each directory may carry a ``geometry.json`` describing its camera. That file is
deliberately the format ``calibrate_camera.py`` already writes, so the honest recipe for a
new environment is to calibrate it and copy the result in::

    cp robot-stack/unitree/go2/visual_nav/go2_front_camera.json bg_lab/geometry.json

A directory with no ``geometry.json`` falls back to the command line and says so loudly,
which is the pre-existing behaviour and is right for the one corridor the flags describe.
A frame whose pixel dimensions disagree with its declared geometry is REFUSED rather than
rescaled: ``focal_px`` is expressed in pixels of a particular image size, and from the
pixels alone a downscale, a crop and a different lens are indistinguishable. Declare the
directory's own ``width``/``height``/``focal_px`` and the frames become legal — the
refusal is of the ASSUMPTION, not of the size.

Generated backgrounds, if they ever arrive, are just files in a directory and reach this
module the same way. Nothing here knows or cares where a background came from — but a
generated frame has no camera, so somebody has to decide what geometry to claim for it,
and writing that claim into ``geometry.json`` is where that decision becomes reviewable.

⚠️ RECORDED FRAMES CARRY THE DEBUG OVERLAY. ``visual_nav`` writes its MP4 from the
ANNOTATED canvas — detection rectangles, the top-right plan-view inset, the bottom-left
status plate. Composite onto those and the detector is taught that a peer comes with an
orange rectangle and a black radar square. Use the raw camera frames, or crop the insets,
before pointing ``--backgrounds`` at a recording.

## Posture is part of the geometry

The Go2 rests PRONE and stands only to walk: it initialises prone, acquires its goal
prone, and lies back down whenever the path stays blocked for ``--rest-after`` seconds. A
dry run never enables the legs at all, so it is prone start to finish whatever its status
line says. Prone is a recurring RUN STATE, not a startup transient, and prone frames are
legitimate training data — but they are shot from a different camera:

===========  ==========  ============================  =====================
posture      height_m    pitch_rad                     source
===========  ==========  ============================  =====================
standing     0.32        0.0                           ``go2_front_camera.json``
prone        0.1540      -0.0227 (1.3 deg NOSE-UP)     tape, 2026-08-24
===========  ==========  ============================  =====================

Measured on the robot with a tape: 6 and 1/16 inches floor to lens centre when prone, and
the lens 1.3 degrees nose-up. Neither number was recorded anywhere before — the deployed
calibration's ``height_m: 0.32`` and ``pitch_rad: 0.0`` are the STANDING values, and
applying them to a prone frame puts the ground line 160 px high at 0.5 m, which is 15% of
frame height at the range where avoidance happens.

⚠️ SIGN CONVENTION. ``pitch_rad`` is ``camera_model.FisheyeCamera``'s: tilt below the
body's forward axis, **positive = nose-DOWN**. A nose-UP lens is therefore NEGATIVE. That
is the trap in this module, and the reason the field is not called something friendlier:
the only file that will ever hand these numbers to this script is a calibration written by
``FisheyeCamera.save()``, and a differently-signed field name here would put a hand
negation with no test behind it on that path. Use ``--posture prone`` and the measured
value arrives already signed.

Usage:

    render_lite3.py --backgrounds FRAMES_DIR --out DATASET_DIR --count 2000
    render_lite3.py --backgrounds corridor/ --backgrounds lab/ --out DS --count 4000
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import mujoco
import numpy as np

#: Measured focal length of the Go2 Walk's front camera, pixels. See the module
#: docstring: this is the camera the BACKGROUNDS were shot with, and it is wrong for any
#: other unit. The committed calibration carries it to full precision
#: (``go2_front_camera.json``, ``focal_px: 1290.1637909789656``); the 0.03% difference is
#: far inside the 3.13 deg RMS residual that fit reports.
DEFAULT_FOCAL_PX = 1290.2

#: Frame size the deployment camera delivers, pixels. ``focal_px`` is meaningless without
#: it — the same lens is a different number of pixels per radian at a different capture
#: size — so the two are always declared together.
DEFAULT_FRAME_PX = (1920, 1080)

#: Camera optical centre above the floor, metres, by posture. Standing matches the
#: deployed calibration; prone is 6 and 1/16 inches, tape-measured 2026-08-24.
STANDING_HEIGHT_M = 0.32
PRONE_HEIGHT_M = 0.1540

#: Camera pitch by posture, radians, in ``camera_model.FisheyeCamera``'s convention:
#: tilt below the body's forward axis, POSITIVE = NOSE-DOWN. The prone lens measures 1.3
#: degrees NOSE-UP, so it is negative here. Standing is 0.0 and that value is calibrated,
#: not assumed — ``go2_front_camera.json`` records it.
STANDING_PITCH_RAD = 0.0
PRONE_PITCH_RAD = -math.radians(1.3)

#: Spread of the per-frame pitch jitter, degrees. Standing is the trunk wobble of a
#: trotting quadruped — ``camera_model``'s docstring sizes it at "a 2 deg trunk-pitch
#: wobble (a trotting Go2 does more)". Prone the chassis is resting ON the floor and does
#: not wobble at all, so the spread there stands for the uncertainty in the 1.3 deg tape
#: reading rather than for any motion.
STANDING_PITCH_JITTER_DEG = 2.0
PRONE_PITCH_JITTER_DEG = 0.3

#: Posture presets. Only the two quantities that actually change with posture — the lens
#: does not move relative to the body, so ``focal_px`` is absent on purpose.
POSTURES = {
    "standing": {"height_m": STANDING_HEIGHT_M, "pitch_rad": STANDING_PITCH_RAD,
                 "pitch_jitter_deg": STANDING_PITCH_JITTER_DEG},
    "prone": {"height_m": PRONE_HEIGHT_M, "pitch_rad": PRONE_PITCH_RAD,
              "pitch_jitter_deg": PRONE_PITCH_JITTER_DEG},
}

#: Per-directory geometry sidecar. Named for what it holds rather than for this script,
#: because the file it is meant to be is a copy of a ``calibrate_camera.py`` output.
GEOMETRY_MANIFEST = "geometry.json"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

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

#: Elevation band of the render camera relative to the peer's centre, degrees.
#:
#: ⚠️ This is a RANDOMISATION, not a derivation, and its premise is the STANDING camera:
#: at ~0.32 m against a body ~0.30 m up the view is close to level and can tilt either
#: way as the robot pitches. From the PRONE camera at 0.154 m the peer is above the lens
#: and genuinely seen from below, which this band does not reproduce. Deriving it would
#: need the peer's stance trunk height, and the vendor MJCF does not give one — it spawns
#: the trunk at z=1.0 in free fall — so the honest move is to leave the band alone and
#: record the gap here rather than fit a number to a guess. Listed again under "What this
#: deliberately does NOT model".
ELEVATION_DEG = (-12.0, 18.0)

#: Fraction of samples deliberately pushed off the frame edge. Truncation is the
#: condition that matters — a peer close enough to avoid is usually clipped — and a set
#: without it teaches the detector that robots are always whole.
TRUNCATED_FRACTION = 0.35


@dataclass(frozen=True)
class BackgroundGeometry:
    """The camera a background frame was taken with.

    The field names are ``camera_model.FisheyeCamera``'s, so a JSON file written by
    :meth:`FisheyeCamera.save` — i.e. by ``calibrate_camera.py`` — loads here unchanged
    and an ``annotations.json`` written here can be diffed against the calibration it
    claims to describe. Sharing the names is the point: the alternative is a second
    vocabulary for the same five quantities, and a hand translation between them.

    ``pitch_rad`` is that module's convention too — tilt below the body's forward axis,
    POSITIVE = NOSE-DOWN. See the module docstring; a nose-up lens is negative.
    """

    width: int
    height: int
    focal_px: float
    height_m: float
    pitch_rad: float = STANDING_PITCH_RAD
    pitch_jitter_deg: float = STANDING_PITCH_JITTER_DEG

    def ground_row(self, range_m: float, pitch_rad: float) -> float:
        """Image row where the floor at ``range_m`` lands, for a camera at ``pitch_rad``.

        Under the equidistant fisheye the deployment camera actually is, a ray at angle
        ``theta`` off the optical axis lands ``focal_px * theta`` from the principal
        point. The contact point of something standing at horizontal range ``r`` is
        ``atan(h / r)`` BELOW the horizontal; the optical axis is ``pitch_rad`` below it
        as well, so the ray is ``atan(h / r) - pitch_rad`` below the AXIS::

            row = cy + focal_px * (atan(h / r) - pitch_rad)

        which is why nose-up (negative ``pitch_rad``) pushes the whole floor DOWN the
        frame by ``focal_px * pitch_rad`` at every range — 29 px for the prone lens' 1.3
        degrees, independent of range.

        What that costs if the posture is wrong, at the committed f = 1290.16 px and
        cy = 540 of ``go2_front_camera.json``::

            range   standing (0.32, level)   prone (0.154, 1.3 up)   shift
            0.5 m           1275                     955             -320
            1.0 m            940                     766             -173
            2.0 m            745                     668              -76

        The standing row at 0.5 m is past the bottom of a 1080-row frame, which is not a
        bug in the arithmetic: from a camera 0.32 m up, a peer that close has its feet
        below the sensor and is truncated. Read the other way round, applying the
        standing numbers to a PRONE frame lifts the sprite 320 px — a third of the frame
        — at exactly the range where avoidance happens.

        The principal point is taken as the frame centre, which the calibration supports
        rather than assumes: ``go2_front_camera.json`` fits cx=960.0, cy=540.0 on a
        1920x1080 frame.
        """
        return self.height / 2.0 + self.focal_px * (
            math.atan2(self.height_m, range_m) - pitch_rad)

    def sample_pitch(self, rng: random.Random) -> float:
        """``pitch_rad`` for one frame, jittered for the trunk's own motion."""
        return self.pitch_rad + math.radians(rng.gauss(0.0, self.pitch_jitter_deg))

    def describe(self) -> str:
        """One line naming the pitch in BOTH phrasings, because one of them is a trap."""
        nose = "level" if self.pitch_rad == 0.0 else (
            f"{abs(math.degrees(self.pitch_rad)):.2f} deg nose-"
            f"{'down' if self.pitch_rad > 0 else 'UP'}")
        return (f"{self.width}x{self.height} f={self.focal_px:.1f}px "
                f"h={self.height_m:.4f}m pitch={self.pitch_rad:+.5f}rad ({nose})")


@dataclass(frozen=True)
class Placement:
    """One composited frame and everything needed to audit where the sprite was put."""

    frame: np.ndarray
    box: tuple[int, int, int, int]
    range_m: float
    pitch_rad: float


def geometry_from_mapping(mapping: dict, fallback: BackgroundGeometry, source: str
                          ) -> BackgroundGeometry:
    """One :class:`BackgroundGeometry` from a manifest entry, inheriting what it omits.

    Inheritance is what makes the common cases short. A directory that is the same camera
    in the other posture needs only ``{"posture": "prone"}``; a directory that is a
    different capture size needs only its ``width``/``height``/``focal_px``. Keys this
    class does not know are IGNORED, because the file is meant to be a copy of a
    calibration and those carry provenance (``method``, ``residual_deg_rms``, ``samples``)
    that is there to be read by a human, not by this.
    """
    fields = asdict(fallback)
    posture = mapping.get("posture")
    if posture is not None:
        if posture not in POSTURES:
            raise ValueError(f"{source}: unknown posture {posture!r}, "
                             f"expected one of {sorted(POSTURES)}")
        fields.update(POSTURES[posture])
    # Explicit values win over the posture preset, so a manifest can say "prone, but this
    # unit's lens measured 0.9 deg up" without having to restate the height as well.
    fields.update({k: v for k, v in mapping.items() if k in fields})
    try:
        return replace(fallback, **{k: type(getattr(fallback, k))(v)
                                    for k, v in fields.items()})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


def load_manifest(directory: Path, fallback: BackgroundGeometry
                  ) -> tuple[BackgroundGeometry, dict[str, BackgroundGeometry]] | None:
    """``(default, per-pattern overrides)`` for a directory, or ``None`` if it has none.

    Two shapes are accepted, because the two cases are genuinely different sizes. A
    directory that is ONE camera is a bare calibration object. A directory whose frames
    differ — a recording that starts prone, stands to walk and lies back down inside a
    single clip, which is what these actually do — needs per-frame statements::

        {"default": {"posture": "standing"},
         "frames": {"gs-*.jpg": {"posture": "prone"}, "chair1_*.jpg": "prone"}}

    A bare string where a geometry object is expected is read as a posture name, since
    that is the only thing that varies frame to frame within one recording.
    """
    path = directory / GEOMETRY_MANIFEST
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")

    if "default" in payload or "frames" in payload:
        default = geometry_from_mapping(payload.get("default", {}), fallback, str(path))
        overrides = {
            pattern: geometry_from_mapping(
                {"posture": entry} if isinstance(entry, str) else entry,
                default, f"{path}[{pattern}]")
            for pattern, entry in payload.get("frames", {}).items()
        }
        return default, overrides
    return geometry_from_mapping(payload, fallback, str(path)), {}


def collect_backgrounds(directories: list[Path], fallback: BackgroundGeometry
                        ) -> list[tuple[Path, BackgroundGeometry]]:
    """Every background frame paired with the geometry of the camera that took it.

    Warns, once per directory, when a directory has no manifest and therefore inherits
    the command line. That is the pre-existing behaviour and it is correct for the one
    corridor the flags were measured against — but it is an ASSUMPTION about someone
    else's pictures, and the same class of unstated assumption is what
    ``FROZEN-FEATURE-CEILING.md`` records costing a day: a gate whose negatives silently
    shared their conditions with the training set reported 0% where another day measured
    38%. Say it out loud instead.
    """
    entries: list[tuple[Path, BackgroundGeometry]] = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
        manifest = load_manifest(directory, fallback)
        if manifest is None:
            print(f"[render_lite3] ⚠️ {directory}/{GEOMETRY_MANIFEST} absent — assuming "
                  f"every frame in it was shot with: {fallback.describe()}")
            print("[render_lite3]    A background from another camera, another capture "
                  "size or the other posture breaks the ground line silently.")
            default, overrides = fallback, {}
        else:
            default, overrides = manifest
            print(f"[render_lite3] {directory}/{GEOMETRY_MANIFEST}: {default.describe()}")
            for pattern, geometry in overrides.items():
                print(f"[render_lite3]   {pattern}: {geometry.describe()}")

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = next((g for pattern, g in overrides.items()
                          if fnmatch.fnmatch(path.name, pattern)), default)
            entries.append((path, match))
    return entries


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
            geometry: BackgroundGeometry, rng: random.Random) -> Placement | None:
    """Paste one scaled sprite; returns the placement and its CLIPPED box, or ``None``.

    The box is clipped to the frame because that is what an annotator would draw and what
    the detector must regress. The sprite's own extent decides the physical size, so a
    robot seen end-on is correctly narrower than one seen broadside — the error a single
    published width would introduce.

    THE ROBOT IS STOOD ON THE FLOOR, not pasted at a free height. Range and image row are
    not independent for an object resting on the ground: the contact point at range ``r``
    lands on the row :meth:`BackgroundGeometry.ground_row` gives, which needs the camera's
    height AND its pitch. Sampling the row freely produces robots near the ceiling and
    robots below the floor, and a detector trained on those learns that apparent size
    carries no information about position — discarding the one prior that makes a
    monocular range estimate believable in the first place.
    """
    height, width = background.shape[:2]
    # The renderer's focal length, in pixels of the FULL render buffer — not of the
    # cropped sprite, which is why RENDER_HEIGHT_PX and not sprite.shape[0].
    render_focal_px = 0.5 * RENDER_HEIGHT_PX / np.tan(np.radians(RENDER_FOVY_DEG) / 2.0)

    range_m = rng.uniform(*RANGE_M)
    physical_h = sprite.shape[0] * RENDER_DISTANCE_M / render_focal_px
    target_h = round(physical_h * geometry.focal_px / range_m)
    target_w = round(target_h * sprite.shape[1] / sprite.shape[0])
    if target_h < 12 or target_w < 12 or target_h > height * 3:
        return None

    # Where the floor at this range lands. The pitch is jittered per frame: standing, for
    # the trunk wobble a walking quadruped carries; prone, for the tape's own resolution.
    pitch_rad = geometry.sample_pitch(rng)
    y = round(geometry.ground_row(range_m, pitch_rad) - target_h)
    if rng.random() < TRUNCATED_FRACTION:
        x = rng.randint(-target_w // 2, width - target_w // 2)
    else:
        x = rng.randint(0, max(0, width - target_w))

    # Now that the sprite's position is known, apply the lens' tangential stretch. It
    # depends on WHERE the sprite sits, so it cannot be folded into the scale above.
    stretch = fisheye_tangential_stretch(x + target_w / 2.0, y + target_h / 2.0,
                                         width, height, geometry.focal_px)
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
    return Placement(frame=frame, box=(x0, y0, x1, y1), range_m=range_m,
                     pitch_rad=pitch_rad)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--mjcf", type=Path, required=True,
                        help="Lite3.xml from DeepRoboticsLab/deep_robotics_model")
    parser.add_argument("--backgrounds", type=Path, required=True, action="append",
                        metavar="DIR",
                        help="directory of PEER-FREE frames from the deployment camera. "
                             "Repeatable, so several environments can be mixed; each may "
                             f"carry a {GEOMETRY_MANIFEST} describing its own camera")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--label", default="lite3")
    parser.add_argument("--focal-px", type=float, default=DEFAULT_FOCAL_PX,
                        help="focal length of the DEPLOYMENT camera — see the module "
                             "docstring, every apparent size scales on it")
    parser.add_argument("--frame-px", type=int, nargs=2, default=list(DEFAULT_FRAME_PX),
                        metavar=("W", "H"),
                        help="capture size --focal-px is expressed in. Backgrounds that "
                             "are not this size are refused rather than rescaled")
    parser.add_argument("--posture", choices=sorted(POSTURES), default="standing",
                        help="posture of the robot that shot the backgrounds. Sets the "
                             "camera height and pitch from the tape measurements, which "
                             "is the safe way to get a NOSE-UP pitch signed correctly")
    parser.add_argument("--camera-height-m", type=float, default=None,
                        help="deployment camera's optical centre above the floor. Sets "
                             "where a robot's feet land in the frame at a given range. "
                             f"Default from --posture ({STANDING_HEIGHT_M} standing, "
                             f"{PRONE_HEIGHT_M} prone)")
    parser.add_argument("--camera-pitch-deg", type=float, default=None,
                        help="deployment camera tilt, POSITIVE = NOSE-DOWN (the "
                             "convention camera_model.py uses). A nose-UP lens is "
                             "NEGATIVE — the prone Go2 measures -1.3. Default from "
                             "--posture")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    preset = POSTURES[args.posture]
    fallback = BackgroundGeometry(
        width=args.frame_px[0], height=args.frame_px[1], focal_px=args.focal_px,
        height_m=(preset["height_m"] if args.camera_height_m is None
                  else args.camera_height_m),
        pitch_rad=(preset["pitch_rad"] if args.camera_pitch_deg is None
                   else math.radians(args.camera_pitch_deg)),
        pitch_jitter_deg=preset["pitch_jitter_deg"])
    print(f"[render_lite3] command-line geometry ({args.posture}): {fallback.describe()}")

    try:
        pool = collect_backgrounds(args.backgrounds, fallback)
    except ValueError as exc:
        parser.error(str(exc))
    if not pool:
        parser.error(f"no images in {', '.join(str(d) for d in args.backgrounds)}")

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
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT_PX, width=RENDER_HEIGHT_PX)
    rng = random.Random(args.seed)

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    records, attempts = [], 0
    #: Geometries actually used, de-duplicated, so a record can name one instead of
    #: repeating six floats and so the set's whole geometry fits on one screen.
    used: list[BackgroundGeometry] = []
    refused: dict[str, int] = {}
    while len(records) < args.count and attempts < args.count * 6 and pool:
        attempts += 1
        _randomise_pose(model, data, rng)
        _recolour(model, geoms, rng)
        made = _sprite(renderer, model, data,
                       view=(rng.uniform(0.0, 360.0), rng.uniform(*ELEVATION_DEG)),
                       geoms=geoms)
        if made is None:
            continue
        index = rng.randrange(len(pool))
        path, geometry = pool[index]
        background = cv2.imread(str(path))
        if background is None:
            pool.pop(index)
            refused["unreadable"] = refused.get("unreadable", 0) + 1
            continue
        # focal_px is pixels per radian AT A PARTICULAR CAPTURE SIZE, so a frame of some
        # other size is a different camera until someone says otherwise. From the pixels
        # alone a downscale, a crop and a different lens cannot be told apart, and two of
        # those three would put the sprite at the wrong scale AND the wrong row. Drop it
        # from the pool so the refusal is counted once, not once per attempt.
        if background.shape[:2] != (geometry.height, geometry.width):
            pool.pop(index)
            actual = f"{background.shape[1]}x{background.shape[0]}"
            key = (f"{actual} but its geometry declares "
                   f"{geometry.width}x{geometry.height}")
            refused[key] = refused.get(key, 0) + 1
            continue

        placed = compose(background, made[0], made[1], geometry, rng)
        if placed is None:
            continue
        if geometry not in used:
            used.append(geometry)
        name = f"{args.label}_{len(records):06d}.jpg"
        cv2.imwrite(str(images / name), placed.frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"image": f"images/{name}", "label": args.label,
                        "box": list(placed.box), "background": str(path),
                        "geometry": used.index(geometry),
                        "range_m": round(placed.range_m, 4),
                        "pitch_rad": round(placed.pitch_rad, 6)})

    for reason, n in sorted(refused.items()):
        print(f"[render_lite3] ⚠️ refused {n} background(s): {reason}")

    # Per-record geometry, not one global camera. A set may now be composited from
    # several environments and both postures, and the number that placed each sprite is
    # the only thing that makes a frame re-checkable afterwards.
    (args.out / "annotations.json").write_text(json.dumps(
        {"label": args.label,
         "geometries": [asdict(g) for g in used],
         "asset": "DeepRoboticsLab/deep_robotics_model (BSD-3-Clause)",
         "records": records}, indent=1))
    print(f"{len(records)} frames -> {args.out}  ({attempts} attempts, "
          f"{len(used)} geometr{'y' if len(used) == 1 else 'ies'})")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
