# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Annotated debug view — the camera frame plus a top-down plan of what the robot believes.

A navigation run either worked or it did not, and the difference is usually invisible
in the raw video: a stop can mean "saw the person" or "the detector missed and the
watchdog fired". The overlay makes the belief state legible — detections with their
estimated range, tracked velocities, the goal, and the arc the planner actually chose —
so a recording is evidence rather than an anecdote.

Drawing is deliberately kept off the control path: ``visual_nav.py`` renders at the
perception rate into a video file, never inside the loop that commands the legs.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

# BGR. Deuteranopia-safe pairing: amber/blue rather than red/green, since the two
# things that must never be confused are "obstacle" and "goal".
COLOUR_PERSON = (60, 160, 250)      # amber
COLOUR_GOAL = (230, 160, 60)        # blue
COLOUR_TRACK = (250, 250, 250)
COLOUR_PATH = (120, 250, 120)
COLOUR_TEXT = (245, 245, 245)
COLOUR_PANEL = (32, 32, 32)
COLOUR_HOLD = (70, 70, 250)
COLOUR_STATIC = (210, 210, 120)     # teal — mapped furniture, not a mover

#: Below this an obstacle is drawn as static (no velocity arrow). A mapped landmark is
#: exactly zero, so this only has to reject arithmetic noise, not classify anything.
STATIC_SPEED_M_S = 0.02

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(image: np.ndarray, ranged: Sequence) -> None:
    """Box every detection and label it with the range and which prior produced it.

    Takes :class:`~person_detector.RangedDetection` records rather than parallel
    box/range/source lists, so a box can never be labelled with another's range.
    """
    for item in ranged:
        detection = item.detection
        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(image, p1, p2, COLOUR_PERSON, 2)
        label = (f"{detection.label} {item.range_m:.1f}m ({item.source}) "
                 f"{detection.score:.2f}")
        _label(image, label, (p1[0], max(18, p1[1] - 6)), COLOUR_PERSON)


def draw_goal(image: np.ndarray, fix) -> None:
    """Label the goal sighting for THIS frame, or draw nothing.

    Deliberately per-sighting rather than persistent: the label carries a bearing and a
    range, and both go stale the moment the robot moves. The latched goal that survives
    between sightings is drawn by :func:`draw_plan_view` instead, in the frame where it
    is still true.
    """
    if fix is not None:
        _label(image, f"GOAL {fix.range_m:.2f}m @ {math.degrees(fix.bearing_rad):+.0f}deg",
               (16, 40), COLOUR_GOAL)


def _label(image: np.ndarray, text: str, origin: tuple[int, int],
           colour: tuple[int, int, int], scale: float = 0.6) -> None:
    """Text on a filled plate, so it stays readable over a bright or busy frame."""
    (width, height), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 3, y - height - 4), (x + width + 3, y + baseline),
                  COLOUR_PANEL, cv2.FILLED)
    cv2.putText(image, text, (x, y), _FONT, scale, colour, 1, cv2.LINE_AA)


def draw_calibration(image: np.ndarray, detection, centre_px, yaw_offset_deg: float,
                     frame_index: int, sightings: int) -> None:
    """Annotate one frame of a spin calibration sweep.

    The method fits exactly two quantities against each other — how far the ROBOT has
    turned, and where the target sits ACROSS THE FRAME — so the overlay shows both and
    nothing else. As the robot yaws one way the marker slides the other; that they move
    together, by equal and opposite angles, IS the calibration.
    """
    height, width = image.shape[:2]
    if detection is not None:
        cv2.rectangle(image, (int(detection.x1), int(detection.y1)),
                      (int(detection.x2), int(detection.y2)), COLOUR_PERSON, 3)
    if centre_px is not None:
        x = int(centre_px[0])
        cv2.line(image, (x, 0), (x, height), COLOUR_PATH, 2)

    # Pixel ruler across the bottom: frame extent, centre tick, and the target marker.
    bar_y = height - 60
    cv2.line(image, (60, bar_y), (width - 60, bar_y), COLOUR_TEXT, 2)
    cv2.line(image, (width // 2, bar_y - 14), (width // 2, bar_y + 14), COLOUR_TEXT, 2)
    if centre_px is not None:
        x = int(np.clip(centre_px[0], 60, width - 60))
        cv2.drawMarker(image, (x, bar_y), COLOUR_PATH, cv2.MARKER_TRIANGLE_DOWN, 26, 3)

    _label(image, f"robot yaw {yaw_offset_deg:+6.1f} deg", (60, 60), COLOUR_GOAL, 1.0)
    target = "target x --" if centre_px is None else f"target x {centre_px[0]:6.0f} px"
    _label(image, target, (60, 105), COLOUR_PATH, 1.0)
    _label(image, f"frame {frame_index}   sightings {sightings}", (60, 145),
           COLOUR_TEXT, 0.7)


def draw_status(image: np.ndarray, lines: Sequence[str]) -> None:
    """Status block, bottom-left."""
    y = image.shape[0] - 12 - 22 * (len(lines) - 1)
    for line in lines:
        _label(image, line, (16, y), COLOUR_TEXT)
        y += 22


def draw_plan_view(image: np.ndarray, pose: tuple[float, float, float],
                   obstacles: Sequence, goal_xy: tuple[float, float] | None,
                   command, size_px: int = 300, span_m: float = 6.0) -> None:
    """Top-down plan inset, drawn in the frame's top-right corner.

    Robot-centric and rotated into the BODY frame — up is where the robot is facing —
    because that is the frame the viewer is watching the camera in. Everything else is
    held in odom, so this is purely a presentation transform.

    ``obstacles`` are :class:`~avoidance.Obstacle` — what the planner ACTUALLY planned
    against, not the tracks they came from. That distinction is the point of the inset:
    an obstacle carries the position extrapolated to now and the radius already inflated
    for uncertainty, so a disc drawn here is the disc the planner refused to enter. Drawn
    from tracks instead, the panel would show a tidy 0.35 m circle while the planner was
    dodging something three times that size, and the recording would make a correct stop
    look inexplicable.
    """
    height, width = image.shape[:2]
    # Clamp to what the frame can hold. The caller picks size_px from the footage (see
    # replay.py), so a small clip would otherwise compute a negative origin and the
    # final assignment would land the panel somewhere else entirely, or raise.
    size_px = max(1, min(size_px, width - 32, height - 32))
    x0, y0 = width - size_px - 16, 16
    panel = np.full((size_px, size_px, 3), 24, dtype=np.uint8)
    centre = size_px // 2
    scale = (size_px / 2.0) / span_m          # pixels per metre

    for ring_m in range(1, int(span_m) + 1):
        cv2.circle(panel, (centre, centre), int(ring_m * scale), (56, 56, 56), 1)

    robot_x, robot_y, robot_yaw = pose
    cos_y, sin_y = math.cos(-robot_yaw), math.sin(-robot_yaw)

    def to_panel(world_x: float, world_y: float) -> tuple[int, int]:
        """Odom metres -> panel pixels, rotated so body +x points up the panel."""
        dx, dy = world_x - robot_x, world_y - robot_y
        body_x = dx * cos_y - dy * sin_y
        body_y = dx * sin_y + dy * cos_y
        return centre - int(body_y * scale), centre - int(body_x * scale)

    if goal_xy is not None:
        gx, gy = to_panel(*goal_xy)
        cv2.drawMarker(panel, (gx, gy), COLOUR_GOAL, cv2.MARKER_STAR, 16, 2)

    for obstacle in obstacles:
        px, py = to_panel(obstacle.x, obstacle.y)
        speed = math.hypot(obstacle.vx, obstacle.vy)
        # Static obstacles get their own colour: the viewer's first question about a
        # stop is "did it see the bin or a ghost of a person?", and the panel should
        # answer that without needing the console log alongside it.
        colour = COLOUR_STATIC if speed < STATIC_SPEED_M_S else COLOUR_TRACK
        cv2.circle(panel, (px, py), max(3, int(obstacle.radius_m * scale)), colour, 2)
        if speed >= STATIC_SPEED_M_S:
            # Velocity arrow, drawn as one second of predicted travel.
            tip_x, tip_y = to_panel(obstacle.x + obstacle.vx, obstacle.y + obstacle.vy)
            if (tip_x - px) ** 2 + (tip_y - py) ** 2 > 9:
                cv2.arrowedLine(panel, (px, py), (tip_x, tip_y), COLOUR_PERSON, 2,
                                tipLength=0.3)
        cv2.putText(panel, f"{obstacle.label} {speed:.1f}", (px + 8, py - 8), _FONT,
                    0.4, colour, 1, cv2.LINE_AA)

    if command is not None:
        _draw_command_arc(panel, centre, scale, command)

    # The robot itself, last, so it sits on top.
    cv2.drawMarker(panel, (centre, centre), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 12, 2)
    cv2.rectangle(panel, (0, 0), (size_px - 1, size_px - 1), (90, 90, 90), 1)
    cv2.putText(panel, f"{span_m:.0f}m", (6, size_px - 8), _FONT, 0.4, (150, 150, 150),
                1, cv2.LINE_AA)
    image[y0:y0 + size_px, x0:x0 + size_px] = panel


def _draw_command_arc(panel: np.ndarray, centre: int, scale: float, command,
                      seconds: float = 2.0, steps: int = 20) -> None:
    """Trace the path the chosen command would follow, in the panel's body frame."""
    colour = COLOUR_HOLD if command.is_stop else COLOUR_PATH
    if command.is_stop:
        cv2.circle(panel, (centre, centre), 10, colour, 2)
        return
    x = y = yaw = 0.0
    dt = seconds / steps
    points = []
    for _ in range(steps):
        x += (command.vx * math.cos(yaw) - command.vy * math.sin(yaw)) * dt
        y += (command.vx * math.sin(yaw) + command.vy * math.cos(yaw)) * dt
        yaw += command.wz * dt
        points.append((centre - int(y * scale), centre - int(x * scale)))
    cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, colour, 2)
