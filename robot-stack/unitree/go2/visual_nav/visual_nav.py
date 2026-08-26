#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Walk a Unitree Go2 to a goal using only its front RGB camera, avoiding people.

    # off-robot rehearsal: perceive and plan, never move a leg
    python3 visual_nav.py --marker-size 0.20 --record run.mp4

    # live: stand, walk, lie back down
    python3 visual_nav.py --marker-size 0.20 --live --record run.mp4

One camera, no depth, no LiDAR. The pipeline is:

    VideoClient JPEG -+-> MobileNet-SSD person boxes -+-> fisheye bearing + size-prior
                      |                                |   range
                      +-> HSV colour blob (a prop no  -+
                      |   detector was trained on)     |
                      |                                +-> constant-velocity tracker
                      |                                |   (MOVERS, odom frame)
                      |                                +-> landmark map (STATIC, odom)
                      +-> cropped SSD pass -> the goal, latched in odom
                                                       |
                                                       v
             dynamic-window planner scored against PREDICTED obstacle positions
                                                       |
                                                       v
                                              SportClient velocity

Each stage's reasoning lives in its own module; three decisions are properties of the
whole and belong here.

**Perception runs at ~5-7 Hz, control at 10 Hz, and they are not the same clock.**
Measured on this Jetson: the person pass costs 114 ms, colour segmentation 10 ms, and
the throttled goal pass another 69 ms on the cycles it lands. Blocking the control loop
on that would leave the legs holding a stale velocity for a fifth of a second at a time,
so perception runs in its own thread and the controller consumes the newest result it
has. That result is a few hundred milliseconds old by the time it is used (measured over
a run with all three passes: median 309 ms, p90 436 ms), so tracks are extrapolated to
the present before planning and their radii grow to cover the extrapolation — the robot
plans against where people are NOW, with honest uncertainty, not where they were at
the frame adapter's timestamp. Landmarks need neither, because they do not move. The
Go2 adapter timestamps a new JPEG at local ARRIVAL rather than at shutter — the RPC
exposes no sensor timestamp — so transport latency stays inside the measured calibration
and the safety margin. Any other camera binding must state which of the two it reports,
because the difference is invisible until the robot is moving.

The margin here is thinner than it looks: ``perception_timeout_s`` is 0.6 s and the
worst observed cycle was 0.598 s. Running the goal pass every cycle rather than on its
throttle put it over repeatedly. Adding a fourth pass needs this re-measured, not
assumed.

**The robot rests prone and stands only to walk.** The D1 arm loads the hind legs
continuously (see ``safety.py``), so standing is treated as a cost rather than the
default posture: the run starts prone, acquires its goal prone, stands to walk, lies
down again if the path stays blocked longer than ``--rest-after``, and always lies
down on the way out. Perception is fully functional while prone, which is what makes
this practical rather than merely careful.

**Motion is opt-in.** Without ``--live`` every stage runs for real against the real
camera and the planner prints what it would command, but no leg moves and the robot
never leaves the floor. That is the mode to develop in.

**The D1 arm must be latched before anything walks.** An unpowered D1 back-drives —
its base yaw crept 13.4 deg during a turning test — and 3.15 kg sitting off the dorsal
centreline throws the vendor locomotion controller off balance. So ``main`` latches it
by default and REFUSES the run if the latch did not take. The operator still hand-poses
it flat along the spine first; ``latch_arm`` only ever holds joints where they already
are.

SCOPE — read this before trusting it near furniture. This pipeline models MOVING
obstacles, plus ONE NAMED STATIC PROP, and the difference between those two things and
"static-obstacle sensing" is the whole of the safety case.

With ``--static-prop`` the robot finds a *specific* object it has been told the colour
and size of, checks it against three shape gates, and maps it in odom. That is not
sensing; it is recognising one thing it was told to expect. A monocular camera with a
size prior can range a *person* because it knows how big people are, and can range the
bin because it was handed a tape measure — it still knows nothing about a wall, a table
leg or a doorframe, and it never will. The lane must still be clear of everything except
the props, and there must still be an operator on the remote. For general static geometry
the robot's LiDAR and ``lib/navigation.py`` are the right tools, and combining the two is
the obvious next step rather than something this module quietly pretends to do.

SCOPE — IT PLANS AROUND PEOPLE IT CAN SEE, PLUS A SHORT GRACE PERIOD. A track that
leaves the field of view coasts on its last velocity while its covariance inflates
(``tracker.py``), and that inflation is added to the obstacle's radius. Once the
uncertainty outgrows the space between robot and person, no sampled command clears the
hard gap and the robot stops — for someone who may be nowhere near its path — until the
track is pruned at :data:`tracker.COAST_TIMEOUT_S`. Measured, for a person seen and then
lost from view, as the time until the commanded velocity reaches zero:

    range when lost   1.0 m   2.0 m   3.0 m   4.0 m   5.0 m   6.0 m
    keeps moving      0.30 s  0.59 s  1.16 s  1.59 s  2.02 s  2.45 s

An earlier version of this table read 0.87/1.45/1.87/2.30/2.73 s past the first column.
Those were measured as the time until the planner REPORTED ``hold``, and the two were
not the same thing: a full-stop candidate was appended to every velocity window and
competed on cost, and a stationary rollout has a clearance cost of zero by construction,
so near an obstacle the planner commanded ``v=(0,0,0)`` while still calling it ``goal``.
The robot had stopped; only the label had not. The numbers above are what the legs do.

Note the direction: the budget is SHORTEST for the closest people, because a smaller
gap is swallowed sooner — this is not a maximum-range limit and staying close does not
help. While the person stays in view the behaviour is sound across the detector's whole
usable band, and stopping for someone 1 m ahead is the correct answer, not a defect.
The operational rule is therefore about visibility, not distance: keep the person in
frame. A volunteer who crosses the frame and walks on is exactly the case this handles;
one who steps out of shot and stays there parks the robot for up to three seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # locomotion/, d1_arm/

import contextlib

import overlay
from avoidance import (
    STATIC_HARD_GAP_M,
    STATIC_SOFT_GAP_M,
    DynamicWindowPlanner,
    Limits,
    Obstacle,
    PlannerConfig,
    base_reason,
)
from camera_model import GO2_CAMERA_HEIGHT_M, FisheyeCamera
from colour_detector import (
    COLOUR_PROFILE_SCHEMA,
    PROFILES,
    ColourBlobDetector,
    ColourProfile,
    load_colour_profile,
)
from expansion import ExpansionConsistency
from goal import (
    DEFAULT_GOAL_CROP,
    DEFAULT_GOAL_INPUT_SIZE,
    DEFAULT_GOAL_REFRESH_S,
    DEFAULT_MARKER_ID,
    DEFAULT_MARKER_SIZE_M,
    ArucoGoal,
    DetectedObjectGoal,
    GoalSource,
    OdomWaypoint,
)
from lifecycle import run_cleanup
from person_detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_STATIC_CONFIDENCE,
    DYNAMIC_CLASSES,
    PERSON_PRIOR,
    STATIC_CLASSES,
    PersonDetector,
    SizePrior,
    range_detections,
)
from robot_bindings import Go2Bindings, warn_if_below_go2_gait_floor
from static_map import StaticObstacleMap
from telemetry import TelemetryWriter, TickProfiler
from tracker import PROCESS_ACCEL_SIGMA, ObstacleTracker, observation_from

DEFAULT_MODEL_DIR = Path.home() / "go2_models"

#: Label every sub-threshold NEURAL static candidate is mapped under, whatever VOC called
#: it. ONE label, deliberately: `StaticObstacleMap` keys its plan-view radii by label, and
#: the operator supplies exactly one measured height/width/radius triple for the prop they
#: staged. Carrying the VOC label through instead would key the map on a classification
#: this detector has been measured to get wrong — the same recycling bin comes back
#: `tvmonitor` in one frame and `chair` in the next, which would split one bin into two
#: landmarks with two different radii.
STATIC_DETECT_LABEL = "static-detect"
# One frame per PERCEPTION cycle, so every frame in the video is a decision the planner
# really made rather than a resampled copy. The playback rate, though, is this fixed
# number and NOT the rate perception achieved — those differ whenever the detector runs
# slower than 7 Hz, which it does under walking load. Measured on a live run: 145 cycles
# over 20.5 s is 7.1 Hz nominal but the recording plays in 13.1 s, so the video runs
# ~1.6x fast. Re-time it before quoting a duration off the footage.
RECORD_FPS = 7.0

# ── Stall detection ─────────────────────────────────────────────────────────
# Seconds of pose history the stall gate judges on. Long enough that a legitimate slow
# manoeuvre is not mistaken for being stuck — a spot turn commands little translation and
# a careful sidestep is slow — and short enough that a robot pushing into something is
# stopped in seconds rather than for the rest of the run's budget.
PROGRESS_WINDOW_S = 4.0
# Fraction of the commanded travel the robot must actually achieve over that window. Low
# on purpose: the vendor gait lags its target, a swerve trades forward speed for lateral,
# and the odometry itself drifts. The failure this catches is total, not marginal —
# across two live runs the measured figure was ZERO for 21 s and then 12 s.
PROGRESS_FRACTION = 0.20
# Below this commanded speed the robot is not being asked to go anywhere, so there is
# nothing to fail to achieve.
PROGRESS_MIN_COMMAND_M_S = 0.05


@dataclass
class NavConfig:
    """Everything tunable about a run."""

    control_hz: float = 10.0
    perception_timeout_s: float = 0.6   # newest frame older than this -> stop
    max_run_s: float = 90.0
    arrive_tolerance_m: float = 1.0     # stop this far short of the goal marker
    rest_after_s: float = 15.0          # held this long -> lie down and wait
    goal_search_s: float = 20.0         # give up if the goal is never sighted
    live: bool = False
    require_arm: bool = True
    latch_arm: bool = True              # hard requirement — see main()
    motion_mode: str = "normal"
    rest_when_blocked: bool = True      # false when posture is operator-controlled
    initially_standing: bool = False


@dataclass(frozen=True)
class PerceptionResult:
    """One completed perception cycle."""

    seq: int
    capture_time: float
    pose: tuple[float, float, float]
    observations: list          # tracker Observations of MOVERS, odom frame
    ranged: list                # RangedDetections, for the overlay
    static_observations: Sequence = ()   # tracker Observations of STATIC props
    goal_fix: object = None
    image: np.ndarray | None = None
    detect_ms: float = 0.0
    #: The WHOLE cycle's compute, and the part of it spent blocked on the camera.
    #: ``detect_ms`` covers the goal pass, the tiered detect and colour segmentation and
    #: nothing else, so it cannot say whether this thread is compute-bound or
    #: camera-bound. Measured on the 2026-08-25 runs, a third of the cycle is outside it
    #: — 317 ms per cycle against 202 ms of detect — and nothing said where.
    cycle_ms: float = 0.0
    wait_ms: float = 0.0
    #: ``detect_ms`` SPLIT INTO THE THREE PASSES IT SPANS — ``goal``, ``detect``,
    #: ``colour``. One number for three passes cannot answer the question issue #18 ends
    #: on: if perception is 202 ms and detection dominates, the 300x300 SSD input is the
    #: lever; if the throttled goal pass is half of it, the lever is its cadence. The
    #: module docstring quotes 114 / 10 / 69 ms for the three from a measurement nothing in
    #: the tree reproduces, and those sum to 193 against this field's measured median of
    #: 194-202 — consistent, and not a substitute for measuring it.
    pass_ms: dict = field(default_factory=dict)


class PerceptionWorker:
    """Runs detection and goal-fixing on its own thread, at whatever rate it manages."""

    #: Failed cycles reported individually before the log is silenced. A detector that
    #: throws on every frame would otherwise bury the console it is trying to warn.
    _ERROR_LOG_LIMIT = 5

    def __init__(self, camera, detector: PersonDetector,
                 camera_model: FisheyeCamera, goal_source: GoalSource,
                 pose_fn, prior: SizePrior = PERSON_PRIOR,
                 colour_detector: ColourBlobDetector | None = None,
                 static_prior: SizePrior | None = None,
                 static_label: str = STATIC_DETECT_LABEL) -> None:
        self._camera = camera
        self._detector = detector
        self._model = camera_model
        self._goal = goal_source
        self._pose_fn = pose_fn
        self._prior = prior
        self._colour = colour_detector
        # A MEASURED prior, or the tier stays off however the detector was built. The
        # neural tier finds an object of unknown size, and `range_detections` turns a
        # size prior into metres — with no prior there is no defensible number to put in
        # the map, and inventing one puts a landmark at a range nothing measured.
        self._static_prior = static_prior
        self._static_label = static_label
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._result: PerceptionResult | None = None
        self._cycles = 0
        self._errors = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="visual-perception",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> PerceptionResult | None:
        with self._lock:
            return self._result

    @property
    def cycles(self) -> int:
        """Completed perception cycles — how much the run actually saw."""
        return self._cycles

    @property
    def errors(self) -> int:
        """Cycles that raised. Non-zero means the belief the planner acted on was
        assembled from fewer frames than the run's duration suggests."""
        return self._errors

    def alive(self) -> bool:
        """Whether the worker thread is still running.

        The navigator checks this rather than inferring it from frame age: a dead
        thread and a stalled camera both freeze ``latest()``, but only one of them is
        worth aborting the run over.
        """
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            try:
                last_seq = self._cycle(last_seq)
            except Exception as exc:
                # A throw here used to kill this thread silently: latest() would then
                # keep returning the last good result forever, the navigator would see
                # a frame that never ages past the stale check for only one tick, and
                # the run would end as "timeout after 90s" with no hint that perception
                # had died 80 s earlier. Record it and carry on — a detector that fails
                # on one frame is usually not fatal, and a detector that fails on every
                # frame now says so.
                self._errors += 1
                if self._errors <= self._ERROR_LOG_LIMIT:
                    print(f"[perception] cycle failed: {exc!r}")
                    if self._errors == self._ERROR_LOG_LIMIT:
                        print("[perception] (further errors suppressed)")
                time.sleep(0.05)   # do not spin hot on a persistent failure

    def _cycle(self, last_seq: int) -> int:
        """One perception cycle. Returns the sequence number to wait past next."""
        wait_started = time.monotonic()
        frame = self._camera.wait_for_new(last_seq, timeout=0.5)
        wait_ms = (time.monotonic() - wait_started) * 1000.0
        if frame is None:
            return last_seq
        last_seq = frame.seq
        cycle_started = time.monotonic()
        # Pose sampled at the camera adapter boundary, not after inference — see the
        # module docstring. The adapters do not claim a sensor shutter timestamp.
        pose = frame.stamp if frame.stamp is not None else self._pose_fn()

        # EACH PASS SEPARATELY, and `detect_ms` still the sum of the three so no consumer
        # of that field sees a different number under the same name. Written out with
        # explicit marks rather than three `with` blocks because this is the hottest
        # function in the process and the marks are what the sum has to be built from.
        started = time.monotonic()
        goal_fix = self._goal.update(frame.image, pose)
        after_goal = time.monotonic()
        detections, static_detections = self._detector.detect_tiered(frame.image)
        after_detect = time.monotonic()
        # Colour segmentation costs 10.2 ms against the person detector's 114 ms, so
        # the static props ride along with the mover pass rather than earning a cadence
        # of their own. The GOAL pass is the expensive one and throttles itself.
        colour_detections = ([] if self._colour is None
                             else self._colour.detect(frame.image))
        finished = time.monotonic()
        detect_ms = (finished - started) * 1000.0
        pass_ms = {"goal": round((after_goal - started) * 1000.0, 2),
                   "detect": round((after_detect - after_goal) * 1000.0, 2),
                   "colour": round((finished - after_detect) * 1000.0, 2)}

        ranged = range_detections(detections, self._model, self._prior)
        # SHAPE RIDES ALONGSIDE THE LABEL, it does not replace it. What must stop the
        # robot is decided by `RangedDetection.person_shaped`, because this classifier
        # cannot tell a person from a robot — the peer came back as `person` on 12 of 12
        # live frames. But the label must stay the VOC class, because `_associate` gates
        # on `obs.label != track.label`: a peer whose shape verdict flipped as it clipped
        # the frame edge would fail to associate, spawn a second track, and leave the
        # first coasting with an inflating radius, which is a HOLD for a ghost.
        height, width = frame.image.shape[:2]
        observations = [
            observation_from(item.bearing_rad, item.range_m, item.source,
                             item.label, pose,
                             person_shaped=item.person_shaped(width, height))
            for item in ranged
        ]
        # Ranged with the PROP's size prior, not the person's — the two detectors find
        # different things and a shared prior would scale one of them wrongly.
        static_ranged = ([] if self._colour is None else
                         range_detections(colour_detections, self._model,
                                          self._colour.profile.prior))
        static_observations = [
            observation_from(item.bearing_rad, item.range_m, item.source,
                             item.label, pose)
            for item in static_ranged
        ]
        # The NEURAL static tier joins the colour tier here and nowhere else. It reaches
        # `static_map` and never the tracker, and the distinction is the safety argument
        # for the whole feature rather than a tidiness choice:
        #
        #   * A landmark cannot invent velocity. These detections sit near the network's
        #     noise floor, so their range jitters; the constant-velocity filter would
        #     differentiate that jitter into motion and the planner would swerve around a
        #     phantom two and a half seconds ahead of where nothing is. `static_map`'s
        #     module docstring works this through for the colour tier and every word of it
        #     applies harder to a 0.10-confidence detection.
        #   * A landmark needs TWO agreeing sightings (`CONFIRM_SIGHTINGS`) before the
        #     planner sees it at all, and expires on disagreement. That is a second,
        #     free filter on exactly the failure mode a lowered floor creates — a
        #     single-frame flicker. Measured on 216 peer-absent corridor frames, requiring
        #     a second agreeing sighting takes the gate's fire rate from 19.0% of frames
        #     to 11.6%.
        #   * A track can HOLD the robot; a landmark cannot. Everything here is routed
        #     around at policy speed, never frozen for. That is the right authority for a
        #     detection this weak — and it is also why `static_shaped` must keep people
        #     out of this list, since a person in it would stop being held for.
        neural_ranged = []
        if static_detections and self._static_prior is not None:
            neural_ranged = range_detections(static_detections, self._model,
                                             self._static_prior)
            static_observations.extend(
                observation_from(item.bearing_rad, item.range_m, item.source,
                                 self._static_label, pose)
                for item in neural_ranged
            )

        self._cycles += 1
        result = PerceptionResult(
            seq=frame.seq, capture_time=frame.capture_time, pose=pose,
            observations=observations,
            ranged=ranged + static_ranged + neural_ranged,
            static_observations=static_observations, goal_fix=goal_fix,
            image=frame.image, detect_ms=detect_ms, wait_ms=wait_ms, pass_ms=pass_ms,
            cycle_ms=(time.monotonic() - cycle_started) * 1000.0)
        with self._lock:
            self._result = result
        return last_seq


#: Ceiling on the measured tick interval handed to the planner's dynamic window. Bounds
#: how far that window may open after one slow tick: 0.5 s against the 0.50 m/s^2 forward
#: limit is 0.25 m/s of allowed change — enough to clear the Go2's gait floor from a
#: standstill, without letting a single long tick authorise the whole envelope at once.
MAX_CONTROL_DT_S = 0.5


def control_interval_s(previous_tick_s: float | None, now_s: float,
                       period_s: float) -> float:
    """Seconds the planner should size its dynamic window with.

    The window is ``accel * control_dt``, so this number decides how much the commanded
    velocity may change on this tick. Passing the NOMINAL period is only correct while
    the loop actually keeps up with it, and it does not always: measured on hardware at
    2.8 Hz against a nominal 10 Hz, the planner budgeted 0.05 m/s of change for an
    interval in which the robot could deliver 0.18. The ramp from a standstill then took
    2.5 s instead of 0.7 s, never reached the 0.35 m/s gait floor before the next
    stale-perception hold zeroed it, and the robot stood still while being commanded
    forward at full speed. The stall gate blamed the tether, which is the wrong place to
    look — see ``MIN_GAIT_COMMAND_M_S`` in ``avoidance``.

    Floored at ``period_s`` so a loop running FASTER than nominal keeps the behaviour it
    has today rather than shrinking its own window, and capped at
    :data:`MAX_CONTROL_DT_S` so a long stall does not hand the planner the whole envelope.
    """
    if previous_tick_s is None:
        return period_s
    return min(MAX_CONTROL_DT_S, max(period_s, now_s - previous_tick_s))


def blocked_stop(command) -> bool:
    """Whether this tick leaves the robot standing still because it cannot get on.

    What starts the rest-after-blocked timer, which is what puts the Go2 prone instead
    of braced. Getting it wrong is expensive and silent: this robot carries a 3.15 kg
    D1 arm on its back, a standing posture holds that arm on the leg joints for as long
    as the run lasts, and NOTHING FAULTS while it does. Both stall gates decline a zero
    command by construction — :meth:`VisualNavigator._blocked_reason` returns ``None``
    below :data:`PROGRESS_MIN_COMMAND_M_S` ("not asking it to go anywhere") and
    ``mappo_drive._note_sub_floor`` returns on ``speed <= 0.0`` ("a stop is a stop") —
    so a robot that never lies down also never says why, and the run ends on its own
    clock as a timeout.

    TWO THINGS THIS TEST USED TO BE, AND IS NOT.

    **Not ``command.reason == "hold"``.** ``integration/mappo_drive.py`` QUALIFIES a
    vetoed tick's reason rather than replacing it, so the planner's own hold arrives
    here as ``"veto-hold"`` and the equality never matched: every policy-driven run
    stood braced for the rest of its budget. :func:`avoidance.base_reason` recovers the
    planner's word, and it strips one qualifier rather than searching for a substring —
    ``"hold" in reason`` would also fire on a reason that merely contains the letters,
    which is a worse bug than the one it fixes.

    **Not the label alone either.** Measured offline on the drive path: a stub policy
    walking at a bin 2.6 m ahead is vetoed at 1.11 m of surface gap and from 1.06 m the
    command is ``v=(0.00, 0.00, 0.00)`` on every remaining tick — labelled
    ``veto-avoid``, not ``veto-hold``, because the stopping-distance cap had ratcheted
    the dynamic window down until the best sampled candidate was zero and ``avoid`` is a
    label applied after that choice. ``base_reason(...) == "hold"`` is False on all of
    them, and a robot that has stopped dead is exactly what the timer is for.
    ``avoidance._window``'s own comment reached the same conclusion from the other
    direction, about a stop labelled ``goal``: "it stands there under load until the run
    times out". So does ``_coast_budget_s`` in ``test_visual_nav``: "Timing the label
    rather than the wheels made this budget read ~50% longer than the robot actually
    managed."

    So the test is what the LEGS are told, plus the one reason a stop is not a failure
    to get on: ``arrived``. Anything else that stops the robot rests it, which is the
    safe direction — lying down costs one 3 s stand-up on the next moving tick, and
    bracing costs the remainder of the run. The qualifier is stripped off ``arrived``
    too, defensively: nothing produces ``veto-arrived`` today, because ``mappo_drive``
    returns ``arrived`` from a branch above its veto, and the point of reading it
    through :func:`avoidance.base_reason` is that the next wrapper cannot re-open this.
    """
    return command.is_stop and base_reason(command.reason) != "arrived"


class VisualNavigator:
    """Drives a quadruped to a goal on camera alone through injected robot bindings."""

    def __init__(self, loco, perception: PerceptionWorker,
                 planner: DynamicWindowPlanner, tracker: ObstacleTracker,
                 goal_source: GoalSource, health, config: NavConfig,
                 recorder: cv2.VideoWriter | None = None,
                 static_map: StaticObstacleMap | None = None,
                 telemetry: TelemetryWriter | None = None,
                 stand_up_fn=None, lie_down_fn=None,
                 expansion: ExpansionConsistency | None = None,
                 raw_recorder: cv2.VideoWriter | None = None,
                 profiler: TickProfiler | None = None) -> None:
        self._loco = loco
        self._perception = perception
        self._planner = planner
        self._tracker = tracker
        self._goal = goal_source
        self._health = health
        self._config = config
        self._recorder = recorder
        #: Undecorated sibling of ``recorder``. Keyword-only in effect, and LAST in the
        #: signature on purpose: ``integration/mappo_drive.py`` calls this constructor
        #: through ``navigator_factory`` with the positional arguments it already knows,
        #: and says in its own docstring that a new positional one must not be added.
        self._raw_recorder = raw_recorder
        self._static_map = static_map
        self._telemetry = telemetry
        # Read in _obstacles only. Passed separately from the tracker it is also
        # attached to, because the two use it for different things and the navigator
        # must not reach into the tracker's private state to find it.
        self._expansion = expansion
        self._stand_up_fn = stand_up_fn or self._default_stand_up
        self._lie_down_fn = lie_down_fn or self._default_lie_down
        #: Always present, never optional. A profiler behind a flag is off during every
        #: run anyone later wants to explain — issue #18 spent eight days on a loop that
        #: had run a hundred times and never been timed. Injectable only so the tests can
        #: drive it from a fake clock. LAST in the signature for the same reason
        #: ``raw_recorder`` is: ``integration/mappo_drive.py`` builds this class through
        #: ``navigator_factory`` and says in its own docstring that no positional argument
        #: may be added.
        self._profiler = profiler if profiler is not None else TickProfiler()

        self._standing = config.initially_standing and config.live
        self._frames_written = 0
        #: (time, x, y) over the last PROGRESS_WINDOW_S, for the stall gate.
        self._progress: deque = deque()
        self._last_command = (0.0, 0.0, 0.0)
        self._last_reason = "goal"
        self._tracker_time = time.monotonic()
        self._consumed_seq = 0
        self._recorded_seq = 0

    # ── Posture ─────────────────────────────────────────────────────────────
    @staticmethod
    def _default_stand_up(loco) -> None:
        from safety import stand_up

        stand_up(loco)

    @staticmethod
    def _default_lie_down(loco) -> None:
        from safety import lie_down

        lie_down(loco)

    def _stand_up(self) -> None:
        """Stand, tracking the posture. Blocks ~3 s — callers must re-plan afterwards."""
        if self._standing or not self._config.live:
            self._standing = True
            return
        print("[visual_nav] standing up")
        self._stand_up_fn(self._loco)
        self._standing = True

    def _lie_down(self) -> None:
        if not self._config.live:
            self._standing = False
            return
        self._lie_down_fn(self._loco)
        self._standing = False

    def park(self) -> None:
        """Stop and lie down. Idempotent; safe from any exit path."""
        if self._standing:
            print("[visual_nav] parking: stop + platform safe-rest action")
            self._lie_down()
        elif self._config.live:
            with contextlib.suppress(Exception):
                self._loco.stop()

    # ── Obstacles ───────────────────────────────────────────────────────────
    def _obstacles(self, now: float) -> list[Obstacle]:
        """Confirmed tracks, extrapolated to NOW and inflated for their uncertainty.

        The filter runs in measurement time (the newest measurement is ~160 ms old),
        so positions are advanced on their estimated velocity to the present. The
        radius absorbs both the filter's own position sigma and the extra error that
        extrapolation could have introduced if the person accelerated.

        THE INTERVAL IS THE PERCEPTION LATENCY, NOT THE AGE OF THE TRACK. ``predict()``
        runs over every track on every perception cycle, so an unobserved track has
        already been advanced — and its covariance already grown — right up to
        ``_tracker_time``. Extrapolating from ``track.last_seen`` instead integrates
        the coast a second time, on top of a sigma that has counted it once: measured,
        a person who simply walked out of shot reached an 11 m radius inside the 3 s
        coast window, and the robot held for a ghost 4 m away and 20 deg off its path.

        WHAT IS STILL NOT FIXED, and why it is left alone. Halving the radius moves the
        stop out but does not remove it: the remainder is ``position_sigma``, the
        filter's own honest account of a target it has not measured for a second or
        more. Capping that would be inventing a number, so the behaviour is documented
        as an operating limit (see the module docstring) rather than tuned from a desk.
        The live run is what should decide whether a track this uncertain belongs in
        the obstacle set at all — note that once inflated it stops carrying direction
        as well as position, so a person off to one side blocks exactly as hard as one
        dead ahead.
        """
        latency = max(0.0, now - self._tracker_time)
        extrapolation_sigma = 0.5 * PROCESS_ACCEL_SIGMA * latency * latency
        confirmed = self._tracker.confirmed_tracks()
        # The expansion gate, where one is fitted, subtracts here and NOWHERE ELSE.
        # A rejected track keeps being predicted, associated and corrected, so it keeps
        # accumulating the evidence that could restore it; what it stops doing is
        # constraining the plan. Deleting it in the tracker would throw that evidence
        # away and let the same box respawn as a new id with a clean sheet, which is
        # the loop `static_map`'s MAX_MISSES was written to break.
        rejected = (set() if self._expansion is None
                    else self._expansion.rejects(t.track_id for t in confirmed))
        obstacles = [
            Obstacle(x=float(track.state[0] + track.state[2] * latency),
                     y=float(track.state[1] + track.state[3] * latency),
                     vx=float(track.state[2]), vy=float(track.state[3]),
                     radius_m=(self._planner.config.obstacle_radius_m
                               + track.position_sigma + extrapolation_sigma),
                     label=track.label, person_shaped=track.person_shaped,
                     kind="tracked", object_id=f"track-{track.track_id}")
            for track in confirmed if track.track_id not in rejected
        ]
        # Landmarks need none of the above. They are not extrapolated because they do
        # not move, and their radius carries no latency term for the same reason — the
        # only uncertainty in a bin's position is where the robot thinks IT is.
        if self._static_map is not None:
            obstacles.extend(
                Obstacle(x=landmark.x, y=landmark.y, vx=0.0, vy=0.0,
                         radius_m=landmark.planning_radius_m, label=landmark.label,
                         person_shaped=False, soft_gap_m=STATIC_SOFT_GAP_M,
                         hard_gap_m=STATIC_HARD_GAP_M,
                         kind="static", object_id=f"landmark-{landmark.landmark_id}")
                for landmark in self._static_map.confirmed())
        return obstacles

    # ── Main loop ───────────────────────────────────────────────────────────
    def run(self) -> str:
        config = self._config
        period = 1.0 / config.control_hz
        started = time.monotonic()
        last_tick: float | None = None
        hold_since: float | None = None
        outcome = "unknown"

        print(f"[visual_nav] goal: {self._goal.description}")
        print(f"[visual_nav] {'LIVE — legs will move' if config.live else 'DRY RUN — no motion'}")

        while True:
            tick_start = time.monotonic()
            # Against the loop's OWN t0, not the profiler's, so `tick_ms` and the trailing
            # sleep are measured from the same instant.
            self._profiler.begin(tick_start)
            now = tick_start
            elapsed = now - started
            control_dt = control_interval_s(last_tick, now, period)
            last_tick = now

            if elapsed > config.max_run_s:
                outcome = f"timeout after {elapsed:.0f}s"
                break

            blocked = self._health.abort_reason()
            if blocked is not None:
                outcome = f"health abort: {blocked}"
                break

            # A dead perception thread freezes latest() at its last good result, which
            # then ages past the stale check and holds the robot for the rest of the
            # budget under the wrong diagnosis. Say what actually happened.
            if not self._perception.alive():
                outcome = (f"perception thread died after {self._perception.cycles} "
                           f"cycles ({self._perception.errors} errors)")
                break

            with self._profiler.stage("perceive"):
                result = self._perception.latest()
            if result is None:
                if elapsed > 5.0:
                    outcome = "no perception result"
                    break
                self._sleep_out_the_period(tick_start, period)
                continue

            # A second `perceive` block rather than one around both: the `result is None`
            # branch above has to `continue` from between them. `stage` ADDS on re-entry
            # precisely so a stage split across a branch is not reduced to its last part.
            with self._profiler.stage("perceive"):
                if result.seq > self._consumed_seq:
                    self._tracker.predict(
                        max(0.0, result.capture_time - self._tracker_time))
                    # Static props first: the map they build is what tells the tracker a
                    # person who vanished did so BEHIND something rather than by leaving.
                    # Both consume the same cycle's pose, so the order is bookkeeping, not
                    # a one-frame lag.
                    occluders = ()
                    if self._static_map is not None:
                        self._static_map.observe(result.static_observations,
                                                 result.capture_time, *result.pose)
                        occluders = tuple(self._static_map.occluders(*result.pose))
                    self._tracker.update(result.observations, result.capture_time,
                                         *result.pose, occluders=occluders)
                    self._tracker_time = result.capture_time
                    self._consumed_seq = result.seq

            # Pose and obstacles are read BEFORE the branches below, not inside the one
            # that happens to need them. Every path through this loop is a tick that a
            # consumer has to be able to see — a stale-perception skip and a goal search
            # are as much a part of the episode as a stride — and the plan-view inset
            # went blank on exactly the paths that used to skip this.
            with self._profiler.stage("obstacles"):
                pose_obj = self._loco.pose()
                pose = (pose_obj.x, pose_obj.y, pose_obj.yaw)
                obstacles = self._obstacles(now)

            frame_age = now - result.capture_time
            if frame_age > config.perception_timeout_s:
                # Blind. Stop rather than coast on a stale belief. Odometry is a DDS
                # topic and is still good here, so the tick is still worth recording.
                self._command((0.0, 0.0, 0.0))
                print(f"[visual_nav] perception stale ({frame_age:.2f}s) — holding")
                # The LATCHED goal, not None. A stale frame means the robot cannot see,
                # not that it has forgotten where it was going, and recording null here
                # reads downstream as "goal lost" — which is a different and much more
                # alarming event. `stale` is what says what actually happened.
                latched = self._goal.goal_xy()
                self._telemetry_tick(
                    elapsed, pose, latched,
                    None if latched is None else
                    math.hypot(latched[0] - pose[0], latched[1] - pose[1]),
                    obstacles, frame_age, result, stale=True)
                self._sleep_out_the_period(tick_start, period)
                continue

            goal_xy = self._goal.goal_xy()
            if goal_xy is None:
                if elapsed > config.goal_search_s:
                    outcome = f"goal never sighted in {config.goal_search_s:.0f}s"
                    break
                self._command((0.0, 0.0, 0.0))
                frame = self._record(result, pose, None, obstacles)
                self._telemetry_tick(elapsed, pose, None, None, obstacles,
                                     frame_age, result, video_frame=frame)
                self._sleep_out_the_period(tick_start, period)
                continue

            distance = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
            if distance <= config.arrive_tolerance_m:
                outcome = f"arrived ({distance:.2f} m from goal)"
                self._command((0.0, 0.0, 0.0))
                frame = self._record(result, pose, None, obstacles)
                self._telemetry_tick(elapsed, pose, goal_xy, distance, obstacles,
                                     frame_age, result, video_frame=frame)
                break

            with self._profiler.stage("plan"):
                command = self._planner.plan(pose, goal_xy, self._last_command,
                                             obstacles, control_dt=control_dt,
                                             last_reason=self._last_reason)
            self._last_reason = command.reason

            # Rest the legs whenever the way stays blocked — the arm makes standing
            # expensive, so a long hold is spent lying down rather than braced. What
            # counts as blocked is `blocked_stop`, which is a named function and not an
            # expression here because the two mistakes it is avoiding both LOOK right
            # inline; read it before changing this.
            if blocked_stop(command):
                # `hold_since or now` would restart the timer on a monotonic clock that
                # happened to read 0.0. Vanishingly unlikely, but the explicit test is
                # the same length and says what it means.
                hold_since = now if hold_since is None else hold_since
                if (config.rest_when_blocked and self._standing
                        and now - hold_since >= config.rest_after_s):
                    print(f"[visual_nav] blocked {now - hold_since:.0f}s — resting prone")
                    self._lie_down()
            else:
                hold_since = None
                if not self._standing:
                    self._stand_up()
                    # Standing blocks this loop for ~3 s, so `command` is now a plan
                    # made three seconds ago — made, in fact, while the robot was
                    # still lying down and the person it is about to walk past was
                    # three seconds further back. Driving off on it is the one moment
                    # in the run where the commanded velocity is guaranteed stale.
                    # Give up this tick and re-plan on the next one, 100 ms later.
                    self._command((0.0, 0.0, 0.0))
                    frame = self._record(result, pose, None, obstacles)
                    self._telemetry_tick(elapsed, pose, goal_xy, distance, obstacles,
                                         frame_age, result, video_frame=frame)
                    continue

            self._command((command.vx, command.vy, command.wz)
                          if self._standing else (0.0, 0.0, 0.0))

            # COMMANDED TO MOVE AND NOT MOVING. Go2Locomotion has measured this all
            # along — is_blocked() compares the odometry's speed against the commanded
            # one over a 3 s grace — and nothing here ever asked it. Observed live: the
            # Ethernet tether went taut, the robot veered into a cubicle wall, and the
            # loop kept commanding 0.13 m/s forward and 0.20 m/s of strafe into it for
            # FORTY SECONDS while the pose sat still. Nothing stopped it, and the run
            # ended on its clock as "timeout".
            #
            # It is also what poisons everything downstream, so a late abort is not a
            # cosmetic improvement. Odometry that says "stationary" while the robot is
            # physically being dragged projects every sighting to a wrong odom point:
            # that run finished with FOUR landmarks for one bin, up to 5.6 m from it,
            # and a goal that jumped a metre between latches. The map is only as good as
            # the frame under it.
            blocked = self._blocked_reason(command, now, pose)
            if blocked is not None:
                outcome = blocked
                break

            self._log(elapsed, command, distance, obstacles, frame_age)
            frame = self._record(result, pose, command, obstacles)
            self._telemetry_tick(elapsed, pose, goal_xy, distance, obstacles,
                                 frame_age, result, command=command,
                                 video_frame=frame)

            self._sleep_out_the_period(tick_start, period)

        # How much the run actually perceived, so a short or empty run can be told
        # apart from a long blind one when reading the log afterwards.
        print(f"[visual_nav] perception: {self._perception.cycles} cycles, "
              f"{self._perception.errors} errors")
        print(f"[visual_nav] outcome: {outcome}")
        if self._telemetry is not None:
            self._telemetry.write_outcome(
                outcome, perception_cycles=self._perception.cycles,
                perception_errors=self._perception.errors,
                elapsed_s=round(time.monotonic() - started, 3))
        return outcome

    def _sleep_out_the_period(self, tick_start: float, period: float) -> None:
        """Sleep whatever is left of this tick's period. Every path through the loop.

        🔴 TWO OF THE FOUR SLEEPS USED TO BE A FLAT ``period``. The stale-perception hold
        and the no-result skip slept a whole one; the goal-search and walking paths slept
        ``max(0, period - elapsed)``. On a tick that has already spent 250 ms — the median
        of the two 2026-08-25 runs — the two normal paths sleep 0 and the two hold paths add
        another 100 ms on top. So the branch that fires BECAUSE the loop is slow cost a
        period more than the branch that does not, which is a small positive feedback on
        exactly the ticks issue #18 is about.

        ⚠️ Not a fix for the rate, and not offered as one: staleness fired on 0 of the 150
        ticks of those two runs, so on them this changes nothing measurable. On 2026-08-17,
        where 16-33% of ticks were stale, it removes up to a period from each of those. What
        it removes for certain is a tick cost that depended on which ``if`` it came through
        — and now that ``TickProfiler`` reports ``tick_ms``, that difference would have been
        attributed to the hold rather than to the sleep.
        """
        time.sleep(max(0.0, period - (time.monotonic() - tick_start)))

    def _blocked_reason(self, command, now: float, pose) -> str | None:
        """``None``, or why the run should stop because the robot is going nowhere.

        Only meaningful while actually walking: a dry run's legs never move, and a prone
        robot is stationary on purpose.

        MEASURED ON NET DISPLACEMENT, not on instantaneous speed, and that is the whole
        point of doing it here rather than deferring to ``Go2Locomotion.is_blocked``.
        That gate asks whether the measured speed is at least a tenth of the commanded
        one — which for a 0.10 m/s command is **one centimetre per second**, a bar a
        robot shuffling on the spot clears easily. It did: across two live runs the pose
        sat still for 21 s and then 12 s while the loop commanded 0.10-0.14 m/s forward
        and a full 0.20 m/s of strafe, and the gate never fired once.

        A quadruped that is trotting hard against a taut tether IS moving its legs, and
        its body velocity estimate is not zero. What it is not doing is getting anywhere.
        So compare where it actually IS against where it was, over a window long enough
        that a legitimate slow manoeuvre — a spot turn, a careful sidestep — is not
        mistaken for it.
        """
        if not (self._config.live and self._standing):
            self._progress.clear()
            return None

        self._progress.append((now, pose[0], pose[1]))
        # Keep the NEWEST sample that is still at least a full window old, by discarding
        # a sample only once the one behind it has itself aged past the window. Pruning
        # everything older than the window instead — the obvious spelling — leaves the
        # oldest survivor younger than the window by construction, so the "have I got a
        # full window yet?" test below can only pass on a tick landing exactly on the
        # boundary. At an irregular 10 Hz that never happens, and the gate silently never
        # fires: a live run sat still for 12 s, commanded 0.23 m/s throughout, and was
        # never once judged. A unit test with dt=0.5 hid it, because 0.5 divides 4.0.
        while len(self._progress) > 1 and now - self._progress[1][0] >= PROGRESS_WINDOW_S:
            self._progress.popleft()

        commanded = math.hypot(command.vx, command.vy)
        if commanded <= PROGRESS_MIN_COMMAND_M_S:
            return None                       # not asking it to go anywhere
        oldest_time, oldest_x, oldest_y = self._progress[0]
        span = now - oldest_time
        if span < PROGRESS_WINDOW_S:
            return None                       # not enough history to judge yet

        moved = math.hypot(pose[0] - oldest_x, pose[1] - oldest_y)
        expected = commanded * span
        if moved >= PROGRESS_FRACTION * expected:
            return None
        return (f"stalled: commanded {commanded:.2f} m/s for {span:.1f}s and moved "
                f"{moved:.2f} m of an expected {expected:.2f} m. Something is holding "
                f"the robot — check the tether, and whether it has walked into "
                f"something this pipeline cannot see.")

    def _telemetry_tick(self, elapsed: float, pose, goal_xy, distance,
                        obstacles: Sequence[Obstacle], frame_age: float,
                        result: PerceptionResult, command=None,
                        video_frame: int | None = None, stale: bool = False) -> None:
        """Record one tick for a machine, whether or not it commanded motion.

        Separate from ``_log``, which is for a person reading a console. The two have
        genuinely different obligations: the console line is edited freely to make a run
        legible (``people=0`` became ``obst=[binx1,personx1]`` the week a mapped bin
        stopped being distinguishable from a ghost), while this one is a versioned
        contract someone else parses.
        """
        if self._telemetry is None:
            return
        # The snapshot is taken as an ARGUMENT, i.e. before the write starts, and the
        # write's own cost is handed to the profiler for the next tick to carry: a record
        # cannot contain the time it took to write itself.
        started = time.monotonic()
        self._telemetry.write_tick(
            elapsed_s=elapsed, pose=pose, goal_xy=goal_xy,
            goal_distance_m=distance, command=command, obstacles=obstacles,
            frame_age_s=frame_age, perception_seq=result.seq,
            detect_ms=result.detect_ms, standing=self._standing,
            live=self._config.live, video_frame=video_frame, stale=stale,
            measured=self._measured_velocity(), health=self._health.latest(),
            sightings=result.ranged, goal_crop=self._goal_crop(),
            profile=self._profiler.snapshot(), cycle_ms=result.cycle_ms,
            wait_ms=result.wait_ms, pass_ms=result.pass_ms)
        self._profiler.wrote((time.monotonic() - started) * 1000.0)

    def _goal_crop(self) -> float | None:
        """The crop the goal source used last, or ``None`` if it does not have one.

        Guarded the same way as :meth:`_measured_velocity`, and for the same reason: not
        every goal source crops — an ArUco marker or a fixed waypoint has no such notion
        — and a telemetry field must never be the thing that ends a run.
        """
        try:
            return float(self._goal.last_crop)
        except Exception:
            return None

    def _measured_velocity(self):
        """What the odometry says the body is doing, or ``None`` if it cannot say.

        Guarded because not every locomotion backend exposes it, and a telemetry field
        must never be the thing that ends a run.
        """
        try:
            return self._loco.velocity()
        except Exception:
            return None

    def _command(self, velocity: tuple[float, float, float]) -> None:
        # Timed HERE and not at the five call sites: every path through the loop commands
        # something, including the ones that command a stop, and a stage wired in at four
        # of five places reads as a cheap transport.
        with self._profiler.stage("command"):
            if self._config.live and self._standing:
                self._loco.set_velocity(*velocity)
        self._last_command = velocity

    def _log(self, elapsed: float, command, distance: float,
             obstacles: list[Obstacle], frame_age: float) -> None:
        health = self._health.latest()
        temperature = f"{health.max_motor_temp_c:.0f}C" if health else "?"
        gap = "inf" if math.isinf(command.gap_m) else f"{command.gap_m:.2f}m"
        # In a dry run the posture is simulated, so say so rather than printing
        # "STAND" next to a robot that is plainly lying on the floor.
        posture = "prone"
        if self._standing:
            posture = "STAND" if self._config.live else "stand(sim)"
        # Count by label rather than printing a bare total. "obst=2" cannot distinguish
        # the run working (bin mapped, nobody about) from the run stopping for a ghost,
        # and that is the first question asked of every log line here.
        tally = Counter(obstacle.label for obstacle in obstacles)
        seen = ",".join(f"{label}x{n}" for label, n in sorted(tally.items())) or "none"
        print(f"[{elapsed:5.1f}s] {command.reason:<7} "
              f"v=({command.vx:+.2f},{command.vy:+.2f},{command.wz:+.2f}) "
              f"goal={distance:.2f}m gap={gap} obst=[{seen}] "
              f"lat={frame_age * 1000:.0f}ms motor={temperature} {posture}")

    def _record(self, result: PerceptionResult, pose, command,
                obstacles: Sequence[Obstacle] = ()) -> int | None:
        """Write one frame to each open recorder — once per PERCEPTION cycle, not per tick.

        The control loop runs faster than perception, so recording per tick would
        duplicate frames and make the video play back faster than the run happened.

        Returns the 0-based index of the frame written, or ``None`` on a tick that wrote
        none. That index is the join key between the telemetry file and the MP4, and it
        is why this returns anything at all — it is the only honest way to answer "what
        did the camera see at this tick?" without putting pixels in a log.

        **Two recorders, one index.** ``--record`` writes the annotated frame — the HUD,
        the plan-view inset and a box around every detection — and that is what makes it
        readable and what makes it useless as training data: the label is burned into the
        pixels the model would have to learn from. ``--record-raw`` writes the same frame
        before any of that is drawn on it. Both are advanced by this one gate, so frame
        *n* of one file is frame *n* of the other and both join to the same
        ``perception.video_frame`` in the telemetry. Counting them separately would have
        been the bug: the moment one recorder skipped a cycle the other's indices would
        silently mean a different thing.
        """
        with self._profiler.stage("record"):
            return self._write_frame(result, pose, command, obstacles)

    def _write_frame(self, result: PerceptionResult, pose, command,
                     obstacles: Sequence[Obstacle] = ()) -> int | None:
        """The body of :meth:`_record`, split out only so the stage timer wraps every
        early return as well as the encode. A timer around the encode alone would report
        the four ticks in five that write nothing as free, which they are — and would
        leave the fifth's overlay drawing outside the number."""
        if (self._recorder is None and self._raw_recorder is None) or result.image is None:
            return None
        if result.seq <= self._recorded_seq:
            return None
        self._recorded_seq = result.seq
        # BEFORE anything is drawn. `result.image` is never mutated here — the annotated
        # path works on a copy — so this is the frame the camera produced, not a frame
        # something has been rubbed off.
        if self._raw_recorder is not None:
            self._raw_recorder.write(result.image)
        self._frames_written += 1
        if self._recorder is None:
            return self._frames_written - 1
        canvas = result.image.copy()
        overlay.draw_detections(canvas, result.ranged)
        overlay.draw_goal(canvas, result.goal_fix)
        overlay.draw_plan_view(canvas, pose, obstacles, self._goal.goal_xy(), command)
        health = self._health.latest()
        if self._standing:
            posture = "STANDING" if self._config.live else "STANDING(sim)"
        else:
            posture = "PRONE"
        overlay.draw_status(canvas, [
            f"{'LIVE' if self._config.live else 'DRY RUN'}  {posture}  "
            f"det {result.detect_ms:.0f}ms",
            (f"cmd {command.reason} v=({command.vx:+.2f},{command.vy:+.2f},"
             f"{command.wz:+.2f})" if command is not None else "cmd -"),
            (f"motor {health.max_motor_temp_c:.0f}C  batt {health.battery_soc_pct:.0f}%"
             if health else "health -"),
        ])
        self._recorder.write(canvas)
        return self._frames_written - 1


def build_camera_model(width: int, height: int,
                       calibration: str | None) -> FisheyeCamera:
    """Load a calibrated model, or fall back to the nominal field of view.

    THE LENS HEIGHT IS CHECKED HERE, before anything moves. `camera.height_m` bounds a
    width-derived range in `person_detector.object_fit_range`, which refuses without it —
    and the first horizontally-clipped detection can be several metres into a live run.
    A refusal that can only fire mid-run is a robot that stops walking because the process
    died, so this reads the field at start-up instead. See `robot-stack/SAFETY.md`.
    """
    if calibration:
        model = FisheyeCamera.load(calibration)
        if (model.width, model.height) != (width, height):
            model = model.scaled(width, height)
        if model.height_m is None:
            raise SystemExit(
                f"[visual_nav] REFUSING TO RUN: {calibration} states no \"height_m\". "
                f"calibrate_camera.py fits the focal length and cannot measure a lens "
                f"height, so the file it writes carries none until somebody adds it. "
                f"Measure floor to lens centre with the robot STANDING and put it in the "
                f"file; the Go2's is {GO2_CAMERA_HEIGHT_M} m, and for a Lite3 run "
                f"deep_robotics/lite3/commissioning/camera_calibration.py --lens-height, "
                f"which stamps it and records what it replaced. It bounds a width-derived "
                f"range estimate, so it is not cosmetic."
            )
        print(f"[visual_nav] camera model: {calibration} "
              f"(f={model.focal_px:.1f}px, HFOV={model.hfov_deg:.1f}deg, "
              f"lens height {model.height_m:.3f}m)")
        return model
    # Every intrinsic on this path is the GO2's published nominal spec already —
    # DEFAULT_HFOV_DEG is the Go2 front fisheye's — so the Go2's lens height is the
    # consistent companion to it, and naming it here is what keeps it out of the shared
    # model's defaults. The print says whose numbers these are, because on any other
    # robot all of them are wrong.
    model = FisheyeCamera.from_hfov(width, height, height_m=GO2_CAMERA_HEIGHT_M)
    # Read back through the accessor rather than formatted straight out of the field.
    # The branch above has its own refusal, which is the better of the two because it can
    # name the file; this branch has no file to name, and formatting the field directly
    # would report a dropped `height_m=` as "unsupported format string passed to
    # NoneType.__format__", which names nothing at all.
    nominal_height_m = model.require_height_m("the nominal model has no floor either")
    print(f"[visual_nav] camera model: NOMINAL HFOV={model.hfov_deg:.1f}deg and lens "
          f"height {nominal_height_m:.2f}m, both the GO2's — ranges and bearings are "
          f"un-calibrated, and on another robot the height is wrong too. "
          f"Run calibrate_camera.py.")
    return model


def static_profile(args) -> ColourProfile:
    """Resolve an evidence-backed custom profile or a named profile with measured overrides.

    Overrides are ``replace`` on a frozen dataclass rather than mutation, so a profile
    is never edited in place — ``PROFILES`` is module-level and shared, and one run
    quietly rewriting the bin's height for every later one is the kind of bug that only
    shows up as ranges being wrong in the NEXT session.
    """
    custom_profile = getattr(args, "static_profile", None)
    if custom_profile is not None:
        if args.prop_height is not None or args.prop_radius is not None:
            raise SystemExit(
                "[visual_nav] --static-profile contains evidenced dimensions; do not override "
                "them with --prop-height or --prop-radius"
            )
        try:
            return load_colour_profile(custom_profile)
        except ValueError as error:
            raise SystemExit(f"[visual_nav] REFUSING TO USE STATIC PROFILE: {error}") from None

    profile = PROFILES[args.static_prop]
    changes = {}
    if args.prop_height is not None:
        changes["height_m"] = args.prop_height
        # Width is the fallback prior once the base drops out of frame at close
        # quarters. Keeping the profile's measured aspect ratio is the only defensible
        # thing to do with it when only the height was re-measured.
        changes["width_m"] = profile.width_m * (args.prop_height / profile.height_m)
    if args.prop_radius is not None:
        changes["radius_m"] = args.prop_radius
    return replace(profile, **changes) if changes else profile


def static_detect_prior(args) -> tuple[SizePrior, float] | None:
    """``(prior, radius_m)`` for the neural static tier, or ``None`` when it is off.

    Split out of ``main`` for the reason ``static_profile`` and ``build_goal_source``
    are: this is where a flag combination is REFUSED, and a test should be able to check
    the refusal without a robot, a camera or a DDS stack.

    ⚠️ BOTH DIMENSIONS ARE REQUIRED AND NEITHER IS INFERRED. ``SizePrior.of_height``
    will happily fill a width from a standing adult's aspect ratio, and its own docstring
    records what that cost the last time it was relied on: 0.514 m of peer implied
    0.151 m against a real 0.31 m, and clipped boxes then ranged the peer at 0.09-0.14 m —
    inside the robot's own footprint, which the planner reads as an unavoidable collision.
    A cardboard box is squatter than a person, so the same inference would be wronger. The
    operator staged the prop; they can measure it.
    """
    if not getattr(args, "static_detect", False):
        return None
    missing = [name for name, value in (("--static-detect-height", args.static_detect_height),
                                        ("--static-detect-width", args.static_detect_width))
               if value is None]
    if missing:
        raise SystemExit(
            f"[visual_nav] --static-detect needs {' and '.join(missing)}: the detector "
            f"finds an object of unknown size and every range scales linearly on the "
            f"prior it is given")
    bad = [name for name, value in (("--static-detect-height", args.static_detect_height),
                                    ("--static-detect-width", args.static_detect_width),
                                    ("--static-detect-radius", args.static_detect_radius))
           if value is not None and not (math.isfinite(value) and value > 0.0)]
    if bad:
        raise SystemExit(
            f"[visual_nav] {', '.join(bad)} must be a positive number of metres")
    radius = (args.static_detect_radius if args.static_detect_radius is not None
              else args.static_detect_width / 2.0)
    return (SizePrior(height_m=float(args.static_detect_height),
                      width_m=float(args.static_detect_width)), float(radius))


def static_profile_telemetry(args, profile: ColourProfile | None) -> dict | None:
    """Return custom-profile provenance without recording its local filename."""
    custom_profile = getattr(args, "static_profile", None)
    if custom_profile is None or profile is None:
        return None
    try:
        digest = hashlib.sha256(Path(custom_profile).read_bytes()).hexdigest()
    except OSError as error:
        raise SystemExit(
            f"[visual_nav] REFUSING TO RECORD STATIC PROFILE: cannot read {custom_profile}: "
            f"{error}"
        ) from None
    return {
        "schema": COLOUR_PROFILE_SCHEMA,
        "sha256": digest,
        "label": profile.label,
        "evidence": dict(profile.evidence),
    }


def build_goal_source(args, camera_model: FisheyeCamera, pose_fn) -> GoalSource:
    """Pick the goal source the flags asked for, newest-to-oldest in specificity.

    Split out of ``main`` for the same reason ``build_parser`` is: it is the one place
    the mutually-exclusive goal flags are resolved, and a test can check that resolution
    without a robot, a camera or a DDS stack.
    """
    if args.waypoint is not None:
        return OdomWaypoint(pose_fn(), args.waypoint[0], args.waypoint[1])
    if args.goal_class is not None:
        if args.goal_height is None:
            raise SystemExit(
                "[visual_nav] --goal-class needs --goal-height: the range to the goal "
                "scales linearly on it, so there is no safe default to guess.")
        # A SECOND detector instance, not the obstacle one. It wants a different class
        # and a different confidence, and it runs on a crop; sharing would mean the
        # obstacle pass paid the goal's threshold or vice versa.
        goal_detector = PersonDetector(args.model_dir, input_size=args.goal_input_size,
                                       confidence=args.goal_confidence,
                                       classes=(args.goal_class,))
        return DetectedObjectGoal(camera_model, goal_detector, label=args.goal_class,
                                  height_m=args.goal_height, width_m=args.goal_width,
                                  crop=args.goal_crop, refresh_s=args.goal_refresh,
                                  min_score=args.goal_confidence)
    return ArucoGoal(camera_model, marker_id=args.marker_id,
                     marker_size_m=args.marker_size,
                     detect_scale=args.goal_detect_scale)


def warn_if_below_gait_floor(max_vx: float) -> bool:
    """Backward-compatible public entry point for the measured Go2 gait warning."""
    return warn_if_below_go2_gait_floor(max_vx)


def build_parser(bindings=None) -> argparse.ArgumentParser:
    """The CLI, separated from ``main`` so it can be exercised without a robot.

    Same split as ``calibrate_camera.build_parser``. It is what lets a test assert
    that the D1 latch is on by default, which is a safety property rather than a
    preference.
    """
    bindings = bindings or Go2Bindings()
    ap = argparse.ArgumentParser(
        description=f"Walk {bindings.platform_name} to a goal on RGB alone, avoiding people.")
    # Defaults are read from the dataclasses rather than repeated here as literals,
    # which would be two places to change and one to forget: the dataclass is what a
    # caller constructing this in-process gets, so the CLI agrees with it by
    # construction rather than by inspection.
    limits, nav, planner = Limits(), NavConfig(), PlannerConfig()

    ap.add_argument("--live", action="store_true",
                    help="DANGER: actually move the legs. Without this the robot "
                         "perceives and plans but never leaves the floor.")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                    help="directory holding the MobileNet-SSD files")
    ap.add_argument("--calibration", default=None,
                    help="camera model JSON from calibrate_camera.py")
    ap.add_argument("--input-size", type=int, default=300,
                    help="detector input size (300 accurate, 224 ~1.7x faster)")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                    help="minimum detection score (lower = fewer misses, more stops)")
    ap.add_argument("--classes", nargs="+", default=list(DYNAMIC_CLASSES),
                    metavar="VOC_CLASS",
                    help="what counts as a moving obstacle. Defaults to person. Adding "
                         "e.g. 'bottle' lets a hand-slid object stand in for a walking "
                         "person when no volunteer is available — but the size priors "
                         "in camera_model are a PERSON's, so any other class is ranged "
                         "with the wrong prior unless you say what it is")
    ap.add_argument("--expansion-filter", action="store_true",
                    help="withhold a confirmed track from the planner when its range "
                         "does not fall as fast as the robot's own odometry demands — "
                         "i.e. it is further off than its size prior claims. OFF by "
                         "default: the gate's noise floor is measured but it has never "
                         "run against a moving robot, so it is opt-in until a walking "
                         "sequence says what it catches. See expansion.py")
    ap.add_argument("--obstacle-height", type=float, default=None,
                    help="true height of the tracked object in metres, when --classes "
                         "is something other than person")
    ap.add_argument("--robot-radius", type=float, default=planner.robot_radius_m,
                    help="the loaded robot's measured plan-view planning radius in "
                         "metres. It sets both obstacle clearance and MAPPO scale; do "
                         "not copy the value from another platform")
    ap.add_argument("--obstacle-radius", type=float, default=planner.obstacle_radius_m,
                    help="plan-view footprint of a TRACKED MOVER in metres. The default "
                         "is a person's; anything smaller wants its own number, because "
                         "inflating a small object to person-size is what turns a "
                         "passable gap into a local minimum")
    ap.add_argument("--obstacle-width", type=float, default=None,
                    help="obstacle width in metres. Without it the width is inferred "
                         "from a standing adult's aspect ratio, which for a low wide "
                         "robot is about half the truth and ranges a vertically "
                         "clipped box about twice too NEAR — measured at 0.09-0.14 m, "
                         "inside this robot's own footprint")

    static = ap.add_argument_group("static obstacles")
    static_source = static.add_mutually_exclusive_group()
    static_source.add_argument("--static-prop", default=None, choices=sorted(PROFILES),
                             help="segment a known-coloured static prop by colour and map "
                             "it in odom. No detector is trained on a recycling bin, "
                             "so this is the only way the pipeline sees one at all")
    static_source.add_argument(
        "--static-profile", type=Path,
        help="evidence-backed custom colour-profile JSON for one named static obstacle",
    )
    static.add_argument("--prop-height", type=float, default=None,
                        help="override the profile's height in metres (measured, not "
                             "estimated — every range scales linearly on it)")
    static.add_argument("--prop-radius", type=float, default=None,
                        help="override the profile's plan-view radius in metres")

    detect_static = ap.add_argument_group(
        "static obstacles from the neural detector (OFF unless --static-detect)")
    detect_static.add_argument(
        "--static-detect", action="store_true",
        help="ALSO map sub-threshold MobileNet-SSD detections as static landmarks. The "
             "published prototxt drops everything under 0.25 inside DetectionOutput, "
             "which is why a cardboard box scoring 0.12 is invisible today. OFF BY "
             "DEFAULT: this changes what the robot avoids, and its recall has never "
             "been measured on more than a single frame")
    detect_static.add_argument(
        "--static-detect-confidence", type=float, default=DEFAULT_STATIC_CONFIDENCE,
        help="score floor for that tier (default %(default)s). Measured cost of the "
             "default: 0 of 705 empty-corridor frames fire; 0.05 would be 20%% of them")
    detect_static.add_argument(
        "--static-detect-height", type=float, default=None, metavar="M",
        help="MEASURED height of the staged object, metres. Required by --static-detect: "
             "the detector finds an object of unknown size and every range scales "
             "linearly on this number")
    detect_static.add_argument(
        "--static-detect-width", type=float, default=None, metavar="M",
        help="MEASURED width, metres. Also required — an inferred width comes from a "
             "standing adult's aspect ratio and reports a clipped box at half its true "
             "range (see SizePrior.of_height)")
    detect_static.add_argument(
        "--static-detect-radius", type=float, default=None, metavar="M",
        help="plan-view radius of the object, metres. Defaults to half the measured "
             "width, i.e. its own footprint with no clearance added — the planner adds "
             "its own and must not double-count")
    detect_static.add_argument(
        "--static-detect-classes", nargs="+", default=list(STATIC_CLASSES),
        metavar="VOC_CLASS",
        help="VOC labels a candidate may carry (default: %(default)s). The labels are "
             "noise — a real bin comes back 'tvmonitor' and 'chair' on consecutive "
             "frames — so this is a coarse junk filter, not a classification")

    goal_group = ap.add_argument_group("goal")
    goal_group.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID,
                            help="ArUco id to walk toward")
    goal_group.add_argument("--goal-detect-scale", type=float, default=0.5,
                            metavar="FRACTION",
                            help="downscale the frame by this before ArUco detection. "
                                 "0.5 is ~4x cheaper and is the default, but it costs "
                                 "range: detection needs about 23.7 px of marker, so a "
                                 "105 mm marker stops locking past roughly 3.0 m. Pass "
                                 "1.0 to trade that latency back for range")
    goal_group.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE_M,
                            help="printed marker's black square, in metres")
    goal_group.add_argument("--waypoint", type=float, nargs=2, metavar=("FWD", "LEFT"),
                            default=None,
                            help="use a dead-reckoned waypoint instead of the marker")
    goal_group.add_argument("--goal-class", default=None, metavar="VOC_CLASS",
                            help="walk toward a detected object (e.g. chair) instead "
                                 "of a marker. Needs --goal-height")
    goal_group.add_argument("--goal-height", type=float, default=None,
                            help="the goal object's real height in metres, floor to top")
    goal_group.add_argument("--goal-width", type=float, default=None,
                            help="the goal object's real width in metres. Optional; "
                                 "used only once the box is cut off by the frame edge")
    goal_group.add_argument("--goal-crop", type=float, default=DEFAULT_GOAL_CROP,
                            help="fraction of the frame the goal detector looks at. "
                                 "Cropping is what makes a distant object detectable "
                                 "at all — the network squashes its input to 300x300, "
                                 "so a chair 240 px wide in a 1920 frame is 37 px to it "
                                 "and is missed entirely. 1.0 disables cropping")
    goal_group.add_argument("--goal-input-size", type=int, default=DEFAULT_GOAL_INPUT_SIZE,
                            help="detector input size for the GOAL pass. Smaller than "
                                 "--input-size on purpose: the crop has already "
                                 "multiplied the target's effective resolution, so 224 "
                                 "on a half crop resolves better than 300 on the full "
                                 "frame and costs 69 ms instead of 130")
    goal_group.add_argument("--goal-refresh", type=float, default=DEFAULT_GOAL_REFRESH_S,
                            help="seconds between goal-detection passes once a goal is "
                                 "held. A latched goal only needs re-measuring as fast "
                                 "as the odometry under it drifts")
    goal_group.add_argument("--goal-confidence", type=float, default=0.5,
                            help="minimum score to latch a goal. Higher than the "
                                 "obstacle threshold on purpose: missing a goal costs "
                                 "a frame, latching a false one walks at a wall")

    envelope = ap.add_argument_group("envelope")
    envelope.add_argument("--max-vx", type=float, default=limits.max_vx,
                          help="m/s forward cap")
    envelope.add_argument("--max-vy", type=float, default=limits.max_vy,
                          help="m/s strafe cap")
    envelope.add_argument("--max-wz", type=float, default=limits.max_wz,
                          help="rad/s yaw cap")
    envelope.add_argument("--derate", type=float, default=1.0,
                          help="scale the whole envelope, including yaw authority")
    envelope.add_argument("--max-seconds", type=float, default=nav.max_run_s,
                          help="hard run-time budget")
    envelope.add_argument("--rest-after", type=float, default=nav.rest_after_s,
                          help="seconds held before the platform's supported rest action")
    envelope.add_argument("--arrive", type=float, default=nav.arrive_tolerance_m,
                          help="stop this far short of the goal. Check it against the "
                               "staging: an arrival circle that encloses a mapped "
                               "obstacle has no reachable point on it")
    envelope.add_argument("--horizon", type=float, default=planner.horizon_s,
                          help="seconds the planner rolls each candidate forward. At "
                               "the default speed 2.5 s sees 0.88 m ahead, which is "
                               "about one blocking radius — raise it if the robot "
                               "reacts to a static obstacle too late to swerve")
    bindings.add_navigation_arguments(ap, envelope)
    ap.add_argument("--record", default=None, help="write an annotated MP4 here")
    ap.add_argument("--record-raw", default=None, metavar="PATH.mp4",
                    help="write the UNDECORATED camera frames here as well: no HUD, no "
                         "plan-view inset, no detection box. Same cadence and the same "
                         "frame indices as --record, so both join to the telemetry's "
                         "perception.video_frame. Off by default. This is the only "
                         "output of a run that is usable as training data — --record "
                         "burns the label into the pixels (issue #77)")
    ap.add_argument("--telemetry", default=None, metavar="PATH.jsonl",
                    help="write a machine-readable record of every control tick here: "
                         "pose, goal, the full obstacle list with positions and radii, "
                         "the command, and the index of the matching --record frame. "
                         "This is the interface for anything downstream — the console "
                         "log is prose and its fields change to stay legible")
    return ap


def main(argv: Sequence[str] | None = None, planner_factory=DynamicWindowPlanner,
         navigator_factory=VisualNavigator, bindings=None) -> None:
    """Run one navigation session.

    ``planner_factory`` is called as ``planner_factory(limits=..., config=...)`` and
    exists so a consumer can substitute its own controller — an RL policy, say — WITHOUT
    copying this function. Everything above and below that one line is what makes a run
    safe: the arm-latch refusal, the health gate, the recorder's codec check, and a
    ``finally`` whose ordering matters (stop the legs, then tear down what was feeding
    them). A duplicated copy of that will drift, and the drift will be in the arm check.

    A factory rather than a planner instance because ``limits`` and ``config`` are built
    here from the parsed arguments, so a caller has nothing to construct one from yet.

    A substituted controller usually needs the MEASURED velocity, which ``plan()`` does
    not carry, and on this robot the commanded and achieved velocities differ by about a
    factor of two — so a controller fed the command believes it is moving twice as fast
    as it is. Take ``loco`` off the returned planner if it wants one: the object this
    factory returns is handed to :class:`VisualNavigator` alongside ``loco``, and a
    factory that closes over its own reference is the simplest way to get both.
    """
    bindings = bindings or Go2Bindings()
    args = build_parser(bindings).parse_args(argv)
    config = NavConfig(
        live=args.live,
        max_run_s=args.max_seconds,
        arrive_tolerance_m=args.arrive,
        rest_after_s=args.rest_after,
        require_arm=not getattr(args, "no_require_arm", True),
        latch_arm=not getattr(args, "no_latch_arm", True),
        motion_mode=getattr(args, "motion_mode", "manual"),
        rest_when_blocked=bindings.rest_when_blocked,
        initially_standing=bindings.initially_standing,
    )

    loco = bindings.create_locomotion(args)
    health = bindings.create_health_monitor(args, live=config.live)
    camera = perception = recorder = raw_recorder = telemetry = None
    navigator = None
    try:
        # Enter the cleanup scope before the first external connection. A health or arm
        # startup failure must not leak the already-connected locomotion transport.
        loco.connect()
        health.start()
        bindings.start(args)
        bindings.preflight_navigation(args, config, health)

        def pose_tuple() -> tuple[float, float, float]:
            pose = loco.pose()
            return (pose.x, pose.y, pose.yaw)

        camera = bindings.create_camera(args, pose_tuple)
        camera.start()
        first = camera.latest()
        height, width = first.image.shape[:2]
        print(f"[visual_nav] camera live: {width}x{height}")

        bindings.validate_camera_calibration(args)
        camera_model = build_camera_model(width, height, args.calibration)
        static_detect = static_detect_prior(args)
        detector = PersonDetector(
            args.model_dir, input_size=args.input_size, confidence=args.confidence,
            classes=tuple(args.classes),
            static_confidence=(None if static_detect is None
                               else args.static_detect_confidence),
            static_classes=tuple(args.static_detect_classes))
        if args.obstacle_height is not None:
            prior = SizePrior.of_height(args.obstacle_height, args.obstacle_width)
        else:
            prior = SizePrior()
            if set(args.classes) != set(DYNAMIC_CLASSES):
                print(f"[visual_nav] WARNING: tracking {args.classes} but ranging with "
                      f"a PERSON size prior ({prior.height_m:.2f} m). Pass "
                      f"--obstacle-height or every range will be wrong.")
        print(f"[visual_nav] obstacles: {args.classes}, height prior "
              f"{prior.height_m:.3f} m")

        goal_source = build_goal_source(args, camera_model, pose_tuple)

        colour_detector = static_map = static_profile_used = None
        radii: dict = {}
        if args.static_prop or args.static_profile:
            profile = static_profile(args)
            static_profile_used = profile
            colour_detector = ColourBlobDetector(profile)
            radii[profile.label] = profile.radius_m
            print(f"[visual_nav] static prop: {profile.label} "
                  f"{profile.height_m:.4f} m tall, {profile.radius_m:.3f} m radius")
        if static_detect is not None:
            static_prior, static_radius = static_detect
            radii[STATIC_DETECT_LABEL] = static_radius
            print(f"[visual_nav] static detections: floor "
                  f"{args.static_detect_confidence:.3f}, classes "
                  f"{sorted(args.static_detect_classes)}, prior "
                  f"{static_prior.height_m:.3f} x {static_prior.width_m:.3f} m, radius "
                  f"{static_radius:.3f} m")
            print("[visual_nav] WARNING: --static-detect surfaces detections the shipped "
                  "prototxt suppresses. Its FALSE-ALARM cost is measured; its RECALL is "
                  "not, and rests on one frame. Supervise the run.")
        if radii:
            static_map = StaticObstacleMap(
                radii=radii, fov_rad=math.radians(camera_model.hfov_deg))

        limits = Limits(max_vx=args.max_vx, max_vy=args.max_vy,
                        max_wz=args.max_wz).scaled(args.derate)
        print(f"[visual_nav] envelope: vx<={limits.max_vx:.2f} vy<={limits.max_vy:.2f} "
              f"wz<={limits.max_wz:.2f}")
        bindings.warn_if_below_gait_floor(limits.max_vx, args)
        robot_radius = bindings.robot_radius(args, PlannerConfig().robot_radius_m)
        planner_config = PlannerConfig(horizon_s=args.horizon,
                                       obstacle_radius_m=args.obstacle_radius,
                                       robot_radius_m=robot_radius)
        print(f"[visual_nav] planner: horizon {planner_config.horizon_s:.1f}s "
              f"({planner_config.horizon_s * limits.max_vx:.2f} m of lookahead at "
              f"top speed), robot radius {planner_config.robot_radius_m:.2f} m, "
              f"mover radius {planner_config.obstacle_radius_m:.2f} m")
        planner = planner_factory(limits=limits, config=planner_config)
        expansion = ExpansionConsistency() if args.expansion_filter else None
        if expansion is not None:
            print("[visual_nav] expansion filter ON: a confirmed track whose range "
                  "does not fall as odometry demands is withheld from the planner")
        tracker = ObstacleTracker(fov_rad=math.radians(camera_model.hfov_deg),
                                  expansion=expansion)

        perception = PerceptionWorker(
            camera, detector, camera_model, goal_source, pose_tuple, prior,
            colour_detector,
            static_prior=(None if static_detect is None else static_detect[0]))
        perception.start()

        def open_recorder(path: str) -> cv2.VideoWriter:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     RECORD_FPS, (width, height))
            # A codec the build cannot encode yields a writer that silently swallows
            # every frame, a 0-byte file and a cheerful "wrote run.mp4" at the end.
            # The recording IS the evidence for a live run, so fail here — before the
            # legs move — rather than after the only chance to capture it has passed.
            if not writer.isOpened():
                raise SystemExit(
                    f"[visual_nav] cannot open {path} for writing (mp4v). "
                    f"The run would produce an empty file, so it is not starting. "
                    f"Check the path is writable and that this OpenCV has FFMPEG.")
            return writer

        if args.record:
            recorder = open_recorder(args.record)
        if args.record_raw:
            raw_recorder = open_recorder(args.record_raw)

        if config.live:
            bindings.prepare_motion(args, loco)

        if args.telemetry:
            telemetry = TelemetryWriter(args.telemetry)
            header = {
                "live": config.live,
                "goal": goal_source.description,
                "classes": list(args.classes),
                "confidence": args.confidence,
                "static_prop": (
                    static_profile_used.label if static_profile_used is not None else None
                ),
                "static_profile": static_profile_telemetry(args, static_profile_used),
                "arrive_tolerance_m": config.arrive_tolerance_m,
                "control_hz": config.control_hz,
                "camera": {"width": width, "height": height,
                           "focal_px": camera_model.focal_px,
                           "hfov_deg": camera_model.hfov_deg,
                           "height_m": camera_model.height_m},
                "envelope": {"max_vx": limits.max_vx, "max_vy": limits.max_vy,
                             "max_wz": limits.max_wz},
                "planner": {"horizon_s": planner_config.horizon_s,
                            "robot_radius_m": planner_config.robot_radius_m,
                            "hard_gap_m": planner_config.hard_gap_m,
                            "soft_gap_m": planner_config.soft_gap_m,
                            "static_soft_gap_m": STATIC_SOFT_GAP_M},
                "video": args.record,
                "video_raw": args.record_raw,
            }
            header.update(bindings.telemetry_config(args))
            telemetry.write_header(**header)

        navigator = navigator_factory(
            loco, perception, planner, tracker, goal_source, health, config, recorder,
            static_map, telemetry, stand_up_fn=bindings.stand_up,
            lie_down_fn=bindings.lie_down, raw_recorder=raw_recorder,
            expansion=expansion,
        )
        navigator.run()
    finally:
        # Order matters: stop the legs first, then tear down what was feeding them.
        def release_recorder() -> None:
            if recorder is not None:
                recorder.release()
                print(f"[visual_nav] wrote {args.record}")

        def release_raw_recorder() -> None:
            if raw_recorder is not None:
                raw_recorder.release()
                print(f"[visual_nav] wrote {args.record_raw} (undecorated)")

        def close_telemetry() -> None:
            if telemetry is not None:
                telemetry.close()
                print(f"[visual_nav] wrote {args.telemetry} "
                      f"({telemetry.records} records)")

        run_cleanup("visual_nav", [
            ("navigator park", None if navigator is None else navigator.park),
            ("perception stop", None if perception is None else perception.stop),
            ("camera stop", None if camera is None else camera.stop),
            ("recorder release", release_recorder),
            ("raw recorder release", release_raw_recorder),
            ("telemetry close", close_telemetry),
            ("platform shutdown", bindings.shutdown),
            ("health stop", health.stop),
            ("locomotion shutdown", loco.shutdown),
        ])


if __name__ == "__main__":
    main()
