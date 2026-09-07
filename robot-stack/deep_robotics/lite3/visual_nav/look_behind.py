#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn around, look at the space the flip travels into, turn back, and only then flip.

⛔ THIS FILE IS THE ONLY PATH IN THIS REPOSITORY FROM AN UNATTENDED ARRIVAL TO A VENDOR
CANNED ACTION, AND EVERY LINE OF IT EXISTS TO MAKE THAT PATH REFUSABLE. Read the seam
argument below before changing anything here.

WHAT THE OPERATOR ASKED FOR. A flip at the goal. Today ``flourish.py`` will not do that
unattended: the two backflips travel BACKWARD into the one direction this platform is
completely blind in, so each demands ``--rear-clearance-metres`` -- a human with a tape
measure -- and ``--operator-triggered``, which no end-of-run path passes.

WHAT THIS DOES INSTEAD. On arrival, with everything below armed:

    1. yaw +180 deg          ``flourish.py --kind look --degrees 180``: the measured yaw
                             primitive, closed on measured heading
    2. LOOK                  the drive's OWN camera, OWN calibration, OWN detector and OWN
                             size prior, ranged by ``person_detector.range_detections`` --
                             the same function ``visual_nav.PerceptionWorker`` calls every
                             cycle. No new detector, no second set of thresholds
    3. yaw -180 deg          back to roughly the arrival heading
    4. flip                  ONLY if every gate in :meth:`Clearance.refusals` passed

⛔ WHAT THE LOOK CAN AND CANNOT PROVE, AND THIS IS THE FIRST THING TO UNDERSTAND. The
detector is MobileNet-SSD over a fixed list of VOC classes. It finds a person. It finds a
chair. **It does not find a wall, a step, a glass door, a table edge, a dropped cable or
the edge of a stage**, because none of those is a VOC class and nothing in this stack
ranges unclassified geometry. So this look answers exactly one question --

    "has something the detector can see moved into the space behind the robot?"

-- which is the question a human standing at the goal cannot answer while watching the
front of the robot, and it is NOT the question "is the floor behind clear". That second
question is still the operator's tape measure, and this file still requires the answer:
``--flip-rear-clearance-metres`` is carried straight through to ``flourish.check_rear``,
unchanged and unweakened. The look is a VETO ADDED ON TOP of the human's measurement, not
a replacement for it. Anything that presents it as a replacement is a claim this hardware
does not support, and the direction that claim fails in is a robot flipping into a wall.

⛔ THE ARRIVAL / OPERATOR_ONLY SEAM. ``flourish.ARRIVAL_KINDS`` may fire unattended
because every one of them keeps the robot's centre where it is;
``flourish.OPERATOR_ONLY_KINDS`` may not, because they travel. This file does NOT move a
kind between those sets and does not weaken either. It adds a third thing: an
AUTHORISATION, which is a :class:`Clearance` whose ``authorised`` is not a stored flag
somebody could set but a property COMPUTED from evidence -- a completed outward turn, a
completed return turn, an armed run, unanimous fresh frames, a nearest ranged obstacle
beyond the distance ``flourish`` itself demands, nothing ranging could not place, and
every precondition ``flourish`` ships no default for. There is no boolean to forge;
forging authorisation would mean forging a turn ``flourish`` exited zero on and frames a
detector produced. :func:`flip_arguments` is the only function on this path that can put
a travelling kind on a command line, and it recomputes every gate itself rather than
trusting the object handed to it, so a hand-assembled ``Clearance`` buys nothing.

⚠️ YAW DRIFT OVER TWO 180 DEG TURNS, because the goal is latched in odom and anything
later navigates from this pose. Each ``look`` leg finishes when it has ACCUMULATED its
span to within ``flourish.LEG_TOLERANCE_RAD`` (0.12 rad = 6.9 deg), so two legs can leave
the robot ~13.7 deg off its arrival heading, plus whatever the vendor pose channel itself
drifts across the ~7.3 s the two turns take at the measured 0.8565 rad/s. NOTHING HERE
MEASURES THAT DRIFT AND NOTHING CAN FROM HERE: ``Flourish`` accumulates wrapped
tick-to-tick deltas from the same pose channel it would have to be checked against, and
it is a separate process that reports only an exit code. Then the flip adds a yaw error
of its own that is not merely unmeasured but UNMEASURABLE -- the pose channel freezes for
the whole manoeuvre (2026-09-07, both Ventures: one distinct ``pos_world`` value across an
entire flip) and resumes only once the robot walks. In ``mission.py`` this is survivable
because arrival is terminal: ``main`` returns 0 straight after, and no leg is planned from
the post-flip pose in that process. It is NOT survivable for a caller that navigates
afterwards, which must re-acquire rather than trust the latched pose. Printed at the end
of every sequence too, because a comment is not where an operator reads it.

⚠️ THE TWO EXTRA TURNS SWEEP THE ROBOT'S OWN FOOTPRINT, and this file invents no rule
about that. ``flourish.check_room`` already owns it -- a turn in place moves the corners
of a 0.90 m diagonal through a 0.90 m circle, this platform has no lateral sensing, and
the lane width is therefore the operator's measurement with no default. Both ``look``
legs are ordinary ``flourish.py`` invocations built by ``mission.flourish_command``, which
passes ``--lane-width-metres``, so ``check_room`` runs on each of them exactly as it runs
on the arrival spin. There is deliberately no second width check here: a copy would be a
second rule to forget to update, and this one is already load-bearing where it is.

DEFAULT OFF. A run that does not pass ``--flip-on-arrival`` never constructs a
:class:`Clearance`, never opens a camera and never runs a gesture: :func:`arrival_flip`
returns ``None`` on its first line.
"""

from __future__ import annotations

import math
import sys
import textwrap
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROBOT_STACK = _HERE.parents[2]
_LOCOMOTION = _HERE.parents[0] / "locomotion"
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
# ⚠️ REVERSED, SO THAT `_HERE` ENDS UP FIRST. Each `insert(0)` pushes the previous one
# along, and BOTH trees ship a `camera.py`, a `safety.py` and a `robot_bindings.py`. A
# process that had this directory at the front of its path -- which is what `mission.py`
# sets up before importing this -- must not have it quietly moved behind the vendored one
# by an import. Everything reached by a BARE name below (`flourish`, `person_detector`,
# `camera_model`, `visual_nav`) exists in exactly one of these; everything with a twin is
# imported by its package path instead. See `open_scan`.
for _path in (str(_LOCOMOTION), str(_COMMON), str(_ROBOT_STACK), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flourish  # noqa: E402
from reverse_along_path import PHASE_TIMEOUT_MARGIN  # noqa: E402

#: The kinds a look BEHIND is evidence about, taken from ``flourish``'s own travel strings
#: rather than listed here.
#:
#: ⛔ ``twist-jump`` IS EXCLUDED BY THIS, AND THAT IS THE POINT. Its entry says "travel
#: UNKNOWN -- never sent from this repository, by anyone, in any direction. It is NOT known
#: to stay in place". Looking backward is evidence about the space behind; it is no
#: evidence at all about a manoeuvre whose direction nobody has observed, and authorising
#: one on the strength of a rear view would be exactly the borrowed confidence this file
#: exists to refuse. Derived, so a kind that later earns a measured backward travel joins
#: on its own and one that loses the claim drops out on its own.
BACKWARD_KINDS = tuple(kind for kind, action in flourish.VENDOR_ACTIONS.items()
                       if "BACKWARD" in action.travel)

#: Frames that must ALL read clear before the flip is authorised.
#:
#: THREE, and the number is ``static_map.CONFIRM_SIGHTINGS`` (2) plus one. That constant
#: exists because a single-frame flicker is the failure mode of a weak detection, so the
#: shared stack will not put a landmark in front of the planner until two frames agree.
#: Here the flicker runs the other way round -- one frame that happens to MISS the chair
#: would authorise a flip -- so the agreement is required on the CLEAR verdict, and one
#: frame more than the map asks for, because the consequence is a whole-body manoeuvre
#: this stack cannot interrupt rather than a swerve it can re-plan next tick.
CONFIRM_CLEAR_FRAMES = 3

#: Oldest frame a verdict may rest on. ``visual_nav.NavConfig.perception_timeout_s`` -- the
#: navigator's OWN staleness limit, past which the drive stops the legs and prints
#: "perception stale ... holding" rather than acting on the belief. A frame too old to
#: steer on is too old to flip on. Read from ``NavConfig`` at each use rather than copied,
#: so the two cannot drift apart; the literal below is only the fallback for a tree where
#: the navigator will not import.
FRAME_STALE_S_FALLBACK = 0.6

#: The drive's own measured perception cycle, seconds. 317 ms per cycle on the 2026-08-25
#: runs (``visual_nav.PerceptionResult.cycle_ms``'s note), against 202 ms of detect.
PERCEPTION_CYCLE_S = 0.317

#: How long the scan may take before it is called a camera fault rather than a slow frame.
#: The frames it needs at the measured cycle above, times the same
#: ``PHASE_TIMEOUT_MARGIN`` every other timed phase in this stack uses. Not a number picked
#: to feel generous: a scan that outruns it has a camera that is not delivering, and that
#: is a refusal like any other.
SCAN_TIMEOUT_S = CONFIRM_CLEAR_FRAMES * PERCEPTION_CYCLE_S * PHASE_TIMEOUT_MARGIN

#: Degrees turned to look behind, and back. Not a tuning knob: a look behind that is not
#: 180 deg is a look at something other than the space the flip travels into.
LOOK_DEGREES = 180.0

#: Returned by a gesture runner that could not build a command at all -- no ``--flourish``,
#: no robot id, no lane width. Distinct from a non-zero exit: nothing ran.
GESTURE_UNAVAILABLE = -1

#: Perception flags copied FROM the drive command rather than restated here.
#:
#: Copied for the reason ``mission._FLOURISH_PASSTHROUGH`` is copied: the look must be
#: taken through EXACTLY the camera, model, thresholds and size prior the run itself drove
#: on. A default restated here would be a second configuration to keep in step, and a scan
#: at a different confidence or a different size prior answers a different question from
#: the one the drive spent its whole run answering.
SCAN_PASSTHROUGH = ("--camera-source", "--camera-rectify", "--calibration",
                    "--model-dir", "--input-size", "--confidence",
                    "--obstacle-height", "--obstacle-width")

#: Multi-valued (``nargs="+"``) flags copied the same way. ``--classes`` decides what
#: counts as an obstacle at all, so a scan that dropped it would be blind to exactly the
#: classes an operator added because they are in this room.
SCAN_PASSTHROUGH_MULTI = ("--classes",)

#: Bare flags copied the same way.
SCAN_FLAGS = ("--camera-gstreamer",)

#: Without these the look cannot be taken at all.
#:
#: ``--calibration`` is required and not merely preferred: without it
#: ``visual_nav.build_camera_model`` falls back to the GO2's nominal intrinsics and lens
#: height, and every range then compared against a clearance floor would be a measurement
#: of a different robot. Fail closed rather than range on somebody else's lens.
SCAN_REQUIRED = ("--camera-source", "--calibration")


@lru_cache(maxsize=1)
def frame_stale_s() -> float:
    """The navigator's own perception staleness limit, or the fallback constant.

    Cached: :meth:`Clearance.refusals` calls this on every gate evaluation and importing
    the navigator drags in the whole shared stack. Caching a CONSTANT is safe -- the number
    is a dataclass default, not a reading -- and it keeps this file importable, and its
    decisions testable, in a tree where the navigator will not import at all.
    """
    try:
        import visual_nav
    except Exception:
        return FRAME_STALE_S_FALLBACK
    return float(getattr(visual_nav.NavConfig(), "perception_timeout_s",
                         FRAME_STALE_S_FALLBACK))


def required_clearance_m(kind: str) -> float:
    """Metres of clear space behind that ``kind`` needs, from ``flourish``'s own table.

    ⛔ NO NUMBER IS INVENTED HERE, AND THIS FUNCTION IS WHERE THAT PROMISE IS KEPT. The
    flip's travel has never been measured -- the vendor pose channel freezes for the whole
    manoeuvre, so both Ventures reported one distinct ``pos_world`` value across an entire
    flip on 2026-09-07 -- and the ~1.5 m in ``flourish.BACKFLIP_TRAVEL_M`` is the
    operator's figure, not a measurement of this robot on this floor.

    It is used for exactly what ``flourish.check_rear`` already uses it for: the FLOOR a
    stated clearance must clear. That is the direction it is safe to be wrong in -- if the
    real travel is further, a clearance was checked against a number that was too small,
    so this is a minimum to refuse below and never a promise of where the robot stops.
    Read out of ``VENDOR_ACTIONS[kind].rear_clearance_m`` rather than copied, so this file
    and the gate that re-checks it downstream can never disagree about what the floor is.
    """
    return flourish.VENDOR_ACTIONS[kind].rear_clearance_m


def clearance_basis(kind: str) -> str:
    """Where :func:`required_clearance_m` got its number, in the operator's own words."""
    if kind not in flourish.VENDOR_ACTIONS:
        return "no basis at all -- this is not a vendor canned action"
    return flourish.VENDOR_ACTIONS[kind].rear_basis


@dataclass(frozen=True)
class RearView:
    """What the camera and the detector actually produced while the robot faced backward.

    ⛔ EVERY DEFAULT DENIES. A ``RearView()`` built with no arguments describes a look that
    never happened -- zero frames, an infinitely stale newest frame, a nearest obstacle at
    zero metres -- so a field this file forgets to fill refuses the flip rather than
    passing it. That is why the deny values are the defaults, and not the ``inf``/``0``
    that would have been convenient for the arithmetic.
    """

    #: Frames that produced a decision at all.
    frames: int = 0
    #: Of those, how many put nothing inside the required distance and left no box unplaced.
    clear_frames: int = 0
    #: Age of the OLDEST frame the verdict rests on, seconds. ``inf`` = there were none.
    stalest_frame_s: float = math.inf
    #: Nearest RANGED detection across the scan, metres. ``0.0`` means unknown, which is
    #: the denying value; ``inf`` means the detector found nothing at all to range.
    nearest_m: float = 0.0
    #: What that nearest thing was labelled, for the console line.
    nearest_label: str = "(nothing was looked at)"
    #: Boxes ``range_detections`` REFUSED, plus boxes it placed from a source in
    #: ``person_detector.UNRANGEABLE_SOURCES``. See :meth:`Clearance.refusals`.
    unrangeable: int = 0
    #: Frames the camera failed to decode during the scan.
    camera_errors: int = 0
    #: Why the scan ended, when that is worth saying.
    detail: str = "no scan was taken"

    @property
    def unanimous(self) -> bool:
        return self.frames > 0 and self.clear_frames == self.frames


@dataclass(frozen=True)
class Clearance:
    """The authorisation, and it is EVIDENCE rather than a decision somebody recorded.

    There is deliberately no ``authorised`` field. :attr:`authorised` is computed from
    :attr:`refusals`, which is computed from the fields below, so the only way to hold an
    authorised clearance is to have actually turned around, actually looked, and actually
    turned back. A stored boolean could be passed ``True`` by a caller in a hurry; a
    completed turn cannot be.

    Every default denies, for the same reason :class:`RearView`'s do.
    """

    #: Which vendor canned action this authorises. Must be in :data:`BACKWARD_KINDS`.
    kind: str = ""
    #: Metres required, from :func:`required_clearance_m`.
    required_m: float = math.inf
    #: Whether the run this rode on was armed. A dry run's ``look`` prints its plan and
    #: exits 0 WITHOUT TURNING, so exit codes alone would read a dry run as two completed
    #: turns. See ``mission.flourish_command``'s comment on the measurement of 2026-09-04
    #: that taught this stack the difference.
    live: bool = False
    #: The outward 180 deg exited zero.
    turned_out: bool = False
    #: The return 180 deg exited zero.
    turned_back: bool = False
    #: What the look saw.
    view: RearView = field(default_factory=RearView)
    #: The operator's own tape measure, still required and still handed to
    #: ``flourish.check_rear``. See the module docstring: the detector cannot see a wall,
    #: so this is not something the look replaces.
    operator_rear_clearance_m: float | None = None
    #: ``flourish``'s two remaining no-default preconditions, carried so that the command
    #: built at the seam is COMPLETE. Without them ``flourish`` refuses anyway; refusing
    #: here means the refusal names the missing flag before a robot has turned twice.
    battery_floor_pct: float | None = None
    hold_seconds: float | None = None

    @property
    def refusals(self) -> tuple[str, ...]:
        """Every reason this is not authorised. Empty means it is.

        ⛔ READ THIS AS THE SAFETY ARGUMENT. Each branch is a way the look could fail to be
        evidence, and every one is written so that ABSENCE refuses. No perception, stale
        perception, a detector that produced nothing, a turn that did not finish, a box
        ranging could not place -- not one of those is "probably clear".
        """
        why: list = []
        view = self.view
        if self.kind not in BACKWARD_KINDS:
            why.append(
                f"{self.kind!r} is not a kind a look BEHIND is evidence about. "
                f"flourish's own travel string decides that, and only "
                f"{', '.join(BACKWARD_KINDS)} claim to travel backward")
        if not self.live:
            why.append(
                "the drive carried no --live, so both look gestures printed a plan and "
                "turned NOTHING. Nothing was looked at and there is no clearance to have")
        if not self.turned_out:
            why.append(
                "the outward 180 deg turn did not complete, so the camera never faced the "
                "space the flip travels into")
        if not self.turned_back:
            why.append(
                "the return 180 deg turn did not complete, so the robot is not facing the "
                "way it arrived and the space behind it is no longer the space that was "
                "looked at")
        if view.frames < CONFIRM_CLEAR_FRAMES:
            why.append(
                f"the look produced {view.frames} usable frame(s) and needs "
                f"{CONFIRM_CLEAR_FRAMES}: {view.detail}")
        elif not view.unanimous:
            why.append(
                f"{view.frames - view.clear_frames} of {view.frames} frames put something "
                f"inside {self.required_m:.2f} m or left a box unplaced")
        if view.stalest_frame_s > frame_stale_s():
            why.append(
                f"the oldest frame the verdict rests on was {view.stalest_frame_s:.2f}s "
                f"old, and the navigator holds the legs past {frame_stale_s():.2f}s. A "
                f"frame too old to steer on is too old to flip on")
        if view.unrangeable:
            why.append(
                f"{view.unrangeable} box(es) behind the robot could not be given a range. "
                f"A detection with no usable range leaves NOTHING behind, and an empty "
                f"obstacle list is what a planner reads as an open world -- that failing "
                f"open is issue #72's collision. A box this stack cannot place is treated "
                f"as a box in the way")
        if view.nearest_m < self.required_m:
            # Kept even when there were no frames at all, where the branch above has
            # already refused: two independent reasons is the right number for a gate that
            # must never fail open. It is WORDED differently there, though, because "the
            # nearest thing is at 0.00 m" reads as a measurement of a wall against the
            # robot's back rather than as the absence of a measurement.
            measured = (f"the nearest thing behind is {view.nearest_label} at "
                        f"{view.nearest_m:.2f} m" if view.frames else
                        "nothing behind was looked at, so there is no nearest anything")
            why.append(
                f"{measured} and {self.kind} needs {self.required_m:.2f} m of MEASURED "
                f"space, which is {clearance_basis(self.kind)}")
        if self.operator_rear_clearance_m is None:
            why.append(
                "no --flip-rear-clearance-metres. The detector finds VOC classes; it does "
                "not find a wall, a step or the edge of a stage, so the static room is "
                "still the operator's tape measure and this look does not replace it")
        elif self.operator_rear_clearance_m < self.required_m:
            why.append(
                f"--flip-rear-clearance-metres says {self.operator_rear_clearance_m:.2f} m "
                f"and {self.kind} needs {self.required_m:.2f} m")
        if self.battery_floor_pct is None:
            why.append(
                "no --flip-battery-floor-pct. Nobody has measured what a Lite3 flip draws "
                "and a brownout mid-flip is a fall; flourish ships no default and neither "
                "does this")
        if self.hold_seconds is None:
            why.append(
                "no --flip-hold-seconds. A backflip took 5.4-5.9 s to return the robot to "
                "force-control 6, and returning sooner hands control back mid-manoeuvre")
        return tuple(why)

    @property
    def authorised(self) -> bool:
        return not self.refusals

    def describe(self) -> str:
        """One console block: what was looked at, and the verdict."""
        view = self.view
        # ⚠️ "0.00 m" and "inf s" are the DENY defaults, and printing them as numbers reads
        # as a measurement of an empty room rather than as the absence of one. Say which it
        # is: an operator scanning this block must not mistake "nothing was looked at" for
        # "nothing is there", which are the two states this whole file exists to separate.
        if view.frames == 0:
            nearest = "NOTHING WAS LOOKED AT"
            oldest = "no frame at all"
        else:
            nearest = ("nothing the detector can see" if math.isinf(view.nearest_m)
                       else f"{view.nearest_label} at {view.nearest_m:.2f} m")
            oldest = f"oldest {view.stalest_frame_s:.2f}s"
        stated = ("(nothing)" if self.operator_rear_clearance_m is None
                  else f"{self.operator_rear_clearance_m:.2f} m, by tape")
        turned = "" if self.live else "   (DRY RUN -- nothing turned)"
        lines = [
            f"  kind             {self.kind}",
            f"  needs            {self.required_m:.2f} m behind, which is "
            f"{clearance_basis(self.kind)}",
            f"  operator stated  {stated}",
            f"  turned out/back  {self.turned_out} / {self.turned_back}{turned}",
            f"  frames           {view.clear_frames} clear of {view.frames}, {oldest}, "
            f"{view.camera_errors} camera error(s)",
            f"  nearest behind   {nearest}",
            f"  unrangeable      {view.unrangeable}",
        ]
        if self.authorised:
            lines.append("  VERDICT          CLEAR of everything the detector can see. "
                         "A WALL IS NOT A VOC CLASS -- the")
            lines.append("                   stated rear clearance is what covers that, "
                         "and it is not waived.")
        else:
            lines.append("  VERDICT          NO FLIP")
            for reason in self.refusals:
                first, *rest = textwrap.wrap(reason, 60) or [""]
                lines.append(f"    - {first}")
                lines.extend(f"      {line}" for line in rest)
        return "\n".join(lines)


def scan_rear(*, camera, detector, camera_model, prior, required_m: float,
              frames: int = CONFIRM_CLEAR_FRAMES, timeout_s: float = SCAN_TIMEOUT_S,
              clock=time.monotonic) -> RearView:
    """Look at what is in front of the camera NOW, and range it the way the drive does.

    ⛔ THIS WRITES NO DETECTOR AND NO RANGING. ``detector.detect`` is the drive's own
    ``PersonDetector`` and ``range_detections`` is the function
    ``visual_nav.PerceptionWorker._cycle`` calls on every one of its own cycles, with the
    same ``refused=`` list it passes so that boxes ranging threw away stay visible instead
    of silently vanishing. The only thing added here is the DECISION, which the drive has
    no reason to make.

    ⚠️ THE WHOLE FIELD OF VIEW COUNTS, WITH NO CORRIDOR CARVED OUT OF IT. A narrower test
    -- "only things within half the robot's width of the flip's centre line" -- would be a
    claim about the manoeuvre's LATERAL extent, and the only figure this repository holds
    for the flip is a backward distance that is itself the operator's rather than a
    measurement. Nothing here knows how wide the swept volume is, so the test is a radius:
    anything the detector places inside ``required_m``, at any bearing, refuses. That is
    strict, and strict is the direction to be wrong in.
    """
    from person_detector import UNRANGEABLE_SOURCES, range_detections

    decided = 0
    clear = 0
    nearest = math.inf
    nearest_label = "nothing the detector can see"
    unrangeable = 0
    stalest = 0.0
    last_seq = 0
    deadline = clock() + timeout_s
    detail = ""

    while decided < frames:
        remaining = deadline - clock()
        if remaining <= 0.0:
            detail = (f"the scan ran out of its {timeout_s:.1f}s after {decided} usable "
                      f"frame(s); the camera is not delivering")
            break
        frame = camera.wait_for_new(last_seq, min(1.0, remaining))
        if frame is None:
            continue
        last_seq = frame.seq
        stalest = max(stalest, max(0.0, clock() - frame.capture_time))
        refused: list = []
        ranged = range_detections(detector.detect(frame.image), camera_model, prior,
                                  refused=refused)
        # A box ranging could not place and a box it placed from a source that is not a
        # measurement count the same. ``frame-fill`` and ``width-capped`` are CONSTANTS the
        # estimator returns when the geometry ran out -- see person_detector's own note on
        # the 2026-08-19 deadlock against a range that could not move -- and a constant is
        # not evidence that the space is clear.
        blind = len(refused) + sum(1 for item in ranged
                                   if item.source in UNRANGEABLE_SOURCES)
        unrangeable += blind
        decided += 1
        for item in ranged:
            if item.range_m < nearest:
                nearest, nearest_label = item.range_m, item.label
        if blind == 0 and not any(item.range_m < required_m for item in ranged):
            clear += 1
    else:
        detail = f"{decided} frame(s) looked at"

    return RearView(frames=decided, clear_frames=clear,
                    stalest_frame_s=stalest if decided else math.inf,
                    nearest_m=nearest if decided else 0.0,
                    nearest_label=nearest_label, unrangeable=unrangeable,
                    camera_errors=int(getattr(camera, "error_count", 0)), detail=detail)


def scan_arguments(command: list) -> tuple:
    """The perception settings for the look, read out of the DRIVE's own command line.

    Returns ``(settings, missing)``. ``settings`` maps flag -> value, ``True`` for a bare
    flag, or a list for a multi-valued one; ``missing`` names any of :data:`SCAN_REQUIRED`
    the drive did not carry.
    """
    settings: dict = {}
    for flag in SCAN_PASSTHROUGH:
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                settings[flag] = command[index + 1]
    for flag in SCAN_PASSTHROUGH_MULTI:
        if flag in command:
            values = []
            for token in command[command.index(flag) + 1:]:
                # ``nargs="+"``: every token up to the next flag belongs to it. None of
                # these take a negative number, so a leading ``--`` is unambiguous.
                if str(token).startswith("--"):
                    break
                values.append(token)
            if values:
                settings[flag] = values
    for flag in SCAN_FLAGS:
        if flag in command:
            settings[flag] = True
    missing = [flag for flag in SCAN_REQUIRED if flag not in settings]
    return settings, missing


def open_scan(settings: dict, *, printer=print) -> tuple:
    """Build the drive's camera, detector, camera model and size prior from ``settings``.

    Imported lazily and constructed here so that the DECISION half of this module -- which
    is what the tests are about -- needs neither OpenCV nor a robot. Returns
    ``(camera, detector, camera_model, prior)`` with the camera already started; the caller
    closes it.
    """
    # ⚠️ THE LITE3 MODULES COME BY PACKAGE PATH, NOT BY BARE NAME.
    # ``robot-stack/unitree/go2/visual_nav/camera.py`` is on this path too, and its
    # ``Go2Camera`` speaks DDS and takes an ``iface=``; whichever directory happened to be
    # inserted last would decide which one a bare ``import camera`` found, and the failure
    # would be a TypeError at the moment the robot has already turned around.
    # ``lite3_vision_shadow.py`` reaches its camera this way, for this reason. The three
    # bare names below have no twin in either tree.
    from deep_robotics.lite3.visual_nav.camera import Lite3Camera, parse_camera_source
    from deep_robotics.lite3.visual_nav.camera_rectify import Rectifier, rectified_camera
    from person_detector import (
        DEFAULT_CONFIDENCE,
        DYNAMIC_CLASSES,
        PersonDetector,
        SizePrior,
    )
    from visual_nav import DEFAULT_MODEL_DIR, build_camera_model

    classes = tuple(settings.get("--classes", list(DYNAMIC_CLASSES)))
    camera = Lite3Camera(parse_camera_source(settings["--camera-source"]),
                         gstreamer=bool(settings.get("--camera-gstreamer")))
    rectify = settings.get("--camera-rectify")
    if rectify is not None:
        # The same wrapping ``Lite3Bindings.create_camera`` applies, for the same reason:
        # without it a bearing is wrong by +8 deg at 40 deg off axis, and a bearing that is
        # wrong is a range that is wrong.
        rectifier = Rectifier(rectify)
        printer(f"[look-behind] {rectifier.describe()}")
        camera = rectified_camera(camera, rectifier)
    camera.start()
    first = camera.latest()
    if first is None:
        camera.stop()
        raise RuntimeError("the camera opened and then delivered no frame")
    height, width = first.image.shape[:2]
    camera_model = build_camera_model(width, height, settings["--calibration"])
    detector = PersonDetector(
        settings.get("--model-dir", str(DEFAULT_MODEL_DIR)),
        input_size=int(settings.get("--input-size", 300)),
        confidence=float(settings.get("--confidence", DEFAULT_CONFIDENCE)),
        classes=classes)
    if settings.get("--obstacle-height") is not None:
        width_m = settings.get("--obstacle-width")
        prior = SizePrior.of_height(float(settings["--obstacle-height"]),
                                    None if width_m is None else float(width_m))
    else:
        prior = SizePrior()
    printer(f"[look-behind] {width}x{height}, classes {classes}, height prior "
            f"{prior.height_m:.3f} m")
    return camera, detector, camera_model, prior


def flip_arguments(clearance: Clearance) -> tuple:
    """The extra ``flourish.py`` flags that fire the flip, or a refusal.

    ⛔ THIS IS THE SEAM. It is the only function on any arrival path that can put a
    travelling kind's preconditions on a command line, and it recomputes the authorisation
    from the clearance's own fields rather than believing a flag -- so handing it a
    hand-assembled ``Clearance`` gains nothing. The fields it insists on are a completed
    outward turn, a completed return turn, an armed run, unanimous fresh frames, a ranged
    nearest beyond the floor ``flourish`` itself sets, no unplaceable box, and every
    precondition ``flourish`` ships no default for.

    ⚠️ ``--operator-triggered`` IS PASSED HERE AND THAT DESERVES ITS OWN PARAGRAPH.
    ``flourish``'s help calls it "a HUMAN is firing this, now, watching this robot", and on
    this path the human is not watching the robot's back at the moment it fires. What they
    DID do is state, at launch, with the room in front of them: this kind, this rear
    clearance by tape, this battery floor, this hold. That is every judgement the flag was
    protecting except one -- whether something has since moved into the space -- and that
    one is precisely what the look answers and a person standing at the front of the robot
    could not. So the flag is not being waived and is not being defaulted: it is carried
    from an operator decision made minutes earlier PLUS a measurement taken seconds
    earlier, and both are required. If that trade is not acceptable for a venue, the answer
    is to leave ``--flip-on-arrival`` off, which is where it ships.
    """
    if not clearance.authorised:
        raise flourish.Refusal(
            "refusing to build a flip command from an unauthorised look: "
            + "; ".join(clearance.refusals))
    return ("--operator-triggered",
            "--rear-clearance-metres", f"{clearance.operator_rear_clearance_m:.4f}",
            "--acrobatic-battery-floor-pct", f"{clearance.battery_floor_pct:.4f}",
            "--action-hold-seconds", f"{clearance.hold_seconds:.4f}")


def flip_settings(args) -> tuple:
    """The operator's flip settings off ``args``, and which of them are unanswered.

    None of them has a default, which is ``flourish``'s rule for these actions restated at
    the layer that now fires them: where a precondition is unknown, refuse until the
    operator states it rather than picking a number and calling it a precondition.
    """
    answered = {
        "--flip-kind": getattr(args, "flip_kind", None),
        "--flip-rear-clearance-metres": getattr(args, "flip_rear_clearance_metres", None),
        "--flip-battery-floor-pct": getattr(args, "flip_battery_floor_pct", None),
        "--flip-hold-seconds": getattr(args, "flip_hold_seconds", None),
    }
    return answered, [name for name, value in answered.items() if value is None]


def _look(run_gesture, degrees: float, printer) -> bool:
    """One 180 deg leg through ``flourish.py --kind look``. True only on a clean exit."""
    code = run_gesture(flourish.Flourish.LOOK, ("--degrees", f"{degrees:.1f}"))
    if code == GESTURE_UNAVAILABLE:
        printer("[look-behind] the turn could not be commanded at all; the reason is on "
                "the line above")
        return False
    if code != 0:
        printer(f"[look-behind] the {degrees:+.0f} deg turn exited {code}. Its own refusal "
                f"is above; a turn that did not finish is not a clearance")
        return False
    return True


def arrival_flip(command: list, args, run_gesture, *, printer=print,
                 scanner=None) -> Clearance | None:
    """Turn around, look, turn back, and flip if -- and only if -- the look authorises it.

    ``run_gesture(kind, extra) -> int`` runs one ``flourish.py`` invocation and returns its
    exit code, or :data:`GESTURE_UNAVAILABLE` if no command could be built. Injected rather
    than built here so this file never has to decide whether a run was armed:
    ``mission.flourish_command`` INHERITS ``--live`` from the drive, which is the property
    that keeps a scene check out of a run the driver reported as ``can_move=False``.

    Returns the :class:`Clearance` reached, or ``None`` when the feature is off.
    """
    if not getattr(args, "flip_on_arrival", False):
        # DEFAULT OFF, on the first line, before anything is imported, opened or turned. A
        # run that did not ask for a flip must behave exactly as it did before this file
        # existed, and leaving immediately is the cheapest way to be sure of that.
        return None

    answered, unanswered = flip_settings(args)
    if unanswered:
        printer(f"[look-behind] --flip-on-arrival needs {', '.join(unanswered)}; not one "
                f"of them has a default, and the flip will NOT be attempted")
        return None
    kind = str(answered["--flip-kind"])
    if kind not in BACKWARD_KINDS:
        printer(f"[look-behind] --flip-kind {kind!r} is not one a look behind is evidence "
                f"about ({', '.join(BACKWARD_KINDS)}); the flip will NOT be attempted")
        return None

    settings, missing = scan_arguments(command)
    if missing:
        printer(f"[look-behind] the drive command carries no {', '.join(missing)}, so the "
                f"look cannot be taken through the run's own camera and calibration. No "
                f"flip.")
        return None

    required = required_clearance_m(kind)
    clearance = Clearance(
        kind=kind, required_m=required, live="--live" in command,
        operator_rear_clearance_m=float(answered["--flip-rear-clearance-metres"]),
        battery_floor_pct=float(answered["--flip-battery-floor-pct"]),
        hold_seconds=float(answered["--flip-hold-seconds"]))

    printer(f"[look-behind] turning {LOOK_DEGREES:+.0f} deg to look at the space {kind} "
            f"travels into; it needs {required:.2f} m of it")
    if not _look(run_gesture, LOOK_DEGREES, printer):
        # ⚠️ NO BLIND CORRECTION. A look that refused part way through stopped the legs at
        # a heading nothing here knows, and a -180 from an unknown heading is not a return,
        # it is a second guess. Say the heading is unknown and stop.
        printer("[look-behind] the outward turn did not complete. THE ROBOT'S HEADING IS "
                "NOW UNKNOWN and nothing here will guess at a correction. No flip; go and "
                "look at the robot.")
        return clearance

    view = RearView(detail="the scan was not attempted")
    try:
        if scanner is not None:
            view = scanner(settings, required)
        elif not clearance.live:
            view = RearView(detail="a dry run turns nothing, so there was nothing to look "
                                   "at and no camera was opened")
        else:
            camera, detector, camera_model, prior = open_scan(settings, printer=printer)
            try:
                view = scan_rear(camera=camera, detector=detector,
                                 camera_model=camera_model, prior=prior,
                                 required_m=required)
            finally:
                camera.stop()
    except Exception as failure:
        # Deliberately broad, and it FAILS CLOSED. A detector that will not load, a
        # calibration that states no lens height, a camera that never opened: every one of
        # them arrives here, and every one of them is "do not flip", never "assume clear".
        view = RearView(detail=f"the look could not be taken ({failure!r})")
        printer(f"[look-behind] the look failed: {failure!r}. That is a refusal, not a "
                f"clearance.")

    clearance = replace(clearance, turned_out=True, view=view)

    # ALWAYS ATTEMPTED, whatever the look said. A robot left facing backward at the goal is
    # a robot pointing its one camera away from everything, and the return turn is what
    # puts the pose the goal is latched against back roughly where the run left it.
    printer(f"[look-behind] turning {-LOOK_DEGREES:+.0f} deg back to the arrival heading")
    clearance = replace(clearance,
                        turned_back=_look(run_gesture, -LOOK_DEGREES, printer))

    printer("[look-behind] rear clearance:")
    printer(clearance.describe())
    if not clearance.authorised:
        return clearance

    printer(f"[look-behind] firing {kind}. Odometry freezes for the manoeuvre and resumes "
            f"only once the robot walks, and two 180 deg turns can leave the heading "
            f"{2 * math.degrees(flourish.LEG_TOLERANCE_RAD):.0f} deg off arrival: a pose "
            f"read after this is stale, and roughly -- not exactly -- where the run "
            f"finished.")
    code = run_gesture(kind, flip_arguments(clearance))
    if code != 0:
        printer(f"[look-behind] {kind} exited {code}; its own refusal is above")
    return clearance
