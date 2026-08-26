# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the Lite3 commissioning measurements.

Every probe in this directory produces **one number, one artefact, and one paragraph an
operator can paste into
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)**. This
module holds the parts they all need, so that the shape of a measurement is decided once:

* :class:`Refusal` and :func:`brief` — the refusal register and the operator briefing.
  A probe that cannot measure what it claims must say so and produce nothing.
* :func:`run_segment` and :func:`body_delta` — the pose-derivative segment. Displacement
  is measured from **pose**, never from the platform's own velocity estimate, and rotated
  into the body frame the segment was actually flown in.
* :func:`fit_ratio` — least squares through the origin, with its residual.
* the commissioning record: :func:`new_record`, :func:`merge_measurement`,
  :func:`write_record`, :func:`read_record`, :func:`require_reviewed`.

**Nothing here carries a measured Lite3 value.** Where a number must come from the robot,
the sentinel is ``None`` and the probe refuses. In particular the Go2's 0.35 m/s forward
and 0.20 m/s lateral gait floors are not defaults here, are not fallbacks here, and do not
appear in this directory at all: they are properties of a different robot, and copying
them is the specific mistake this harness exists to stop.

The one constant this module does carry, :data:`WALKED_MARGIN_M`, is a property of the
*method* rather than of the robot — see its docstring, and note that every probe that uses
it also runs interleaved zero-command controls whose job is to falsify it.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Artefact schema. Bumped when a reader would misread an older file, not when a field is
#: added, so a v1 reader may find keys it does not know and must ignore them.
SCHEMA = "lite3-commissioning/v1"

#: ``provenance`` values. A record is ``provisional`` from the moment it is written and
#: stays that way until a human names themselves in it. Nothing marked ``provisional``
#: may be turned into live-movement flags -- see :func:`require_reviewed`.
PROVISIONAL = "provisional"
REVIEWED = "reviewed"

#: Net body-frame displacement, in metres, below which a segment is treated as "this
#: robot did not travel".
#:
#: This is **not** a Lite3 property and it is not a gait floor. It is the smallest
#: displacement this method can tell apart from the odometry drift of a robot that is
#: standing still, and it is stated in metres rather than m/s so it does not change
#: meaning when the segment length changes.
#:
#: It is also falsifiable, which is the point. Every probe that uses it interleaves
#: zero-command control segments, and :func:`check_controls_are_still` refuses the whole
#: run if any control drifts this far -- because at that point the threshold is too small
#: for this unit and every "did it walk?" answer built on it is noise. A threshold that
#: nothing can ever exceed is not a threshold.
WALKED_MARGIN_M = 0.05

#: The VMAS agent radius the shipped MAPPO checkpoint was trained with. Used only to turn
#: a measured Lite3 radius into ``--policy-scale``; it describes the policy, not the robot.
POLICY_AGENT_RADIUS_M = 0.10


class Refusal(Exception):
    """A precondition, or a post-hoc validity check, failed.

    Raising this is a *result*: the probe has decided that no number it could print would
    mean anything. Callers turn it into a printed ``REFUSING`` banner and a non-zero exit
    rather than into a traceback, so an operator reads a sentence instead of a stack.
    """


def refuse_unmeasured(**values: object) -> None:
    """Refuse for every keyword whose value is the unmeasured sentinel ``None``.

    The keyword name is the command-line flag, so the message tells the operator exactly
    what to go and measure rather than what a variable is called.
    """
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise Refusal(
            "these values must be MEASURED on this robot, and this harness ships no "
            "default for any of them: " + ", ".join(sorted(missing))
        )


def require_positive_finite(**values: float) -> None:
    """Refuse for every keyword that is not a finite, strictly positive number."""
    bad = [name for name, value in values.items()
           if value is None or not math.isfinite(float(value)) or float(value) <= 0.0]
    if bad:
        raise Refusal("these values must be finite and positive: " + ", ".join(sorted(bad)))


def brief(title: str, *, does: str, needs: Sequence[str], means: str,
          moves: bool, printer: Callable[[str], None] = print) -> None:
    """Print what this probe is about to do, before it does any of it.

    Written for an operator in Shanghai who did not write the code and cannot read it
    while the robot is standing in front of them. Three questions, in the order they are
    asked: what will happen, what do I have to do, and what will the number mean.
    """
    rule = "=" * 78
    printer(rule)
    printer(title)
    printer(rule)
    printer("MOVES THE ROBOT: " + ("YES -- keep the emergency stop in your hand"
                                   if moves else "no"))
    printer("")
    printer("WHAT IT DOES")
    for line in does.strip().splitlines():
        printer("  " + line.strip())
    printer("")
    printer("WHAT IT NEEDS FROM YOU")
    for item in needs:
        printer("  - " + item)
    printer("")
    printer("WHAT THE NUMBER MEANS")
    for line in means.strip().splitlines():
        printer("  " + line.strip())
    printer(rule)


# ── Pose-derivative segments ────────────────────────────────────────────────────────────
def body_delta(start: Sequence[float], end: Sequence[float]) -> tuple:
    """World displacement rotated into the body frame the segment was flown in.

    ``start``/``end`` are ``(x, y, yaw_rad)``. Uses the MEAN of the start and end yaw
    rather than either endpoint: over a segment that turns, either endpoint attributes
    part of the arc to the wrong axis, and the mean is the first-order correction.

    Same construction as the Go2's ``visual_nav/lateral_floor_probe.py``, deliberately --
    the two robots' lateral numbers are only comparable if the frames are.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    yaw = math.atan2(math.sin(start[2]) + math.sin(end[2]),
                     math.cos(start[2]) + math.cos(end[2]))
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


@dataclass(frozen=True)
class Segment:
    """One commanded interval, measured from pose.

    ``role`` is what the segment is *for*, and it decides which refusal applies to it:

    ``anchor``
        commanded at a speed already known to walk this robot. If an anchor does not
        travel, the legs were not walking and nothing else in the run means anything.
    ``control``
        commanded zero. Interleaved *between* treatments, never recorded once at the
        start: a warm robot walks differently from a cold one, and a contemporaneous
        control is the only thing that separates the treatment from the drift.
    ``treatment``
        the setting actually under test. A treatment that does not travel may be a real
        finding -- that is what a floor *is* -- so the no-motion refusal must never be
        applied to one.

    ``estimator_forward_mps`` is the platform's own body-velocity estimate, averaged over
    the segment. It is recorded for comparison and is never fitted against: on the G1 a
    0.45 actuator gain was read as 0.17 m/s of noise by trusting that field.
    """

    role: str
    commanded_vx: float
    commanded_vy: float
    elapsed_s: float
    forward_m: float
    lateral_m: float
    yaw_change_deg: float
    estimator_forward_mps: float
    estimator_samples: int
    estimator_failures: int

    @property
    def forward_mps(self) -> float:
        return self.forward_m / self.elapsed_s

    @property
    def lateral_mps(self) -> float:
        return self.lateral_m / self.elapsed_s

    @property
    def travelled(self) -> bool:
        """Whether this segment moved further forward than the method can resolve."""
        return abs(self.forward_m) >= WALKED_MARGIN_M

    def as_dict(self) -> dict:
        data = asdict(self)
        data.update({"forward_mps": self.forward_mps, "lateral_mps": self.lateral_mps,
                     "travelled": self.travelled})
        return data


def run_segment(loco, *, role: str, vx: float, vy: float, duration_s: float,
                tick_s: float, clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep) -> Segment:
    """Hold ``(vx, vy)`` for ``duration_s`` and measure what the body actually did.

    The command is re-sent every tick because the vendor high-level interface is
    edge-triggered -- it applies what it last received -- so a single send would be
    indistinguishable from a dropped datagram for the rest of the segment.

    Displacement comes from ``loco.pose()`` at the two ends, not from integrating
    ``loco.velocity()``: a velocity estimate integrated over a segment carries its bias
    into the answer, while pose endpoints carry only their own error.
    """
    start_pose = loco.pose()
    start_t = clock()
    estimates: list = []
    failures = 0
    while clock() - start_t < duration_s:
        loco.set_velocity(vx, vy, 0.0)
        try:
            estimates.append(float(loco.velocity()[0]))
        except Exception:
            # The estimate is a cross-check, never the measurement. Losing it must not
            # end a segment the robot is halfway through, but it is counted rather than
            # swallowed, so a run whose estimator never answered is visible in the record.
            failures += 1
        sleep(tick_s)
    end_pose = loco.pose()
    elapsed = clock() - start_t
    if elapsed <= 0.0:
        raise Refusal("a segment reported zero elapsed time; the clock is not advancing")
    forward, lateral = body_delta((start_pose.x, start_pose.y, start_pose.yaw),
                                  (end_pose.x, end_pose.y, end_pose.yaw))
    return Segment(
        role=role, commanded_vx=vx, commanded_vy=vy, elapsed_s=elapsed,
        forward_m=forward, lateral_m=lateral,
        yaw_change_deg=math.degrees(end_pose.yaw - start_pose.yaw),
        estimator_forward_mps=(sum(estimates) / len(estimates)) if estimates else math.nan,
        estimator_samples=len(estimates), estimator_failures=failures,
    )


def check_anchors_walked(segments: Iterable[Segment]) -> None:
    """Refuse the whole run if a segment commanded at a known-good speed did not travel.

    This is the Go2 lesson, carried across. ``lateral_floor_probe.py`` once called only
    ``stand()`` where getting up needs ``recover()`` *and* ``stand()``; the robot stayed
    prone, every command was silently ignored, and the run reported 0.000 m/s on every
    axis -- which reads exactly like "the floor is real and total" rather than like
    "nothing moved".

    The Go2 probe could apply that check to every segment because it held ``vx`` at a
    known-good speed throughout and only varied ``vy``. Here ``vx`` is itself the thing
    under test, so a treatment that does not travel is a legitimate result and the check
    would refuse every real measurement. The anchors exist precisely so the check still
    has something it can be applied to.
    """
    anchors = [segment for segment in segments if segment.role == "anchor"]
    if not anchors:
        raise Refusal(
            "this run has no anchor segment, so there is nothing that proves the legs "
            "were walking at all; a run of zeros would be indistinguishable from a floor"
        )
    dead = [segment for segment in anchors if not segment.travelled]
    if dead:
        raise Refusal(
            f"{len(dead)} of {len(anchors)} anchor segments travelled less than "
            f"{WALKED_MARGIN_M:.2f} m even though they were commanded at a speed this "
            f"robot is known to walk at. The legs were not walking, so every zero below "
            f"is a dead robot rather than a floor. Check the stand sequence, the control "
            f"mode, and that the operator handed the robot over."
        )


def check_every_segment_walked(segments: Iterable[Segment]) -> None:
    """The Go2's refusal, unmodified, for a phase where every segment commands a walk.

    ``lateral_floor_probe.py`` holds ``vx`` at a known-walking speed for every segment so
    that ``vy`` is the only variable. In that shape a segment with no forward travel can
    only mean the legs were not running, so the probe refuses to tabulate the run. The
    diagonal phase here and the whole of ``actuator_gain_probe.py`` have that same shape,
    so they get that same refusal with nothing weakened.
    """
    segments = list(segments)
    if not segments:
        raise Refusal("no segments were recorded, so there is nothing to tabulate")
    dead = [segment for segment in segments if not segment.travelled]
    if dead:
        raise Refusal(
            f"{len(dead)} of {len(segments)} segments travelled less than "
            f"{WALKED_MARGIN_M:.2f} m forward, even though every segment in this phase "
            f"was commanded forward at or above a speed this robot walks at. The legs "
            f"were not walking, so a zero here means nothing at all -- and it reads "
            f"exactly like a floor that is real and total. Refusing to tabulate."
        )


def check_controls_are_still(segments: Iterable[Segment], axis: str = "forward") -> None:
    """Refuse if a control drifted, on the axis under test, as far as a walk has to travel.

    ``axis`` is the axis the phase is measuring: a forward-ladder control is commanded
    zero on both axes and is checked forward, while a diagonal control is commanded
    *forward at the measured floor* with zero lateral and is checked laterally. Checking
    the wrong axis would refuse every diagonal run for successfully walking forwards.

    When this fires, ``WALKED_MARGIN_M`` is too small for this unit and every "did it
    walk?" verdict in the run is a coin toss. It is the guard that stops that threshold
    from being an article of faith.
    """
    if axis not in ("forward", "lateral"):
        raise Refusal(f"unknown control axis {axis!r}")
    controls = [segment for segment in segments if segment.role == "control"]
    drifted = [segment for segment in controls
               if abs(getattr(segment, f"{axis}_m")) >= WALKED_MARGIN_M]
    if drifted:
        worst = max(abs(getattr(segment, f"{axis}_m")) for segment in drifted)
        raise Refusal(
            f"{len(drifted)} of {len(controls)} control segments drifted {worst:.3f} m "
            f"{axis}, which is at or past the {WALKED_MARGIN_M:.2f} m a segment has to "
            f"travel to count as walking. This run cannot tell motion from odometry "
            f"drift. Lengthen --segment, or fix the odometry, before believing any "
            f"number measured this way."
        )


def control_baseline(segments: Iterable[Segment], axis: str = "lateral") -> float:
    """Mean per-second travel of the zero-command controls on one axis.

    Subtracted from a treatment so the reported delivery is net of drift.
    """
    values = [getattr(segment, f"{axis}_mps") for segment in segments
              if segment.role == "control"]
    return sum(values) / len(values) if values else 0.0


# ── Fits ────────────────────────────────────────────────────────────────────────────────
def fit_ratio(pairs: Sequence) -> dict:
    """Least-squares ``delivered = gain x commanded`` through the origin, with residual.

    Through the origin because a commanded zero must deliver zero: an intercept term
    would let the fit buy accuracy at the demo envelope by claiming the robot creeps when
    it is asked for nothing, which is both false and the wrong direction to be wrong in.

    Returns the fitted ``gain``, the residual RMS in m/s, and the spread of the individual
    point ratios. The spread is there because a residual can be small while the ratios
    fan out -- one number that looks tight and one that does not is a confound worth
    seeing before the gain is written down.
    """
    pairs = [(float(commanded), float(delivered)) for commanded, delivered in pairs]
    if not pairs:
        raise Refusal("no command/delivery pairs survived; there is nothing to fit")
    sum_cc = sum(commanded * commanded for commanded, _ in pairs)
    if sum_cc <= 0.0:
        raise Refusal("every commanded speed in the fit was zero; a gain is undefined")
    gain = sum(commanded * delivered for commanded, delivered in pairs) / sum_cc
    residuals = [delivered - gain * commanded for commanded, delivered in pairs]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    ratios = sorted(delivered / commanded for commanded, delivered in pairs
                    if commanded != 0.0)
    return {
        "gain": gain,
        "residual_rms_m_s": rms,
        "samples": len(pairs),
        "ratio_min": ratios[0] if ratios else math.nan,
        "ratio_median": ratios[len(ratios) // 2] if ratios else math.nan,
        "ratio_max": ratios[-1] if ratios else math.nan,
    }


# ── The commissioning record ────────────────────────────────────────────────────────────
@dataclass
class Record:
    """One robot's commissioning artefact.

    ``provenance`` starts at :data:`PROVISIONAL` and only a human moves it. Everything
    else in here is written by a probe.
    """

    robot_id: str
    context: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    provenance: str = PROVISIONAL
    reviewed_by: object = None
    reviewed_utc: object = None
    schema: str = SCHEMA
    recorded_utc: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def utc_now() -> str:
    """Wall-clock UTC, for the record only.

    Deliberately not used for any interval: every duration in this directory is measured
    on a monotonic clock, because mixing the two produces ages of -1.8e9 seconds and gates
    that fail open.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_record(robot_id: str, **context) -> Record:
    """Start a provisional record. There is no way to construct a reviewed one."""
    if not robot_id or not robot_id.strip():
        raise Refusal(
            "--robot-id is required on every measurement. Issue #13 says it plainly: do "
            "not copy values between the two Ventures, so a number without a robot on it "
            "is not a measurement."
        )
    return Record(robot_id=robot_id.strip(), context=dict(context), recorded_utc=utc_now())


def merge_measurement(record: Record, name: str, payload: dict) -> Record:
    """Attach one probe's result, stamped, without disturbing the others."""
    record.measurements[name] = {"measured_utc": utc_now(), **payload}
    return record


def write_record(path, record: Record) -> Path:
    """Write the artefact. Always provisional unless a human has already reviewed it."""
    destination = Path(path)
    destination.write_text(
        json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_record(path) -> Record:
    """Read an artefact back, refusing anything this reader would misinterpret."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refusal(f"cannot read commissioning record {source}: {error}") from None
    if not isinstance(data, dict):
        raise Refusal(f"{source} must contain a JSON object")
    if data.get("schema") != SCHEMA:
        raise Refusal(
            f"{source} is schema {data.get('schema')!r}, not {SCHEMA!r}; this reader "
            f"would misread it"
        )
    known = {"robot_id", "context", "measurements", "provenance", "reviewed_by",
             "reviewed_utc", "schema", "recorded_utc"}
    return Record(**{key: value for key, value in data.items() if key in known})


def require_reviewed(path) -> Record:
    """Return the record only if a human has reviewed it. Refuse otherwise.

    This is the gate that makes ``provenance`` mean something. It matches how
    ``Lite3Bindings.validate_camera_calibration`` treats a calibration file -- a file that
    is not what it claims to be stops the run rather than being used with a warning -- and
    it is why ``commission.py --emit-flags`` cannot print a ``--gait-floor`` for a
    provisional record. An unreviewed number cannot become a live-movement flag by
    accident; it takes a person putting their name on it.
    """
    record = read_record(path)
    if record.provenance != REVIEWED:
        raise Refusal(
            f"{Path(path)} is marked {record.provenance!r}. Nothing marked "
            f"{PROVISIONAL!r} may be used for live movement. A human reads the numbers, "
            f"agrees they describe this robot, then runs:\n"
            f"    python3 commission.py --record {Path(path)} --review 'YOUR NAME'"
        )
    if not isinstance(record.reviewed_by, str) or not record.reviewed_by.strip():
        raise Refusal(
            f"{Path(path)} claims to be {REVIEWED!r} but names no reviewer; a review "
            f"nobody signed is not a review"
        )
    return record


def paste_block(title: str, rows: Sequence, notes: Sequence = ()) -> str:
    """Render a result as the Markdown table issue #13 asks for.

    Self-contained on purpose: no images and no ``user-attachments`` links, because those
    are slow or unreachable from mainland China and the argument has to survive without
    them.
    """
    lines = [f"### {title}", "", "| item | value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if notes:
        lines.append("")
        lines += list(notes)
    return "\n".join(lines)


def print_paste_block(text: str, printer: Callable[[str], None] = print) -> None:
    printer("")
    printer("-" * 78)
    printer("PASTE THIS INTO ISSUE #13")
    printer("-" * 78)
    printer(text)
    printer("-" * 78)


def run_main(entry: Callable[[], int], prefix: str,
             printer: Callable[[str], None] = print) -> int:
    """Turn a :class:`Refusal` into a banner and an exit code, not a traceback."""
    try:
        return entry()
    except Refusal as refusal:
        printer("")
        printer("!" * 78)
        printer(f"[{prefix}] REFUSING TO REPORT A NUMBER")
        for line in str(refusal).splitlines():
            printer("    " + line)
        printer("!" * 78)
        return 2


@contextlib.contextmanager
def stopped_afterwards(loco):
    """Guarantee a stop, and only then let the caller's exception out."""
    try:
        yield loco
    finally:
        with contextlib.suppress(Exception):
            loco.stop()
