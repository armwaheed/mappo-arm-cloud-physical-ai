#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the look behind the robot that authorises a flip at the goal.

No robot, no camera, no network. Two halves, and they are tested differently.

THE DECISION half -- ``Clearance``, ``RearView``, ``flip_arguments`` -- is pure, so every
way a look can fail to be evidence is enumerated here rather than discovered on a floor
somebody cleared. The property under test is always the same one: **absence refuses**. No
perception, stale perception, a detector that produced nothing, a turn that did not
finish, a box ranging could not place, a dry run -- each of them must be "do not flip",
never "assume clear", because there is no second safety layer downstream of this one.

THE RANGING half is NOT faked. ``scan_rear`` is run against the REAL
``person_detector.range_detections`` and a real ``FisheyeCamera``, with synthetic boxes of
known size at known distances. Substituting a fake ranger would test that this file calls
something, which is not the claim; the claim is that the metres it compares against
``flourish``'s clearance floor are the metres the drive itself would have computed.
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROBOT_STACK = _HERE.parents[2]
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
_LOCOMOTION = _HERE.parents[0] / "locomotion"
for _path in (str(_HERE), str(_ROBOT_STACK), str(_COMMON), str(_LOCOMOTION)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flourish

import look_behind
from look_behind import (
    BACKWARD_KINDS,
    CONFIRM_CLEAR_FRAMES,
    GESTURE_UNAVAILABLE,
    SCAN_REQUIRED,
    Clearance,
    RearView,
    arrival_flip,
    clearance_basis,
    flip_arguments,
    frame_stale_s,
    required_clearance_m,
    scan_arguments,
    scan_rear,
)

KIND = BACKWARD_KINDS[0]
REQUIRED_M = required_clearance_m(KIND)


# ── fixtures ────────────────────────────────────────────────────────────────
def _view(**overrides) -> RearView:
    """A look that saw a clear space, overridable one field at a time."""
    fields = {"frames": CONFIRM_CLEAR_FRAMES, "clear_frames": CONFIRM_CLEAR_FRAMES,
              "stalest_frame_s": 0.05, "nearest_m": math.inf,
              "nearest_label": "nothing the detector can see", "unrangeable": 0,
              "camera_errors": 0, "detail": "looked"}
    fields.update(overrides)
    return RearView(**fields)


def _clear(**overrides) -> Clearance:
    """A clearance that IS authorised, so a test can break exactly one thing about it."""
    fields = {"kind": KIND, "required_m": REQUIRED_M, "live": True, "turned_out": True,
              "turned_back": True, "view": _view(),
              "operator_rear_clearance_m": REQUIRED_M + 1.0,
              "battery_floor_pct": 60.0, "hold_seconds": 7.0}
    fields.update(overrides)
    return Clearance(**fields)


def _args(**overrides) -> argparse.Namespace:
    fields = {"flourish": True, "flourish_lane_width": 3.0, "robot_id": "LITE3-A",
              "firmware": "V1.0.8", "payload": "none", "flip_on_arrival": True,
              "flip_kind": KIND, "flip_rear_clearance_metres": REQUIRED_M + 1.0,
              "flip_battery_floor_pct": 60.0, "flip_hold_seconds": 7.0}
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _drive(*, live=True, extra=()) -> list:
    command = [sys.executable, "-u", "mappo_drive.py", "--arrive", "0.5",
               "--locomotion-transport", "axis", "--motion-host", "127.0.0.1",
               "--camera-source", "rtsp://robot/stream",
               "--calibration", "/tmp/lite3_front_camera.json", *extra]
    return command + (["--live"] if live else [])


class _Runner:
    """A gesture runner that records what it was asked to fire and answers with codes."""

    def __init__(self, codes=None) -> None:
        #: (kind, extra) in order.
        self.fired: list = []
        #: kind -> exit code, defaulting to 0.
        self.codes = codes or {}

    def __call__(self, kind, extra=()):
        self.fired.append((kind, tuple(extra)))
        return self.codes.get(kind, 0)

    @property
    def kinds(self) -> list:
        return [kind for kind, _extra in self.fired]


def _scanner(view: RearView):
    def scan(_settings, _required):
        return view
    return scan


def _recorder(sink: list):
    """A stand-in for ``subprocess.run`` that records the argv and claims success."""
    def run(argv, **_kwargs):
        sink.append(list(argv))
        return types.SimpleNamespace(returncode=0)
    return run


# ── the clearance number is not invented ────────────────────────────────────
def test_the_required_distance_comes_from_flourish_and_not_from_here():
    """⛔ THE CONSTRAINT. The flip's travel is UNMEASURED -- the pose channel freezes for
    the manoeuvre -- so the only defensible floor is the one `flourish` already refuses
    below, and it must be READ rather than copied so the two cannot drift apart."""
    for kind in BACKWARD_KINDS:
        assert required_clearance_m(kind) == flourish.VENDOR_ACTIONS[kind].rear_clearance_m
        assert clearance_basis(kind) == flourish.VENDOR_ACTIONS[kind].rear_basis
    # And no distance literal of its own: the module may not carry the number.
    source = (_HERE / "look_behind.py").read_text()
    tree = ast.parse(source, filename="look_behind.py")
    floors = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Constant)
              and isinstance(node.value, float)
              and node.value == flourish.BACKFLIP_TRAVEL_M]
    assert not floors, "the clearance floor is written out as a literal somewhere"


def test_a_kind_whose_travel_direction_is_unknown_is_not_offered():
    """A look BEHIND is evidence about the space behind. `twist-jump`'s own entry says its
    travel is unknown "in any direction", so no rear view is evidence about it, and the
    set is derived from flourish's travel strings rather than typed out here."""
    unknown = [kind for kind, action in flourish.VENDOR_ACTIONS.items()
               if "BACKWARD" not in action.travel]
    assert unknown, "the fixture is stale: every kind now claims a backward travel"
    for kind in unknown:
        assert kind not in BACKWARD_KINDS, kind
        assert not _clear(kind=kind).authorised
        assert any("evidence about" in reason
                   for reason in _clear(kind=kind).refusals)
    assert set(BACKWARD_KINDS) <= set(flourish.OPERATOR_ONLY_KINDS)


# ── default deny ────────────────────────────────────────────────────────────
def test_a_clearance_built_with_nothing_refuses_everything():
    """⛔ THE DEFAULT IS DENY, and it is checked on the object rather than argued about in
    a docstring: a field this file forgets to fill must refuse, not pass."""
    empty = Clearance()
    assert not empty.authorised
    for expected in ("evidence about", "--live", "outward", "return", "usable frame",
                     "--flip-rear-clearance-metres", "--flip-battery-floor-pct",
                     "--flip-hold-seconds"):
        assert any(expected in reason for reason in empty.refusals), expected
    assert not RearView().unanimous


def test_the_fixture_itself_is_authorised_so_the_negatives_mean_something():
    """Every test below breaks ONE thing about this. If it were refused for some unrelated
    reason they would all pass while proving nothing."""
    assert _clear().authorised, _clear().refusals


def test_no_perception_refuses():
    for view in (RearView(),
                 _view(frames=0, clear_frames=0, stalest_frame_s=math.inf, nearest_m=0.0),
                 _view(frames=CONFIRM_CLEAR_FRAMES - 1,
                       clear_frames=CONFIRM_CLEAR_FRAMES - 1)):
        clearance = _clear(view=view)
        assert not clearance.authorised, view
        assert any("usable frame" in reason for reason in clearance.refusals)


def test_stale_perception_refuses_at_the_navigators_own_limit():
    """The limit is `NavConfig.perception_timeout_s`, past which the DRIVE stops the legs
    rather than acting on the belief. A frame too old to steer on is too old to flip on,
    and reading the navigator's number means the two cannot drift apart."""
    limit = frame_stale_s()
    assert _clear(view=_view(stalest_frame_s=limit)).authorised
    stale = _clear(view=_view(stalest_frame_s=limit + 0.01))
    assert not stale.authorised
    assert any("too old to flip on" in reason for reason in stale.refusals)


def test_an_obstacle_behind_refuses_and_says_what_and_how_far():
    close = _clear(view=_view(nearest_m=REQUIRED_M - 0.01, nearest_label="person",
                              clear_frames=0))
    assert not close.authorised
    assert any("person" in reason and "needs" in reason for reason in close.refusals)
    # ...and the boundary is not off by one: exactly the floor passes.
    assert _clear(view=_view(nearest_m=REQUIRED_M)).authorised


def test_one_dissenting_frame_is_enough_to_refuse():
    """The agreement is required on the CLEAR verdict. A single frame that put something in
    the way outvotes the rest, because the frame that MISSES the chair is the failure mode
    that matters when the consequence cannot be interrupted."""
    split = _clear(view=_view(clear_frames=CONFIRM_CLEAR_FRAMES - 1))
    assert not split.authorised
    assert any("put something inside" in reason for reason in split.refusals)


def test_a_box_that_could_not_be_ranged_is_treated_as_a_box_in_the_way():
    """⛔ FAILING OPEN IS ISSUE #72's COLLISION. A detection with no usable range leaves
    nothing behind, and an empty obstacle list is what a planner reads as an open world."""
    blind = _clear(view=_view(unrangeable=1))
    assert not blind.authorised
    assert any("could not be given a range" in reason for reason in blind.refusals)


def test_a_turn_that_did_not_complete_refuses():
    for broken in ({"turned_out": False}, {"turned_back": False}):
        clearance = _clear(**broken)
        assert not clearance.authorised, broken
    assert any("outward" in reason for reason in _clear(turned_out=False).refusals)
    assert any("return" in reason for reason in _clear(turned_back=False).refusals)


def test_a_dry_run_is_not_a_clearance_however_the_turns_exited():
    """⛔ THE MEASUREMENT OF 2026-09-04. Without `--live`, flourish prints its plan and
    exits 0 WITHOUT TURNING, so exit codes alone read a dry run as two completed turns."""
    dry = _clear(live=False)
    assert not dry.authorised
    assert any("--live" in reason and "turned NOTHING" in reason
               for reason in dry.refusals)


def test_the_operators_tape_measure_is_still_required_and_still_a_floor():
    """The look does not replace it and must never be described as replacing it: the
    detector finds VOC classes, and a wall is not one."""
    assert not _clear(operator_rear_clearance_m=None).authorised
    short = _clear(operator_rear_clearance_m=REQUIRED_M - 0.01)
    assert not short.authorised
    assert any("--flip-rear-clearance-metres says" in reason
               for reason in short.refusals)
    assert _clear(operator_rear_clearance_m=REQUIRED_M).authorised


def test_flourishs_own_no_default_preconditions_are_not_defaulted_here_either():
    for missing in ("battery_floor_pct", "hold_seconds"):
        clearance = _clear(**{missing: None})
        assert not clearance.authorised, missing


# ── the seam ────────────────────────────────────────────────────────────────
def test_an_unauthorised_clearance_cannot_produce_a_flip_command():
    """⛔ THE SEAM, TESTED AS A FUNCTION. `flip_arguments` recomputes the authorisation
    rather than trusting the object, so every broken clearance above must raise here too --
    including a hand-assembled one, since there is no boolean to set."""
    broken = [Clearance(), _clear(live=False), _clear(turned_out=False),
              _clear(turned_back=False), _clear(view=RearView()),
              _clear(view=_view(unrangeable=2)),
              _clear(view=_view(nearest_m=0.4, clear_frames=0)),
              _clear(view=_view(stalest_frame_s=frame_stale_s() + 1.0)),
              _clear(operator_rear_clearance_m=None), _clear(battery_floor_pct=None),
              _clear(hold_seconds=None), _clear(kind="twist-jump")]
    for clearance in broken:
        try:
            flip_arguments(clearance)
        except flourish.Refusal as refusal:
            assert "unauthorised look" in str(refusal), refusal
        else:
            raise AssertionError(f"a flip command was built from {clearance.refusals}")


def test_an_authorised_clearance_carries_every_precondition_flourish_demands():
    """The command that comes out has to be one flourish will actually accept, or the whole
    sequence ends in a refusal after the robot has turned twice."""
    extra = flip_arguments(_clear())
    for flag in ("--operator-triggered", "--rear-clearance-metres",
                 "--acrobatic-battery-floor-pct", "--action-hold-seconds"):
        assert flag in extra, (flag, extra)
    stated = float(extra[extra.index("--rear-clearance-metres") + 1])
    assert stated >= required_clearance_m(KIND)


def test_the_flip_command_flourish_would_be_given_is_actually_accepted_by_it():
    """End to end against the REAL flourish CLI, dry: the flags this seam emits, through
    `mission.flourish_command`, must reach `--live` rather than a refusal. It is the
    complement of `test_flourish.py`'s "a mission-shaped invocation is refused" -- that one
    pins that nothing else gets through, this one pins that this does."""
    import mission

    argv = mission.flourish_command(_drive(live=False), KIND, _args(),
                                    flip_arguments(_clear()))
    assert argv is not None
    result = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert "nothing was sent" in result.stdout, result.stdout
    assert "VENDOR CANNED ACTION" in result.stdout


# ── the sequence ────────────────────────────────────────────────────────────
def test_default_off_touches_nothing_at_all():
    """⛔ A RUN THAT DID NOT ASK FOR A FLIP MUST BEHAVE AS IT DID BEFORE THIS EXISTED."""
    runner = _Runner()
    said: list = []
    for off in (_args(flip_on_arrival=False), argparse.Namespace()):
        assert arrival_flip(_drive(), off, runner, printer=said.append) is None
    assert runner.fired == [], runner.fired
    assert said == [], said


def test_a_clear_space_turns_around_looks_turns_back_and_flips_in_that_order():
    runner = _Runner()
    clearance = arrival_flip(_drive(), _args(), runner, printer=lambda _line: None,
                             scanner=_scanner(_view()))
    assert clearance is not None and clearance.authorised, clearance.refusals
    assert runner.kinds == [flourish.Flourish.LOOK, flourish.Flourish.LOOK, KIND], \
        runner.kinds
    assert runner.fired[0][1] == ("--degrees", "180.0"), runner.fired[0]
    assert runner.fired[1][1] == ("--degrees", "-180.0"), runner.fired[1]
    assert "--operator-triggered" in runner.fired[2][1]


def test_an_obstacle_behind_still_turns_back_and_does_not_flip():
    """The return turn is not conditional on the verdict. A robot left facing backward at
    the goal is a robot pointing its one camera away from everything."""
    runner = _Runner()
    clearance = arrival_flip(
        _drive(), _args(), runner, printer=lambda _line: None,
        scanner=_scanner(_view(nearest_m=0.6, nearest_label="chair", clear_frames=0)))
    assert runner.kinds == [flourish.Flourish.LOOK, flourish.Flourish.LOOK]
    assert not clearance.authorised


def test_a_failed_outward_turn_stops_everything_and_does_not_guess_a_correction():
    """A `look` that refused part way stopped the legs at a heading nothing here knows. A
    -180 from an unknown heading is not a return, it is a second guess."""
    runner = _Runner(codes={flourish.Flourish.LOOK: 1})
    said: list = []
    clearance = arrival_flip(_drive(), _args(), runner, printer=said.append,
                             scanner=_scanner(_view()))
    assert runner.kinds == [flourish.Flourish.LOOK], "it turned again after a failed turn"
    assert not clearance.authorised
    assert any("HEADING IS NOW UNKNOWN" in line for line in said), said


def test_a_failed_return_turn_refuses_the_flip():
    codes = {}
    calls = {"n": 0}

    def runner(kind, extra=()):
        calls["n"] += 1
        codes.setdefault(kind, 0)
        return 0 if calls["n"] == 1 else 1

    fired: list = []
    original = look_behind.flip_arguments
    look_behind.flip_arguments = lambda clearance: fired.append(clearance) or ()
    try:
        clearance = arrival_flip(_drive(), _args(), runner,
                                 printer=lambda _line: None, scanner=_scanner(_view()))
    finally:
        look_behind.flip_arguments = original
    assert not clearance.authorised and not fired, fired
    assert any("return" in reason for reason in clearance.refusals)


def test_a_detector_that_will_not_load_is_a_refusal_and_not_a_clearance():
    """Every way the look can fail -- no model, no calibration, a camera that never opened
    -- arrives at one broad catch, and that catch FAILS CLOSED."""
    def explode(_settings, _required):
        raise FileNotFoundError("MobileNetSSD_deploy.caffemodel not found")

    runner = _Runner()
    said: list = []
    clearance = arrival_flip(_drive(), _args(), runner, printer=said.append,
                             scanner=explode)
    assert runner.kinds == [flourish.Flourish.LOOK, flourish.Flourish.LOOK]
    assert not clearance.authorised
    assert any("refusal, not a" in line for line in said), said


def test_a_dry_run_looks_at_nothing_and_never_reaches_the_flip():
    runner = _Runner()
    clearance = arrival_flip(_drive(live=False), _args(), runner,
                             printer=lambda _line: None)
    assert KIND not in runner.kinds, runner.kinds
    assert not clearance.authorised
    assert any("--live" in reason for reason in clearance.refusals)


def test_an_unanswered_flip_setting_skips_the_sequence_rather_than_defaulting_it():
    for missing in ("flip_kind", "flip_rear_clearance_metres", "flip_battery_floor_pct",
                    "flip_hold_seconds"):
        runner = _Runner()
        said: list = []
        assert arrival_flip(_drive(), _args(**{missing: None}), runner,
                            printer=said.append) is None
        assert runner.fired == [], (missing, runner.fired)
        assert any(missing.replace("_", "-") in line for line in said), (missing, said)


def test_a_drive_with_no_camera_or_calibration_cannot_be_looked_through():
    """Without `--calibration` the camera model falls back to the GO2's nominal lens, and
    every range compared against a clearance floor would be a different robot's."""
    for flag in SCAN_REQUIRED:
        command = [token for token in _drive() if token != flag]
        # ...and its value, which is now orphaned.
        command = [token for index, token in enumerate(command)
                   if not (index and command[index - 1] not in ("--live",)
                           and token.startswith(("rtsp://", "/tmp/")))]
        runner = _Runner()
        said: list = []
        assert arrival_flip(command, _args(), runner, printer=said.append) is None
        assert runner.fired == [], (flag, runner.fired)
        assert any(flag in line for line in said), (flag, said)


def test_a_gesture_that_could_not_be_built_at_all_refuses():
    runner = _Runner(codes={flourish.Flourish.LOOK: GESTURE_UNAVAILABLE})
    clearance = arrival_flip(_drive(), _args(), runner, printer=lambda _line: None,
                             scanner=_scanner(_view()))
    assert runner.kinds == [flourish.Flourish.LOOK]
    assert not clearance.authorised


def test_the_console_block_says_what_a_voc_detector_cannot_see():
    """The operator reading the terminal cannot read this file, and "CLEAR" on its own is
    the sentence that would let somebody flip into a wall."""
    said = _clear().describe()
    assert "WALL IS NOT A VOC CLASS" in said, said
    assert "not waived" in said, said
    refused = _clear(view=_view(unrangeable=3)).describe()
    assert "NO FLIP" in refused and "could not be given a range" in refused


def test_a_look_that_never_happened_does_not_print_as_a_measurement():
    """⚠️ THE DENY DEFAULTS ARE `0.00 m` AND `inf`, and printed as numbers they read as a
    measurement of an empty room. An operator scanning the block must not mistake "nothing
    was looked at" for "nothing is there" -- those are the two states this file exists to
    keep apart."""
    said = _clear(view=RearView()).describe()
    assert "NOTHING WAS LOOKED AT" in said, said
    assert "0.00 m" not in said, said
    assert "no frame at all" in said, said
    assert any("no nearest anything" in reason
               for reason in _clear(view=RearView()).refusals)
    # ...and a look that DID happen and found nothing still says so as a measurement.
    looked = _clear(view=_view()).describe()
    assert "nothing the detector can see" in looked, looked


# ── the perception is the drive's own ───────────────────────────────────────
def _camera_and_model(boxes_per_frame, *, ages=None, width=1280, height=720):
    """A camera and camera model that feed REAL ranging synthetic boxes.

    ``boxes_per_frame`` is a list of lists of ``Detection``. Nothing here fakes
    ``range_detections``: the metres this test asserts on are the metres
    ``visual_nav.PerceptionWorker`` would have computed from the same boxes.
    """
    from camera_model import FisheyeCamera

    model = FisheyeCamera.from_hfov(width, height, 134.0, height_m=0.32)
    clock = {"t": 100.0}
    frames = list(boxes_per_frame)
    ages = list(ages or [0.05] * len(frames))

    class _Frame:
        def __init__(self, seq, image, capture_time):
            self.seq, self.image, self.capture_time = seq, image, capture_time

    class _Image:
        shape = (height, width, 3)

    class _Camera:
        error_count = 0

        def __init__(self):
            self.seq = 0

        def wait_for_new(self, after_seq, timeout):
            if self.seq >= len(frames):
                clock["t"] += timeout
                return None
            self.seq += 1
            return _Frame(self.seq, _Image(), clock["t"] - ages[self.seq - 1])

    class _Detector:
        def detect(self, _image):
            return frames[camera.seq - 1]

    camera = _Camera()
    return camera, _Detector(), model, lambda: clock["t"]


def _box_at(model, range_m: float, height_m: float = 1.70, label: str = "person"):
    """A ``Detection`` whose HEIGHT prior ranges to ``range_m`` through the real model.

    Under the equidistant model the box's angular height is ``height_m / range_m``, so its
    pixel height is ``focal_px * that``. Built from the model rather than from a
    remembered pixel count, so it stays true if the fixture's field of view changes.
    """
    from person_detector import Detection

    span_px = model.focal_px * (height_m / range_m)
    cx, cy = model.width / 2.0, model.height / 2.0
    return Detection(x1=cx - span_px / 6.0, y1=cy - span_px / 2.0,
                     x2=cx + span_px / 6.0, y2=cy + span_px / 2.0,
                     score=0.9, label=label)


def test_the_scan_ranges_with_the_stacks_own_estimator_and_not_a_copy():
    """The metres compared against flourish's floor must be the metres the drive would
    have computed. Real `range_detections`, real `FisheyeCamera`, synthetic boxes."""
    from camera_model import FisheyeCamera
    from person_detector import SizePrior, estimate_range

    reference = FisheyeCamera.from_hfov(1280, 720, 134.0, height_m=0.32)
    # Two boxes, one on each side of the floor flourish demands, both far enough off the
    # lens that a 1.70 m person still fits the frame so both take the estimator's HEIGHT
    # path. What is asserted is not a nominal distance -- the equidistant model and the
    # small-angle box this fixture draws disagree by a few percent at these angles -- but
    # that `scan_rear` reports EXACTLY what `estimate_range` reported. That is the claim:
    # the metres compared against the clearance floor are the stack's own metres.
    for nominal, expect_clear in ((3.00, True), (1.40, False)):
        box = _box_at(reference, nominal)
        expected, source = estimate_range(box, reference, SizePrior())
        assert source == "height", (nominal, source)
        assert (expected >= REQUIRED_M) is expect_clear, (nominal, expected)
        camera, detector, model, clock = _camera_and_model([[box]] * CONFIRM_CLEAR_FRAMES)
        view = scan_rear(camera=camera, detector=detector, camera_model=model,
                         prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
        assert view.frames == CONFIRM_CLEAR_FRAMES, view
        assert view.nearest_m == expected, (nominal, view.nearest_m, expected)
        assert view.unanimous is expect_clear, (nominal, view)
        assert _clear(view=view).authorised is expect_clear, (nominal, view)


def test_a_person_close_enough_to_clip_the_frame_still_refuses():
    """The prior switches from height to WIDTH when the box runs off the top and bottom,
    and this platform's 720-pixel frame clips a standing adult from about 1.33 m in --
    which is INSIDE the clearance the flip needs. So the near case is the clipped case,
    and it must still refuse rather than falling into a gap between the two priors."""
    from camera_model import FisheyeCamera
    from person_detector import SizePrior

    reference = FisheyeCamera.from_hfov(1280, 720, 134.0, height_m=0.32)
    camera, detector, model, clock = _camera_and_model(
        [[_box_at(reference, 0.9)]] * CONFIRM_CLEAR_FRAMES)
    view = scan_rear(camera=camera, detector=detector, camera_model=model,
                     prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
    assert view.clear_frames == 0, view
    assert view.nearest_m < REQUIRED_M, view.nearest_m
    assert not _clear(view=view).authorised


def test_an_empty_frame_reads_as_clear_and_reports_no_nearest():
    from person_detector import SizePrior

    camera, detector, model, clock = _camera_and_model([[]] * CONFIRM_CLEAR_FRAMES)
    view = scan_rear(camera=camera, detector=detector, camera_model=model,
                     prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
    assert view.unanimous and math.isinf(view.nearest_m), view
    assert _clear(view=view).authorised


def test_a_frame_filling_box_is_counted_as_unrangeable_rather_than_placed_at_a_constant():
    """`estimate_range` answers `frame-fill` with a CONSTANT when the box runs off every
    edge. A constant cannot move however the robot does, and it is not evidence."""
    from person_detector import Detection, SizePrior

    filling = Detection(x1=0.0, y1=0.0, x2=1280.0, y2=720.0, score=0.9, label="person")
    camera, detector, model, clock = _camera_and_model(
        [[filling]] * CONFIRM_CLEAR_FRAMES)
    view = scan_rear(camera=camera, detector=detector, camera_model=model,
                     prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
    assert view.unrangeable >= CONFIRM_CLEAR_FRAMES, view
    assert view.clear_frames == 0, view
    assert not _clear(view=view).authorised


def test_a_camera_that_stops_delivering_runs_out_of_time_rather_than_deciding():
    from person_detector import SizePrior

    camera, detector, model, clock = _camera_and_model([[]])
    view = scan_rear(camera=camera, detector=detector, camera_model=model,
                     prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
    assert view.frames < CONFIRM_CLEAR_FRAMES, view
    assert "not delivering" in view.detail, view.detail
    assert not _clear(view=view).authorised


def test_an_old_frame_is_carried_through_to_the_verdict():
    from person_detector import SizePrior

    old = frame_stale_s() + 1.0
    camera, detector, model, clock = _camera_and_model(
        [[]] * CONFIRM_CLEAR_FRAMES, ages=[0.05, old, 0.05])
    view = scan_rear(camera=camera, detector=detector, camera_model=model,
                     prior=SizePrior(), required_m=REQUIRED_M, clock=clock)
    assert view.stalest_frame_s > frame_stale_s(), view
    assert abs(view.stalest_frame_s - old) < 1e-6, view
    assert not _clear(view=view).authorised


# ── the settings come from the drive, not from here ─────────────────────────
def test_the_look_is_taken_through_the_runs_own_camera_and_thresholds():
    """Copied FROM the drive command for the reason `_FLOURISH_PASSTHROUGH` is: a scan at a
    different confidence or size prior answers a different question from the one the drive
    spent its whole run answering."""
    command = _drive(extra=["--model-dir", "/opt/models", "--confidence", "0.3",
                            "--input-size", "224", "--obstacle-height", "0.514",
                            "--obstacle-width", "0.31", "--camera-gstreamer",
                            "--classes", "person", "chair"])
    settings, missing = scan_arguments(command)
    assert not missing, missing
    assert settings["--camera-source"] == "rtsp://robot/stream"
    assert settings["--model-dir"] == "/opt/models"
    assert settings["--confidence"] == "0.3" and settings["--input-size"] == "224"
    assert settings["--obstacle-height"] == "0.514"
    assert settings["--camera-gstreamer"] is True
    assert settings["--classes"] == ["person", "chair"], settings["--classes"]


def test_the_scan_opens_the_lite3_camera_and_not_the_vendored_go2_one():
    """⚠️ BOTH TREES SHIP A `camera.py`, and both directories are on this path. A bare
    `import camera` resolves to whichever was inserted last, and the Go2's `Go2Camera`
    speaks DDS and takes an `iface=` -- so the failure would be a TypeError at the moment
    the robot has already turned around. The lite3 modules come by package path."""
    import inspect

    source = inspect.getsource(look_behind.open_scan)
    for module in ("camera", "camera_rectify"):
        assert f"from deep_robotics.lite3.visual_nav.{module} import" in source, module
        assert f"\n    from {module} import" not in source, module
    from deep_robotics.lite3.visual_nav.camera import Lite3Camera

    assert Lite3Camera.__module__.endswith("lite3.visual_nav.camera"), \
        Lite3Camera.__module__
    # And look_behind's own path setup does not move this directory behind the vendored
    # one, which is the state `mission.py` had established before importing it. Asserted on
    # the source because every other module in this suite also edits sys.path, so the live
    # ordering by the time this runs is not look_behind's doing.
    setup = (_HERE / "look_behind.py").read_text()
    assert "for _path in (str(_LOCOMOTION), str(_COMMON), str(_ROBOT_STACK), str(_HERE)):" \
        in setup, "the sys.path order changed; _HERE must be inserted last, so it is first"


def test_a_multi_valued_flag_stops_at_the_next_flag():
    settings, _missing = scan_arguments(
        _drive(extra=["--classes", "person", "chair", "--input-size", "300"]))
    assert settings["--classes"] == ["person", "chair"]
    assert settings["--input-size"] == "300"


# ── the arrival / operator-only seam, pinned from both ends ─────────────────
def test_no_travelling_kind_is_named_anywhere_on_the_arrival_path():
    """⛔ AT LEAST AS STRONG AS `test_flourish.py`'s SCAN, AND IT HAS TO BE: this is the
    file that made a travelling kind reachable from an arrival at all.

    `test_flourish.py` asserts that no travelling kind's NAME appears in `mission.py` or
    `venue_run.py`, and that every constant handed to `play_flourish` is an arrival kind.
    Both still hold. What is new is `run_gesture`, which takes a kind and runs it, so this
    asserts the same property over it: `mission.py` must never name a kind for it either,
    and `look_behind.py` -- which does decide the kind -- must take it from the operator's
    validated `--flip-kind` rather than from a literal of its own.
    """
    mission_source = (_HERE / "mission.py").read_text()
    for kind in flourish.OPERATOR_ONLY_KINDS:
        assert kind not in mission_source, f"mission.py names {kind!r}"
        assert kind not in (_HERE / "venue_run.py").read_text(), f"venue_run.py: {kind!r}"

    # `look_behind.py` is allowed to DISCUSS these kinds -- the argument for excluding the
    # one whose travel is unknown is half the reason the file exists -- but it may not
    # carry one as a value. A string constant that IS a kind is the only thing that could
    # reach a command line, so that is what is scanned for, rather than the word appearing
    # in a sentence.
    look_tree = ast.parse((_HERE / "look_behind.py").read_text(), filename="look_behind.py")
    literals = [node.value for node in ast.walk(look_tree)
                if isinstance(node, ast.Constant)
                and node.value in flourish.OPERATOR_ONLY_KINDS]
    assert not literals, (
        f"look_behind.py carries {literals} as a value; the kind must come from the "
        f"operator's --flip-kind, validated against BACKWARD_KINDS")

    tree = ast.parse(mission_source, filename="mission.py")
    for name in ("play_flourish", "run_gesture"):
        constants = [argument.value for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and getattr(node.func, "id", None) == name
                     for argument in node.args if isinstance(argument, ast.Constant)
                     and isinstance(argument.value, str)]
        assert set(constants) <= set(flourish.ARRIVAL_KINDS), (name, constants)
    called = {getattr(node.func, "id", None) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    assert "play_flourish" in called and "run_gesture" in called, (
        "mission.py no longer calls these; fix this scan, do not delete it")


def test_the_flip_is_unreachable_without_a_completed_successful_look():
    """⛔ THE WHOLE SAFETY ARGUMENT, AS A SWEEP. Every single way the sequence can go wrong
    must end with the travelling kind never having been asked for."""
    ways = [
        ("not armed", _args(flip_on_arrival=False), _drive(), _view(), {}),
        ("dry run", _args(), _drive(live=False), _view(), {}),
        ("no kind", _args(flip_kind=None), _drive(), _view(), {}),
        ("no tape", _args(flip_rear_clearance_metres=None), _drive(), _view(), {}),
        ("short tape", _args(flip_rear_clearance_metres=0.1), _drive(), _view(), {}),
        ("no battery floor", _args(flip_battery_floor_pct=None), _drive(), _view(), {}),
        ("no hold", _args(flip_hold_seconds=None), _drive(), _view(), {}),
        ("outward turn failed", _args(), _drive(), _view(),
         {flourish.Flourish.LOOK: 1}),
        ("no frames", _args(), _drive(), RearView(), {}),
        ("too few frames", _args(), _drive(),
         _view(frames=1, clear_frames=1), {}),
        ("one frame dissented", _args(), _drive(),
         _view(clear_frames=CONFIRM_CLEAR_FRAMES - 1), {}),
        ("stale frames", _args(), _drive(),
         _view(stalest_frame_s=frame_stale_s() + 5.0), {}),
        ("something behind", _args(), _drive(),
         _view(nearest_m=0.3, clear_frames=0), {}),
        ("unrangeable box", _args(), _drive(), _view(unrangeable=1), {}),
        ("no camera in the drive", _args(),
         [token for token in _drive() if token != "--camera-source"], _view(), {}),
    ]
    for label, args, command, view, codes in ways:
        runner = _Runner(codes=codes)
        arrival_flip(command, args, runner, printer=lambda _line: None,
                     scanner=_scanner(view))
        assert all(kind in flourish.ARRIVAL_KINDS for kind in runner.kinds), \
            f"{label}: reached {runner.kinds}"

    # And the complement, so the sweep above cannot pass by refusing everything always.
    runner = _Runner()
    arrival_flip(_drive(), _args(), runner, printer=lambda _line: None,
                 scanner=_scanner(_view()))
    assert KIND in runner.kinds, "nothing can ever fire; the sweep proves nothing"


def test_look_behind_is_the_only_module_that_can_reach_a_travelling_kind():
    """A sweep of this directory: no other file may name one, and none may import the
    seam. If a second caller of `flip_arguments` ever appears, this fails and somebody has
    to make the same argument again."""
    callers = []
    for path in sorted(_HERE.glob("*.py")):
        if path.name in ("look_behind.py", "test_look_behind.py"):
            continue
        source = path.read_text()
        literals = [node.value for node in ast.walk(ast.parse(source, filename=path.name))
                    if isinstance(node, ast.Constant)
                    and node.value in flourish.OPERATOR_ONLY_KINDS]
        assert not literals, f"{path.name} carries {literals} as a value"
        if "flip_arguments" in source:
            callers.append(path.name)
    assert not callers, f"{callers} reach the seam directly, bypassing arrival_flip"


def test_the_module_docstring_keeps_the_argument_that_licensed_this():
    """⛔ THE REASONING IS THE SAFETY LAYER. If the paragraph about what a VOC detector
    cannot see is deleted, the next reader inherits "the robot checked, so it is clear"."""
    doc = " ".join(look_behind.__doc__.split())
    for phrase in ("does not find a wall", "VETO ADDED ON TOP", "DEFAULT OFF",
                   "LEG_TOLERANCE_RAD", "the pose channel freezes for the whole manoeuvre",
                   "check_room", "does NOT move a kind between those sets"):
        assert phrase in doc, phrase
    # And the seam's own argument, which is the one a reader will want to disagree with.
    seam = " ".join(look_behind.flip_arguments.__doc__.split())
    assert "``--operator-triggered`` IS PASSED HERE" in seam, seam
    assert "THIS IS THE SEAM" in seam, seam
    assert "both are required" in seam, seam


# ── nothing changed for a run that did not ask ──────────────────────────────
def test_a_run_without_the_flip_flags_is_byte_identical_to_the_previous_commit():
    """⛔ THE STRONGEST FORM THIS CAN TAKE OFFLINE: run the mission supervisor from
    ``git show main:mission.py`` and from the working tree over the same fake drive, and
    compare the two transcripts byte for byte.

    Not "assert no new line appears", which only tests the lines somebody remembered to
    look for. If the working tree's arrival path prints, turns, or decides anything a run
    that did not ask for a flip did not do before, the two strings differ.
    """
    try:
        old = subprocess.run(
            ["git", "show", "main:robot-stack/deep_robotics/lite3/visual_nav/mission.py"],
            cwd=_ROBOT_STACK.parents[0], capture_output=True, text=True, check=True).stdout
    except Exception as failure:            # no git, no `main`, a shallow clone
        print(f"  .. skipped: cannot read the previous mission.py ({failure!r})")
        return

    directory = Path(tempfile.mkdtemp())
    (directory / "old_mission.py").write_text(old)
    sys.path.insert(0, str(directory))
    import old_mission

    import mission
    assert old_mission.main is not mission.main

    drive = [sys.executable, "-c",
             "print('[  1.0s] policy'); print('[  2.0s] goal'); "
             "print('outcome: arrived (0.05 m from goal)')"]
    transcripts = []
    for module in (old_mission, mission):
        said: list = []
        module.print = said.append           # the module's own print, not builtins
        # Only the GESTURES go through `subprocess.run`; the drive itself is a `Popen`, so
        # it still really runs and both modules read the same transcript off its stdout.
        ran: list = []
        original = module.subprocess.run
        module.subprocess.run = _recorder(ran)
        try:
            code = module.main(["--no-voice", "--max-attempts", "1", "--", *drive])
        finally:
            module.subprocess.run = original
            del module.print
        transcripts.append((code, "\n".join(said), ran))

    assert transcripts[0][0] == transcripts[1][0], "the exit code changed"
    assert transcripts[0][1] == transcripts[1][1], (
        "the transcript of a run that did not ask for a flip changed:\n"
        f"--- before ---\n{transcripts[0][1]}\n--- after ---\n{transcripts[1][1]}")
    assert transcripts[0][2] == transcripts[1][2], (
        f"a different set of subprocesses ran: {transcripts[0][2]} vs {transcripts[1][2]}")


def test_a_tree_without_the_look_behind_still_runs_every_other_mission():
    """⚠️ `look_behind` IMPORTS `flourish`, WHICH LIVES IN A SIBLING DIRECTORY, and
    `flourish_command` already anticipates a deployment where that file is not staged --
    it checks `is_file()` and skips the gesture rather than failing. A bare import in
    `mission.py` would turn that survivable absence into a crash on every run, including
    the ones that never wanted a gesture. So the import is guarded, and the whole mission
    still works with the module missing; only the flip becomes unreachable."""
    import mission

    guard = ast.parse((_HERE / "mission.py").read_text(), filename="mission.py")
    guarded = [node for node in ast.walk(guard) if isinstance(node, ast.Try)
               and any(isinstance(child, ast.Import)
                       and any(alias.name == "look_behind" for alias in child.names)
                       for child in node.body)]
    assert guarded, "the look_behind import is no longer guarded"

    original = mission.look_behind
    mission.look_behind = None
    try:
        # It still arrives, still spins, and does not reach for the module that is gone.
        said: list = []
        mission.print = said.append
        ran: list = []
        run_original = mission.subprocess.run
        mission.subprocess.run = _recorder(ran)
        try:
            code = mission.main(["--no-voice", "--max-attempts", "1", "--",
                                 sys.executable, "-c",
                                 "print('outcome: arrived (0.05 m from goal)')"])
        finally:
            mission.subprocess.run = run_original
            del mission.print
        assert code == 0, (code, said)
        assert any("ARRIVED" in line for line in said), said
        # ...and arming a flip in that tree is refused at the parser, naming the reason.
        try:
            mission.main(["--flourish", "--flip-on-arrival", "--",
                          sys.executable, "-c", "pass"])
        except SystemExit as exit_code:
            assert exit_code.code == 2, exit_code.code
        else:
            raise AssertionError("--flip-on-arrival was accepted with no look-behind")
    finally:
        mission.look_behind = original


def test_arming_the_flip_without_the_gesture_is_refused_at_the_parser():
    """Said before the run rather than discovered after the robot has arrived: every turn
    the look-behind commands is built by `flourish_command`, which needs `--flourish`."""
    import mission

    try:
        mission.main(["--flip-on-arrival", "--", sys.executable, "-c", "pass"])
    except SystemExit as exit_code:
        assert exit_code.code == 2, exit_code.code
    else:
        raise AssertionError("--flip-on-arrival without --flourish was accepted")


def test_the_dashboard_path_can_arm_a_flip_but_only_with_every_measurement_answered():
    """SUPERSEDES `test_nothing_on_the_dashboard_path_can_arm_a_flip`, and the change is
    deliberate rather than a guard rotting away.

    That test asserted `venue_run.py` builds no `--flip-*` flag from any variable, so a
    flip could only be armed by somebody typing a command line -- and the dashboard could
    not reach it at all. On 2026-09-07 the operator asked for a dashboard checkbox,
    because "a flip is only ever armed by somebody typing it" in practice meant the flip
    had never been armed once, on either robot, in the life of the repository.

    So the property being pinned is no longer "unreachable". It is "reachable only with
    every measurement of the room answered": `MAPPO_FLIP` alone arms nothing, and the four
    settings have no defaults and never will, because the manoeuvre travels ~1.5 m into
    the one direction this platform cannot sense. `test_venue_run.py` holds the behavioural
    half -- partial answers, the off-switch spellings, and the flourish requirement. This
    holds the structural half: the switch is read from the environment, and the four
    measurements are read from it too rather than being written into this file.
    """
    source = (_HERE / "venue_run.py").read_text()
    assert "--flip-on-arrival" in source, (
        "venue_run.py no longer arms a flip at all. If that is deliberate, restore the "
        "old assertion and say why; do not leave this passing on a technicality")

    # The switch and every measurement come from the ENVIRONMENT. A number written into
    # this file would be the repo guessing about a room it cannot see.
    for name in ("MAPPO_FLIP", "MAPPO_FLIP_KIND", "MAPPO_FLIP_REAR_CLEARANCE_M",
                 "MAPPO_FLIP_BATTERY_FLOOR_PCT", "MAPPO_FLIP_HOLD_SECONDS"):
        assert name in source, f"{name} is not read from the environment"

    # And no travelling kind is named here, so the sweep above still covers this file.
    literals = [node.value for node in ast.walk(ast.parse(source, filename="venue_run.py"))
                if isinstance(node, ast.Constant)
                and node.value in flourish.OPERATOR_ONLY_KINDS]
    assert not literals, f"venue_run.py carries {literals} as a value"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_look_behind: {len(tests)}/{len(tests)} passed")
