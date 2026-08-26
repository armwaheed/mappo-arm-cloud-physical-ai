#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 camera-calibration wrapper.

Runs without ``opencv-python``: everything the shared fitter needs is imported lazily, so
these tests exercise the wrapper's own three jobs -- the inference-config assertion, the
delegated command line, and the lens-height rewrite -- on a laptop that cannot open a
camera.

The assertion test is the one carried from the Go2 corpus: a probe run at a different
detector input size from production sets a threshold that describes nothing. The lens
height tests are the ones this tree needed: nothing on the calibration path has ever asked
for a Lite3 lens height, so the file arrives carrying whatever the shared camera model left
in ``height_m`` -- the Go2's 0.32 m where that default is still in place, and ``null``
where #96 removed it. Both shapes are exercised below, because pinning only the one the
fitter used to write is how this wrapper's own prose went stale without a test noticing.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning import camera_calibration
from deep_robotics.lite3.commissioning.camera_calibration import (
    PLATFORM,
    assert_matches_production,
    build_parser,
    delegated_argv,
    describe_replaced,
    main,
    probe_inference_config,
    selected_mode,
    stamp_lens_height,
)
from deep_robotics.lite3.commissioning.measurement import Refusal, new_record, run_main

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none"]
_LENS = ["--lens-height", "0.31", "--lens-height-source", "tape, standing"]


def _quiet(callable_):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


def _args(*extra):
    return build_parser().parse_args(_CONTEXT + list(extra))


class _FakeDetectorModule:
    """Stands in for ``person_detector`` so the assertion can be tested without cv2."""

    def __init__(self, **defaults):
        import inspect

        parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        parameters += [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
                                         default=value)
                       for name, value in defaults.items()]
        signature = inspect.Signature(parameters)

        class PersonDetector:
            def __init__(self, *args, **kwargs):
                pass

        PersonDetector.__init__.__signature__ = signature
        self.PersonDetector = PersonDetector


# ── the inference-config assertion ──────────────────────────────────────────────────────
def test_the_probes_config_is_read_out_of_the_detector_not_restated_here():
    module = _FakeDetectorModule(input_size=224, confidence=0.55)
    assert probe_inference_config(module) == {"input_size": 224, "confidence": 0.55}


def test_a_detector_that_no_longer_has_a_default_refuses_rather_than_guessing():
    module = _FakeDetectorModule(confidence=0.4)
    try:
        probe_inference_config(module)
    except Refusal as refusal:
        assert "input_size" in str(refusal)
    else:
        raise AssertionError("a missing default must refuse, not be assumed")


def test_a_mismatched_input_size_refuses_the_calibration():
    """300-pixel probe against 224-pixel production: the Go2's threshold-that-described-nothing."""
    try:
        assert_matches_production({"input_size": 300, "confidence": 0.4},
                                  {"input_size": 224, "confidence": 0.4},
                                  mode="object", printer=lambda _line: None)
    except Refusal as refusal:
        assert "300" in str(refusal) and "224" in str(refusal)
        assert "--marker" in str(refusal)
    else:
        raise AssertionError("a probe/production mismatch must refuse")


def test_a_mismatched_confidence_refuses_too():
    try:
        assert_matches_production({"input_size": 300, "confidence": 0.4},
                                  {"input_size": 300, "confidence": 0.25},
                                  mode="spin-object", printer=lambda _line: None)
    except Refusal:
        return
    raise AssertionError("a confidence mismatch changes what is detected")


def test_a_matching_config_passes_and_records_what_it_compared():
    result = assert_matches_production({"input_size": 300, "confidence": 0.4},
                                       {"input_size": 300, "confidence": 0.4},
                                       mode="object", printer=lambda _line: None)
    assert result["applicable"] is True
    assert result["probe"] == {"input_size": 300, "confidence": 0.4}


def test_marker_mode_reports_the_assertion_as_not_applicable_rather_than_passing_quietly():
    lines = []
    result = assert_matches_production({"input_size": 300}, {"input_size": 224},
                                       mode="marker", printer=lines.append)
    assert result["applicable"] is False
    assert any("NOT APPLICABLE" in line for line in lines)


# ── the delegated command ───────────────────────────────────────────────────────────────
def test_marker_mode_delegates_a_marker_fit_and_asks_for_no_motion():
    argv = delegated_argv(_args("--camera-source", "0", "--marker", "1.5",
                                "--marker-size", "0.15", *_LENS))
    assert "--marker" in argv and "1.5" in argv
    assert "--live" not in argv and "--spin" not in argv


def test_spin_mode_delegates_live_and_carries_the_measured_spin_rate():
    argv = delegated_argv(_args("--camera-source", "0", "--spin", "--spin-rate", "0.9",
                                "--operator-ready", *_LENS))
    assert "--spin" in argv and "--live" in argv
    assert argv[argv.index("--spin-rate") + 1] == "0.9"
    assert "--operator-ready" in argv


def test_the_selected_mode_names_the_spin_target_because_only_one_uses_a_detector():
    assert selected_mode(_args("--camera-source", "0", "--marker", "1.5",
                               "--marker-size", "0.1", *_LENS)) == "marker"
    assert selected_mode(_args("--camera-source", "0", "--spin", "--spin-rate", "0.9",
                               *_LENS)) == "spin-marker"
    assert selected_mode(_args("--camera-source", "0", "--spin", "--spin-target",
                               "object", "--spin-rate", "0.9", *_LENS)) == "spin-object"


# ── the lens height ─────────────────────────────────────────────────────────────────────
def _written_calibration(directory, **extra):
    path = Path(directory) / "cal.json"
    data = {"platform": PLATFORM, "width": 1280, "height": 720, "focal_px": 900.0,
            "cx": 640.0, "cy": 360.0, "pitch_rad": 0.0, "height_m": 0.32,
            "hfov_deg": 81.5, "method": "marker", "samples": 20}
    data.update(extra)
    path.write_text(json.dumps(data))
    return path


def test_the_go2s_inherited_lens_height_is_replaced_and_the_old_value_is_kept():
    """The pre-#96 file shape, which a Go2 fitter still writes and is still on disk."""
    with tempfile.TemporaryDirectory() as directory:
        path = _written_calibration(directory)
        result = stamp_lens_height(path, 0.31, "tape, standing", {"robot_id": "LITE3-A"})
        data = json.loads(path.read_text())
        assert data["height_m"] == 0.31
        assert data["height_m_replaced"] == 0.32
        assert data["height_m_source"] == "tape, standing"
        assert data["commissioning_robot_id"] == "LITE3-A"
        assert result["lens_height_replaced"] == 0.32


def test_a_fitter_that_left_no_lens_height_is_reported_as_unset_rather_than_as_None():
    """The other shape, and the one this wrapper's own prose had stopped describing.

    #96 gave ``FisheyeCamera.height_m`` no default, and the shared model writes an unset
    height into the file as ``null`` rather than omitting it. So a Lite3 fit run through
    this repository's copy of the fitter now arrives here with ``height_m: null``, not
    with 0.32 -- while five sentences in ``camera_calibration.py`` still said every
    calibration in the tree carries the Go2's number, and the operator paste block
    rendered the absence as "Lens height was **None m**".

    Nothing failed, because the fixture above pins the shape the fitter used to write.
    That is the whole mechanism: a test that shares its premise with the code it tests
    stays green through exactly the change it exists to catch.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = _written_calibration(directory, height_m=None)
        result = stamp_lens_height(path, 0.31, "tape, standing", {})
        assert result["lens_height_replaced"] is None
        data = json.loads(path.read_text())
        assert data["height_m"] == 0.31
        assert data["height_m_replaced"] is None  # the absence is recorded, not dropped

    assert describe_replaced(None) == "unset (null)"
    assert describe_replaced(0.32) == "0.32 m"


def test_the_operator_is_never_told_the_lens_height_was_None_m():
    """The paste block is copied into issue #13 verbatim, so it is the artefact.

    "None m" reads as a defect in this script rather than as a fact about the
    calibration, on the one line the operator is being asked to act on.
    """
    record = new_record("LITE3-A", firmware="V1.0.8", payload="none", camera_source="0")
    result = {"focal_px": 470.0, "hfov_deg": 85.0, "width": 1280, "height": 720,
              "method": "marker", "samples": 20, "lens_height_m": 0.31,
              "lens_height_replaced": None, "lens_height_source": "tape, standing",
              "calibration_path": "/tmp/cal.json"}
    note = camera_calibration._paste(record, result, {"applicable": False})
    assert "None m" not in note
    assert "unset (null)" in note

    result["lens_height_replaced"] = 0.32
    assert "0.32 m" in camera_calibration._paste(record, result, {"applicable": False})


def test_a_calibration_that_is_not_lite3_tagged_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        path = _written_calibration(directory, platform="unitree-go2")
        try:
            stamp_lens_height(path, 0.31, "tape", {})
        except Refusal as refusal:
            assert "unitree-go2" in str(refusal)
        else:
            raise AssertionError("a Go2 calibration must not be stamped as a Lite3 one")


def test_an_unreadable_calibration_is_refused_with_its_path():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "missing.json"
        try:
            stamp_lens_height(path, 0.31, "tape", {})
        except Refusal as refusal:
            assert "missing.json" in str(refusal)
        else:
            raise AssertionError("a missing calibration must refuse")


def test_the_reported_focal_and_hfov_come_from_the_file_the_fitter_wrote():
    with tempfile.TemporaryDirectory() as directory:
        path = _written_calibration(directory, focal_px=1234.5, hfov_deg=59.4)
        result = stamp_lens_height(path, 0.31, "tape", {})
        assert result["focal_px"] == 1234.5
        assert result["hfov_deg"] == 59.4


# ── refusing to invent ──────────────────────────────────────────────────────────────────
def test_an_unmeasured_lens_height_is_refused_rather_than_left_at_the_go2s_default():
    """Isolated from the --lens-height-source guard, which would otherwise mask it.

    The source IS supplied here, so the only thing that can refuse this run is the
    missing height itself. Verified by mutation: removing that guard turns this red.
    """
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--camera-source", "0", "--marker", "1.5",
                      "--marker-size", "0.15",
                      "--lens-height-source", "tape, standing"]),
        "camera", printer=lambda _line: None))
    assert code == 2


def test_a_lens_height_with_no_method_attached_is_refused():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--camera-source", "0", "--marker", "1.5",
                                 "--marker-size", "0.15", "--lens-height", "0.31"]),
        "camera", printer=lambda _line: None))
    assert code == 2


def test_a_spin_without_a_measured_spin_rate_is_refused():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--camera-source", "0", "--spin",
                                 "--operator-ready", *_LENS]),
        "camera", printer=lambda _line: None))
    assert code == 2


def test_a_spin_without_operator_ready_is_refused():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--camera-source", "0", "--spin", "--spin-rate", "0.9",
                                 *_LENS]),
        "camera", printer=lambda _line: None))
    assert code == 2


def test_a_marker_fit_with_no_marker_size_is_refused():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--camera-source", "0", "--marker", "1.5", *_LENS]),
        "camera", printer=lambda _line: None))
    assert code == 2


def test_a_dry_run_prints_the_delegated_command_and_opens_no_camera():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main([*_CONTEXT, "--camera-source", "0", "--marker", "1.5",
                                "--marker-size", "0.15", *_LENS])
    assert code == 0
    assert "calibrate_camera.py" in buffer.getvalue()
    assert "DRY RUN" in buffer.getvalue()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"camera_calibration: {len(tests)}/{len(tests)} passed")
