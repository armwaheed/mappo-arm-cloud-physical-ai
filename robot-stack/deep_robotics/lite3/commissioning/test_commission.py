#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the commissioning driver.

Two properties carry the safety argument and both are tested by trying to break them:

* **the order.** Read-only, then tape, then camera, then the two that walk -- and the gain
  after the floor, because a gain fitted across an unknown floor is dragged down by every
  sub-floor point it swallowed.
* **provisional means provisional.** ``--emit-flags`` is the only route from this artefact
  to a live run's ``--gait-floor``, and it must refuse a record no human has signed. If
  that gate can be walked past, the ``provenance`` field is decoration.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.commission import (
    MOVING_STAGES,
    STAGES,
    build_parser,
    gait_floor_from,
    issue_status,
    live_flags,
    main,
    merge_stage_artefacts,
    stage_argv,
)
from deep_robotics.lite3.commissioning.measurement import (
    PROVISIONAL,
    REVIEWED,
    Refusal,
    merge_measurement,
    new_record,
    read_record,
    run_main,
    write_record,
)

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none"]


def _quiet(callable_):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


def _args(*extra):
    return build_parser().parse_args(list(extra))


def _complete_record(robot_id="LITE3-A"):
    record = new_record(robot_id, firmware="V1.0.8", payload="none")
    merge_measurement(record, "loaded_radius", {"radius_m": 0.484, "policy_scale": 4.84})
    merge_measurement(record, "gait_floor_forward",
                      {"conservative_m_s": 0.30, "lowest_walking_m_s": 0.20,
                       "bracketed": True})
    merge_measurement(record, "actuator_gain", {"pose_fit": {"gain": 0.62}})
    merge_measurement(record, "camera_calibration",
                      {"calibration_path": "/tmp/lite3_front_camera_LITE3-A.json",
                       "focal_px": 900.0, "hfov_deg": 81.5})
    merge_measurement(record, "motor_temperatures", {"channel": "absent"})
    return record


# ── the order ───────────────────────────────────────────────────────────────────────────
def test_the_read_only_stages_come_before_anything_that_moves():
    for stage in MOVING_STAGES:
        assert STAGES.index(stage) >= len(STAGES) - len(MOVING_STAGES), stage
    assert STAGES.index("gait") < STAGES.index("gain")


def test_only_the_two_walking_stages_are_marked_as_moving():
    assert set(MOVING_STAGES) == {"gait", "gain"}


def test_the_gain_stage_starts_from_the_floor_the_gait_stage_measured():
    record = new_record("LITE3-A")
    merge_measurement(record, "gait_floor_forward", {"conservative_m_s": 0.27})
    assert abs(gait_floor_from(record) - 0.27) < 1e-9


def test_an_unbracketed_ladder_stops_the_gain_stage_instead_of_guessing_a_floor():
    record = new_record("LITE3-A")
    merge_measurement(record, "gait_floor_forward",
                      {"conservative_m_s": None, "note": "every rung walked"})
    try:
        gait_floor_from(record)
    except Refusal as refusal:
        assert "must not be fitted across an unknown floor" in str(refusal)
    else:
        raise AssertionError("a gain from an unknown floor must refuse")


def test_a_missing_gait_result_stops_the_gain_stage():
    try:
        gait_floor_from(new_record("LITE3-A"))
    except Refusal:
        return
    raise AssertionError("no floor means no gain")


# ── stage arguments ─────────────────────────────────────────────────────────────────────
def test_every_stage_carries_the_issue_13_context_forward():
    args = _args(*_CONTEXT, "--front", "0.4", "--back", "0.4", "--left", "0.2",
                 "--right", "0.2", "--out", "r.json")
    for stage in STAGES:
        argv = stage_argv(stage, args, gait_floor=0.3)
        for flag in ("--robot-id", "--firmware", "--payload"):
            assert flag in argv, (stage, flag)


def test_the_walking_stages_only_receive_live_when_the_driver_was_given_it():
    args = _args(*_CONTEXT, "--out", "r.json")
    assert "--live" not in stage_argv("gait", args)
    args = _args(*_CONTEXT, "--out", "r.json", "--live", "--operator-ready")
    argv = stage_argv("gait", args)
    assert "--live" in argv and "--operator-ready" in argv


def test_the_gain_stage_is_handed_the_measured_floor_verbatim():
    args = _args(*_CONTEXT, "--out", "r.json", "--envelope-vx", "0.4")
    argv = stage_argv("gain", args, gait_floor=0.275)
    assert argv[argv.index("--gait-floor") + 1] == "0.275"


def test_each_stage_writes_its_own_artefact_beside_the_combined_one():
    args = _args(*_CONTEXT, "--out", "/tmp/lite3-commissioning-LITE3-A.json")
    paths = {stage: stage_argv(stage, args, 0.3)[
        stage_argv(stage, args, 0.3).index("--artefact") + 1] for stage in STAGES}
    assert len(set(paths.values())) == len(STAGES)
    assert all(path.endswith(f"-{stage}.json") for stage, path in paths.items())


# ── merging ─────────────────────────────────────────────────────────────────────────────
def test_merging_folds_each_stages_measurements_into_one_record():
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "a.json"
        second = Path(directory) / "b.json"
        write_record(first, merge_measurement(new_record("LITE3-A"), "loaded_radius",
                                              {"radius_m": 0.5}))
        write_record(second, merge_measurement(new_record("LITE3-A"), "actuator_gain",
                                               {"pose_fit": {"gain": 0.6}}))
        record = merge_stage_artefacts(new_record("LITE3-A"), [first, second],
                                       printer=lambda _line: None)
        assert set(record.measurements) == {"loaded_radius", "actuator_gain"}


def test_a_stage_artefact_from_the_other_robot_is_refused():
    """Issue #13's whole premise: nothing transfers between the two Ventures."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "b.json"
        write_record(path, merge_measurement(new_record("LITE3-B"), "loaded_radius",
                                             {"radius_m": 0.5}))
        try:
            merge_stage_artefacts(new_record("LITE3-A"), [path],
                                  printer=lambda _line: None)
        except Refusal as refusal:
            assert "LITE3-B" in str(refusal) and "LITE3-A" in str(refusal)
        else:
            raise AssertionError("a number from the other robot must not be merged in")


def test_a_stage_that_did_not_run_is_skipped_rather_than_failing_the_merge():
    record = merge_stage_artefacts(new_record("LITE3-A"),
                                   [Path("/nonexistent/never-written.json")],
                                   printer=lambda _line: None)
    assert record.measurements == {}


# ── the provisional gate ────────────────────────────────────────────────────────────────
def test_emit_flags_refuses_a_provisional_record():
    """If this can be walked past, the provenance field is decoration."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "r.json"
        write_record(path, _complete_record())
        assert read_record(path).provenance == PROVISIONAL
        code = _quiet(lambda: run_main(
            lambda: main(["--record", str(path), "--emit-flags"]),
            "commission", printer=lambda _line: None))
        assert code == 2


def test_review_signs_the_record_and_only_then_do_flags_come_out():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "r.json"
        write_record(path, _complete_record())
        assert _quiet(lambda: main(["--record", str(path), "--review", "A Reviewer"])) == 0
        signed = read_record(path)
        assert signed.provenance == REVIEWED
        assert signed.reviewed_by == "A Reviewer"
        assert signed.reviewed_utc

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert main(["--record", str(path), "--emit-flags"]) == 0
        emitted = buffer.getvalue()
        for flag in ("--gait-floor", "--actuator-gain", "--robot-radius",
                     "--policy-scale", "--calibration"):
            assert flag in emitted, flag
        assert "0.300" in emitted and "0.620" in emitted and "0.484" in emitted


def test_an_unsigned_review_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "r.json"
        write_record(path, _complete_record())
        code = _quiet(lambda: run_main(
            lambda: main(["--record", str(path), "--review", "   "]),
            "commission", printer=lambda _line: None))
        assert code == 2


def test_flags_are_refused_for_a_record_with_a_hole_in_it():
    record = _complete_record()
    del record.measurements["actuator_gain"]
    record.provenance = REVIEWED
    record.reviewed_by = "A Reviewer"
    try:
        live_flags(record)
    except Refusal as refusal:
        assert "actuator_gain" in str(refusal)
    else:
        raise AssertionError("a partial record must not produce a partial command line")


def test_flags_are_refused_when_the_ladder_never_bracketed_a_floor():
    record = _complete_record()
    record.measurements["gait_floor_forward"]["conservative_m_s"] = None
    try:
        live_flags(record)
    except Refusal as refusal:
        assert "never bracketed" in str(refusal)
    else:
        raise AssertionError("no floor, no flags")


# ── issue #13 accounting ────────────────────────────────────────────────────────────────
def test_status_reports_a_complete_record_as_closing_the_measurement_checkboxes():
    rows = issue_status(_complete_record())
    states = {label: state for label, state, _payload in rows}
    assert all(state != "not measured" for state in states.values()), states


def test_an_absent_temperature_channel_is_reported_as_evidence_not_as_closed():
    rows = issue_status(_complete_record())
    temperature = [state for label, state, _ in rows if "motor temperatures" in label]
    assert temperature and "vendor question" in temperature[0]


def test_a_present_temperature_channel_closes_the_checkbox():
    record = _complete_record()
    record.measurements["motor_temperatures"]["channel"] = "present"
    rows = issue_status(record)
    temperature = [state for label, state, _ in rows if "motor temperatures" in label]
    assert temperature == ["closed"]


def test_an_empty_record_closes_nothing():
    rows = issue_status(new_record("LITE3-A"))
    assert all(state == "not measured" for _label, state, _payload in rows)


# ── merging standalone artefacts ────────────────────────────────────────────────────────
def test_merge_folds_standalone_probe_artefacts_into_one_record():
    """The runbook has an operator run the probes one at a time, so this path must work."""
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "lite3-loaded-radius-LITE3-A.json"
        second = Path(directory) / "lite3-camera-LITE3-A.json"
        out = Path(directory) / "combined.json"
        write_record(first, merge_measurement(new_record("LITE3-A"), "loaded_radius",
                                              {"radius_m": 0.5, "policy_scale": 5.0}))
        write_record(second, merge_measurement(new_record("LITE3-A"),
                                               "camera_calibration",
                                               {"focal_px": 900.0}))
        assert _quiet(lambda: main(["--robot-id", "LITE3-A", "--out", str(out),
                                    "--merge", str(first), str(second)])) == 0
        record = read_record(out)
        assert set(record.measurements) == {"loaded_radius", "camera_calibration"}
        assert record.provenance == PROVISIONAL


def test_merge_refuses_an_artefact_that_does_not_exist_rather_than_skipping_it():
    """Silently skipping one would produce a record with a hole and no sign of it."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "combined.json"
        code = _quiet(lambda: run_main(
            lambda: main(["--robot-id", "LITE3-A", "--out", str(out),
                          "--merge", str(Path(directory) / "nope.json")]),
            "commission", printer=lambda _line: None))
        assert code == 2
        assert not out.exists()


def test_merge_without_a_robot_id_is_refused():
    code = _quiet(lambda: run_main(lambda: main(["--merge", "a.json"]),
                                   "commission", printer=lambda _line: None))
    assert code == 2


# ── driver preconditions ────────────────────────────────────────────────────────────────
def test_running_stages_without_the_issue_13_context_is_refused():
    code = _quiet(lambda: run_main(lambda: main(["--stage", "radius"]),
                                   "commission", printer=lambda _line: None))
    assert code == 2


def test_the_record_management_modes_need_a_record():
    code = _quiet(lambda: run_main(lambda: main(["--emit-flags"]),
                                   "commission", printer=lambda _line: None))
    assert code == 2


def test_a_dry_driver_run_stops_at_the_first_stage_refusal_rather_than_writing_a_hole():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "r.json"
        code = _quiet(lambda: run_main(lambda: main([*_CONTEXT,
            "--stage", "radius", "--out", str(out)]),
            "commission", printer=lambda _line: None))
        assert code == 2                      # no extents supplied
        assert not out.exists()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"commission: {len(tests)}/{len(tests)} passed")
