#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the one place the detector's preprocessing is written down.

THE POINT OF THIS FILE IS THAT THE GUARD FIRES. Issue #129 is not a wrong number, it is a
robot that runs three different detectors depending on which script starts it, plus a
checkpoint sweep that scored through a fourth configuration no script runs at all. So the
tests that matter here are the ones that make ``resolve()`` refuse, the ones that bind each
declared deployment to the artefact that defines it, and the ones that would go red if the
literals came back.

Every structural check is paired with a NEGATIVE control that feeds it text it must reject,
because a grep that finds nothing is indistinguishable from a grep that cannot find
anything — this repository has already shipped a latch check that proved the arm was held
by asserting its joints had stopped moving, which an unpowered arm passes.

``cv2`` is required, like the rest of this directory: ``person_detector`` imports it, and
the cross-check against ``person_detector``'s own constants is half of what makes this a
single source rather than a second copy.

Run: ``python3 test_inference_profile.py``
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_profile
import person_detector
from inference_profile import (
    DEPLOYMENTS,
    GO2_NAVIGATOR_DEFAULT,
    GO2_PEER_SUPERVISED,
    GO2_RUN_SMOKE,
    MOBILENET_SSD_TRAINED,
    PROFILES,
    PreprocessingMismatch,
    assert_matches_person_detector,
    assert_prototxt_floor,
    matching_profile,
    resolve,
    resolve_many,
    stamp,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LAUNCHER = REPO / "deploy" / "run-peer-supervised.sh"
RUN_PROFILE = REPO / "dashboard" / "run-profile.example.json"
DETECTOR = REPO / "detector"

#: What ``deploy/run-peer-supervised.sh`` passed as literals BEFORE #129, recovered from
#: the file as it stood at 4d79b45 and pinned here. The launcher now derives its flags from
#: :data:`inference_profile.GO2_PEER_SUPERVISED` instead, and this is the evidence that the
#: derivation is the same argv — a tooling change, not a robot behaviour change.
#: **If this test fails, the robot's preprocessing moved and needs a live run.**
LAUNCHER_ARGV_BEFORE_129 = (
    "--input-size", "224",
    "--confidence", "0.25",
    "--classes", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)

#: A scorer declaring its own square. ``INPUT_SIZE = 300`` is the exact form all five
#: ``detector/`` modules carried; the second catches ``(300, 300)`` passed straight into
#: ``blobFromImage``, which is what ``peer_recall.py`` had instead of a constant.
_OWN_SQUARE = (
    re.compile(r"^\s*INPUT_SIZE\s*=\s*\d+\s*$", re.MULTILINE),
    re.compile(r"blobFromImage\([^)]*\(\s*\d+\s*,\s*\d+\s*\)", re.DOTALL),
)


def _namespace(**kwargs) -> argparse.Namespace:
    """A parsed command line, the way a scorer's ``main`` would hand one to ``resolve``."""
    defaults = {"preprocessing": GO2_PEER_SUPERVISED.name, "input_size": None,
                "allow_preprocessing_mismatch": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _declares_own_square(text: str) -> bool:
    return any(pattern.search(text) for pattern in _OWN_SQUARE)


def _shell_code(text: str) -> str:
    """``text`` with whole-line comments removed.

    The check below is about what the launcher PASSES, not what it explains: the header
    added by #129 quotes the old ``--input-size 224`` line in prose, and a scan that could
    not tell those apart would either fail on the explanation or force the explanation out
    of the file.
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


# ── there is no single production, and the registry has to say so ───────────────

def test_the_robot_runs_more_than_one_configuration():
    """The finding, as an assertion. If this ever collapses to one, the reconciliation
    happened and most of this module can go — but it must be noticed, not assumed."""
    assert len(DEPLOYMENTS) >= 2, DEPLOYMENTS
    assert {d.input_size for d in DEPLOYMENTS} == {224, 300}
    assert {d.confidence for d in DEPLOYMENTS} == {0.25, 0.4, 0.45}
    # And they disagree about which labels reach the planner, which decides the hold
    # denominator as surely as the threshold does.
    assert {len(d.classes) for d in DEPLOYMENTS} == {1, 20}


def test_the_sweeps_configuration_is_run_by_nothing():
    """300 px at 0.25 — the square from the trainer, the floor from the peer launcher.
    Both real numbers about the weights; their pair is not a run."""
    assert MOBILENET_SSD_TRAINED.input_size == 300
    assert MOBILENET_SSD_TRAINED.confidence == 0.25
    assert not MOBILENET_SSD_TRAINED.is_deployed
    assert MOBILENET_SSD_TRAINED.deployments == ()


def test_every_deployment_names_the_file_that_decides_it():
    for deployment in DEPLOYMENTS:
        assert deployment.source, deployment
        assert deployment.is_deployed, deployment
        assert deployment.source in deployment.deployments, deployment


# ── the guard ───────────────────────────────────────────────────────────────────

def test_there_is_no_default_preprocessing():
    """A default would have to be one of three real configurations, and picking one
    silently is the accident. argparse must require the flag."""
    parser = inference_profile.add_arguments(argparse.ArgumentParser())
    action = next(a for a in parser._actions if a.dest == "preprocessing")
    assert action.required is True
    assert action.default is None, action.default


def test_a_deployment_is_allowed_and_costs_no_excuse():
    for deployment in DEPLOYMENTS:
        profile, reason = resolve(_namespace(preprocessing=deployment.name))
        assert profile is deployment
        assert reason is None
        record = stamp(profile, reason)
        assert record["deployed"] is True, record
        assert "differences_from_deployments" not in record, record


def test_scoring_at_the_configuration_no_launcher_runs_is_refused():
    """The exact mistake of #129."""
    try:
        resolve(_namespace(preprocessing=MOBILENET_SSD_TRAINED.name))
    except PreprocessingMismatch as refusal:
        message = str(refusal)
    else:
        raise AssertionError("300 px at 0.25 was allowed; the guard did not fire")
    # The refusal must name what IS run, not merely what is not: a reader who has just been
    # stopped needs the list of real configurations, in the one message they will read.
    assert "NO LAUNCHER RUNS THAT" in message, message
    for deployment in DEPLOYMENTS:
        assert deployment.name in message, message
        assert deployment.source in message, message
    assert "#129" in message, message
    assert "--allow-preprocessing-mismatch" in message, message


def test_an_input_size_override_is_refused_without_a_reason():
    """The free integer is still reachable — a size sweep is a real experiment — but it is
    not reachable quietly."""
    try:
        resolve(_namespace(preprocessing=GO2_PEER_SUPERVISED.name, input_size=416))
    except PreprocessingMismatch as refusal:
        assert "416 px" in str(refusal), refusal
    else:
        raise AssertionError("--input-size 416 was allowed with no acknowledgement")


def test_an_override_is_judged_on_the_whole_configuration_not_the_flag():
    """224 px at 0.45 is not a deployment; 300 px at 0.45 is. What is tested is the
    configuration that results, not which flag produced it."""
    profile, _ = resolve(_namespace(preprocessing=GO2_RUN_SMOKE.name, input_size=300))
    assert profile.is_deployed, profile
    try:
        resolve(_namespace(preprocessing=GO2_RUN_SMOKE.name, input_size=224))
    except PreprocessingMismatch as refusal:
        assert "224 px at confidence 0.45" in str(refusal), refusal
    else:
        raise AssertionError("224 px at 0.45 is run by nothing and was allowed")


def test_an_acknowledged_mismatch_carries_its_reason_into_the_output():
    profile, reason = resolve(_namespace(preprocessing=MOBILENET_SSD_TRAINED.name,
                                         allow_preprocessing_mismatch="the sweep's row"))
    assert profile is MOBILENET_SSD_TRAINED
    record = stamp(profile, reason)
    assert record["deployed"] is False, record
    assert record["mismatch_reason"] == "the sweep's row", record
    # It must say what it differs FROM, per deployment, or the reader has to hold three
    # configurations in their head to interpret one row.
    differences = record["differences_from_deployments"]
    assert set(differences) == {d.name for d in DEPLOYMENTS}, differences
    assert differences[GO2_PEER_SUPERVISED.name] == {
        "input_size": {"this_run": 300, "deployed": 224}}, differences
    assert differences[GO2_RUN_SMOKE.name] == {
        "confidence": {"this_run": 0.25, "deployed": 0.45},
        "classes": {"this_run": list(MOBILENET_SSD_TRAINED.classes),
                    "deployed": list(GO2_RUN_SMOKE.classes)}}, differences


def test_several_profiles_resolve_together_under_one_reason():
    args = _namespace(preprocessing=[GO2_RUN_SMOKE.name, MOBILENET_SSD_TRAINED.name],
                      allow_preprocessing_mismatch="lining up the published tables")
    profiles, reason = resolve_many(args)
    assert [p.name for p in profiles] == [GO2_RUN_SMOKE.name, MOBILENET_SSD_TRAINED.name]
    assert reason == "lining up the published tables"
    assert profiles[0].is_deployed
    assert not profiles[1].is_deployed


def test_several_profiles_still_refuse_when_no_reason_is_given():
    try:
        resolve_many(_namespace(preprocessing=[GO2_RUN_SMOKE.name,
                                               MOBILENET_SSD_TRAINED.name]))
    except PreprocessingMismatch as refusal:
        assert "NO LAUNCHER RUNS THAT" in str(refusal), refusal
    else:
        raise AssertionError("a non-deployed profile passed inside a list")


def test_identity_is_by_value_not_by_name():
    """A profile carrying a deployment's name while computing something else is not that
    deployment, and one computing a deployment's configuration under another name is."""
    impostor = inference_profile.InferenceProfile(
        name=GO2_PEER_SUPERVISED.name, role="deployment", input_size=256,
        confidence=GO2_PEER_SUPERVISED.confidence, scale=GO2_PEER_SUPERVISED.scale,
        mean=GO2_PEER_SUPERVISED.mean, swap_rb=False,
        classes=GO2_PEER_SUPERVISED.classes,
        source="nowhere", why="a name is not a configuration")
    assert not impostor.is_deployed
    twin = inference_profile.InferenceProfile(
        name="something-else", role="ad-hoc", input_size=GO2_RUN_SMOKE.input_size,
        confidence=GO2_RUN_SMOKE.confidence, scale=GO2_RUN_SMOKE.scale,
        mean=GO2_RUN_SMOKE.mean, swap_rb=False, classes=GO2_RUN_SMOKE.classes,
        source="nowhere", why="same configuration, different name")
    assert twin.is_deployed
    assert twin.deployments == (GO2_RUN_SMOKE.source,)


def test_matching_profile_names_a_live_configuration_or_admits_it_cannot():
    """What the telemetry header uses. ``None`` is a real answer: an operator can type a
    square nothing here has measured, and the record must say so rather than round it."""
    assert matching_profile(224, 0.25, GO2_PEER_SUPERVISED.classes) is GO2_PEER_SUPERVISED
    assert matching_profile(300, 0.45, ("person",)) is GO2_RUN_SMOKE
    assert matching_profile(300, 0.4, ("person",)) is GO2_NAVIGATOR_DEFAULT
    assert matching_profile(256, 0.3, ("person",)) is None
    assert matching_profile(224, 0.25, ("person",), mean=0.0) is None
    # The class list is part of the configuration, not decoration: the peer launcher's
    # square and floor with the navigator's class list is a configuration nothing runs.
    assert matching_profile(224, 0.25, ("person",)) is None


# ── the confidence floor, which lives in the prototxt and not in the caller ──────

def _prototxt(floor: str) -> str:
    return ("layer {\n  name: \"detection_out\"\n  type: \"DetectionOutput\"\n"
            "  detection_output_param {\n    num_classes: 21\n"
            f"    confidence_threshold: {floor}\n  }}\n}}\n")


def test_a_profile_below_the_layer_floor_is_refused():
    """``DetectionOutput`` deletes the rows inside ``forward()``; a run asking for 0.10
    against a 0.25 prototxt measures 0.25 and would print it as 0.10."""
    permissive = inference_profile.InferenceProfile(
        name="low", role="ad-hoc", input_size=224, confidence=0.10,
        scale=GO2_PEER_SUPERVISED.scale, mean=GO2_PEER_SUPERVISED.mean, swap_rb=False,
        classes=(), source="nowhere", why="asks for boxes the layer has already deleted")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m.prototxt"
        path.write_text(_prototxt("0.25"))
        try:
            assert_prototxt_floor(path, permissive)
        except PreprocessingMismatch as refusal:
            assert "already discards everything below 0.25" in str(refusal), refusal
            assert "prototxt_with_floor" in str(refusal), refusal
        else:
            raise AssertionError("a request below the layer's own floor was accepted")


def test_a_profile_above_the_layer_floor_is_allowed_because_it_is_honest():
    """0.45 against a 0.25 prototxt really does measure 0.45 — Python discards the extra
    rows itself. Refusing it would forbid scoring ``run-smoke.sh``, which is the deployment
    the whole 89-run corpus was recorded through."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m.prototxt"
        path.write_text(_prototxt("0.25"))
        assert assert_prototxt_floor(path, GO2_RUN_SMOKE) == 0.25
        assert assert_prototxt_floor(path, GO2_PEER_SUPERVISED) == 0.25


def test_two_floors_in_one_prototxt_are_refused():
    """With two, one of them is silently authoritative and no caller can tell which."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m.prototxt"
        path.write_text(_prototxt("0.25") + _prototxt("0.05"))
        try:
            assert_prototxt_floor(path, GO2_PEER_SUPERVISED)
        except PreprocessingMismatch as refusal:
            assert "found 2" in str(refusal), refusal
        else:
            raise AssertionError("two confidence_threshold lines were accepted")


# ── the declarations that cannot be re-derived, checked against their sources ────

def test_the_profile_matches_person_detectors_own_constants():
    """``scale``/``mean``/the class list are declared twice by necessity — this module must
    import without ``cv2`` and ``person_detector`` cannot. This is what makes the second
    copy a check rather than a copy."""
    checked = assert_matches_person_detector(person_detector)
    assert checked["_SSD_SCALE"] == 1.0 / 127.5, checked
    assert checked["_SSD_MEAN"] == 127.5, checked
    assert checked["VOC_CLASSES"] == 21, checked


def test_that_cross_check_can_fail():
    """Negative control. A module whose constants have drifted must be rejected, or the
    check passes because it looks at nothing."""
    class Drifted:
        _SSD_SCALE = 1.0 / 128.0
        _SSD_MEAN = 127.5
        VOC_CLASSES = person_detector.VOC_CLASSES
        _CONFIDENCE_FLOOR_RE = person_detector._CONFIDENCE_FLOOR_RE
    try:
        assert_matches_person_detector(Drifted)
    except PreprocessingMismatch as refusal:
        assert "_SSD_SCALE" in str(refusal), refusal
    else:
        raise AssertionError("a drifted _SSD_SCALE passed the cross-check")


def test_the_navigator_default_profile_matches_the_navigator():
    """``go2-navigator-default`` claims to be what a bare ``visual_nav.py`` computes. Read
    that out of the parser rather than trusting the declaration — this is the one
    deployment whose source file IS in the repository and can be interrogated."""
    import visual_nav
    defaults = visual_nav.build_parser().parse_args([])
    assert GO2_NAVIGATOR_DEFAULT.input_size == defaults.input_size, defaults.input_size
    assert GO2_NAVIGATOR_DEFAULT.confidence == defaults.confidence, defaults.confidence


def test_the_run_smoke_profile_matches_the_run_profile_this_repo_holds():
    """``go2-run-smoke`` is declared by hand because its launcher is on the robot. The one
    copy of that invocation this repository holds is the dashboard's example run profile,
    which says in its own header that it was copied from ``run-smoke.sh``. Bind them, so
    the hand-written declaration cannot drift from the only artefact that can check it."""
    extra = list(json.loads(RUN_PROFILE.read_text())["extra_args"])
    assert "--input-size" not in extra, (
        "run-profile.example.json now passes --input-size; go2-run-smoke's 300 came from "
        "visual_nav.py's default precisely because it did not")
    confidence = float(extra[extra.index("--confidence") + 1])
    assert GO2_RUN_SMOKE.confidence == confidence, (GO2_RUN_SMOKE.confidence, confidence)
    assert GO2_RUN_SMOKE.input_size == GO2_NAVIGATOR_DEFAULT.input_size, (
        "with no --input-size, run-smoke gets whatever visual_nav.py defaults to")


def test_swap_rb_matches_the_call_site():
    """BGR, because ``blobFromImage``'s ``swapRB`` defaults to False and neither
    ``person_detector`` nor any scorer passes it. Asserted against the default rather than
    against the docstring that claims it."""
    assert all(d.swap_rb is False for d in DEPLOYMENTS)
    source = inspect.getsource(person_detector.PersonDetector.detect_tiered)
    assert "blobFromImage" in source, "the call site moved; re-point this assertion"
    assert "swapRB" not in source, ("person_detector now passes swapRB explicitly; the "
                                    "declared swap_rb must be checked against it")


# ── the literals that must not come back ────────────────────────────────────────

def test_the_launcher_derives_its_flags_and_holds_no_literal():
    code = _shell_code(LAUNCHER.read_text())
    assert "inference_profile.py" in code, "the launcher no longer reads the profile"
    for flag in ("--input-size", "--confidence"):
        literal = re.search(re.escape(flag) + r"\s+[\d.]+", code)
        assert literal is None, (
            f"{LAUNCHER.name} has gone back to a literal {literal.group(0)!r}; that is "
            f"the copy #129 is about")


def test_that_launcher_scan_can_fail():
    """Negative control, both directions: the scan must catch a re-introduced literal and
    must not be fooled by the same text inside a comment."""
    assert re.search(r"--input-size\s+[\d.]+",
                     _shell_code("COMMON=(\n  --input-size 224\n)\n"))
    assert not re.search(r"--input-size\s+[\d.]+",
                         _shell_code("# it used to say --input-size 224\nCOMMON=()\n"))


def test_the_launcher_gets_exactly_the_argv_it_used_to_hardcode():
    """The evidence that this is a tooling change and not a robot behaviour change.

    Order is not compared — argparse does not care, and the derived form emits
    ``--confidence`` before ``--classes`` — but every flag and value is."""
    emitted = subprocess.run(
        [sys.executable, str(HERE / "inference_profile.py"), "--argv",
         GO2_PEER_SUPERVISED.name],
        check=True, stdout=subprocess.PIPE, text=True).stdout.split("\n")
    emitted = [token for token in emitted if token]
    assert sorted(emitted) == sorted(LAUNCHER_ARGV_BEFORE_129), (
        "the launch argv changed:\n"
        f"  now:    {emitted}\n  before: {list(LAUNCHER_ARGV_BEFORE_129)}")


def test_the_navigator_parses_the_derived_argv_into_the_intended_run():
    """The end of the chain, through the REAL parser rather than a comparison of strings.
    ``--classes`` takes ``nargs="+"``, so the emitted list has to terminate on the next
    flag the launcher appends; that it does is a property of argparse this asserts."""
    import visual_nav
    parser = visual_nav.build_parser()
    argv = [*GO2_PEER_SUPERVISED.argv(), "--obstacle-height", "0.514",
            "--obstacle-width", "0.31"]
    args, unparsed = parser.parse_known_args(argv)
    assert not unparsed, unparsed
    assert args.input_size == 224, args.input_size
    assert args.confidence == 0.25, args.confidence
    assert args.classes == list(GO2_PEER_SUPERVISED.classes), args.classes
    assert args.obstacle_height == 0.514, args.obstacle_height


def test_the_telemetry_header_names_the_configuration_a_run_computed():
    """The half of #129 that matters most, and the one no code could work around.

    89 recorded runs had to have their preprocessing established by READING THE LAUNCHERS,
    because the header named the confidence and the class list and nothing about the
    square. Every deployment must now be identifiable from a run's own record, and a
    configuration that is not a deployment must be reported as such rather than rounded to
    the nearest name.
    """
    import visual_nav

    for deployment in DEPLOYMENTS:
        argv = ["--input-size", str(deployment.input_size),
                "--confidence", str(deployment.confidence),
                "--classes", *deployment.classes]
        record = visual_nav._preprocessing_record(
            visual_nav.build_parser().parse_args(argv))
        assert record["profile"] == deployment.name, (deployment.name, record)
        assert record["deployments"] == list(deployment.deployments), record
        assert record["input_size"] == deployment.input_size
        assert record["confidence"] == deployment.confidence

    improvised = visual_nav._preprocessing_record(
        visual_nav.build_parser().parse_args(["--input-size", "256"]))
    # 256 px is nothing's square; so is the peer launcher's square with the navigator's
    # class list, which is the case a size-only check would miss.
    crossed = visual_nav._preprocessing_record(
        visual_nav.build_parser().parse_args(["--input-size", "224",
                                              "--confidence", "0.25"]))
    assert crossed["profile"] is None, crossed
    assert improvised["profile"] is None, improvised
    assert improvised["deployments"] == [], improvised
    assert improvised["input_size"] == 256, improvised
    # And the goal detector is reported apart from the obstacle detector, because it runs
    # its own square on a half crop and a reader given one number cannot tell them apart.
    assert improvised["goal"]["input_size"] != improvised["input_size"], improvised


def test_no_detector_scorer_declares_its_own_square():
    offenders = sorted(p.name for p in DETECTOR.glob("*.py")
                       if _declares_own_square(p.read_text()))
    assert not offenders, (
        f"{offenders} declare a network input size of their own. Every one of them is a "
        f"copy that can drift from what a launcher runs, which is #129. Take it from "
        f"inference_profile instead.")


def test_that_scan_can_fail():
    """Negative control for the scan above, on both forms it looks for."""
    assert _declares_own_square("INPUT_SIZE = 300\n")
    assert _declares_own_square("blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 1)")
    assert not _declares_own_square(
        "INPUT_SIZE = inference_profile.MOBILENET_SSD_TRAINED.input_size\n")
    assert not _declares_own_square(
        "cv2.dnn.blobFromImage(image, _SCALE, (size, size), _MEAN)")


def test_every_registered_profile_is_reachable_by_name_from_the_command_line():
    """``--argv`` is what the launcher runs. A profile that cannot be emitted is a profile
    the robot cannot be launched with."""
    for name, profile in PROFILES.items():
        emitted = subprocess.run(
            [sys.executable, str(HERE / "inference_profile.py"), "--argv", name],
            check=True, stdout=subprocess.PIPE, text=True).stdout.split()
        assert emitted[:2] == ["--input-size", str(profile.input_size)], (name, emitted)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"inference_profile: {len(tests)}/{len(tests)} passed")
