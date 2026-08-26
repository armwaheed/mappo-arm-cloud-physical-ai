#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the on-a-robot virtualenv refusal.

Every test injects its environment and its filesystem, so the whole matrix -- robot vs
laptop vs CI, venv vs system Python, live vs not -- runs on any machine and none of it
depends on where the suite happens to be executed. That is the point: the guard's own
tests must not be the thing that decides whether the guard works.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from venv_guard import (
    CI_ENV_VARS,
    DEVICE_TREE_MODEL_FILE,
    JETSON_RELEASE_FILE,
    ROBOT_HOST_ENV,
    Decision,
    describe,
    evaluate,
    find_virtualenvs,
    in_virtualenv,
    refusal_message,
    require_virtualenv,
    robot_host_evidence,
)

#: Measured on the lab Go2 2026-08-26 -- the first line of its /etc/nv_tegra_release.
GO2_TEGRA_RELEASE = ("# R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, "
                     "EABI: aarch64, DATE: Sun Mar 19 15:19:21 UTC 2023\n")
#: Measured on the same robot -- device-tree model, NUL-terminated as the kernel exports it.
GO2_DEVICE_TREE_MODEL = "NVIDIA Orin NX Developer Kit\x00"


def _fs(**files):
    """A read_file stub. Absent paths answer None, which is what the real one does."""
    def read_file(path):
        return files.get(path)
    return read_file


_NOTHING = _fs()
_GO2 = _fs(**{JETSON_RELEASE_FILE: GO2_TEGRA_RELEASE})


def _go2_live(**overrides):
    """A live run on the Go2 from the SYSTEM Python -- the case the guard exists for."""
    kwargs = {"component": "test", "reaching_hardware": True, "env": {},
              "read_file": _GO2, "prefix": "/usr", "base_prefix": "/usr",
              "executable": "/usr/bin/python3", "venvs": []}
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- venv detection

def test_in_virtualenv_reads_the_prefixes_and_not_the_environment():
    """VIRTUAL_ENV is not consulted, in either direction.

    A wrapper, a systemd unit, sudo without -E or a subprocess with a scrubbed env= all
    reach the interpreter without VIRTUAL_ENV set while genuinely being inside the venv;
    an exported-and-stale VIRTUAL_ENV is the opposite error. Both are ordinary on a robot.
    """
    assert in_virtualenv(prefix="/home/unitree/envs/x", base_prefix="/usr") is True
    assert in_virtualenv(prefix="/usr", base_prefix="/usr") is False


def test_in_virtualenv_defaults_to_this_interpreter():
    assert in_virtualenv() is (sys.prefix != sys.base_prefix)


# ----------------------------------------------------------------------- host evidence

def test_the_go2s_measured_jetson_release_file_is_evidence():
    evidence = robot_host_evidence(env={}, read_file=_GO2)
    assert evidence is not None
    assert JETSON_RELEASE_FILE in evidence


def test_the_go2s_measured_device_tree_model_is_evidence():
    evidence = robot_host_evidence(
        env={}, read_file=_fs(**{DEVICE_TREE_MODEL_FILE: GO2_DEVICE_TREE_MODEL}))
    assert evidence is not None
    assert "Orin NX" in evidence
    assert "\x00" not in evidence, "the NUL the kernel exports must not reach the operator"


def test_a_developer_laptop_is_not_a_robot():
    assert robot_host_evidence(env={}, read_file=_NOTHING) is None


def test_a_device_tree_that_is_not_a_robot_carrier_is_not_evidence():
    read_file = _fs(**{DEVICE_TREE_MODEL_FILE: "Raspberry Pi 4 Model B Rev 1.4\x00"})
    assert robot_host_evidence(env={}, read_file=read_file) is None


def test_ci_is_never_a_robot_even_on_a_jetson_runner():
    """A self-hosted aarch64 runner is a real thing and it must never refuse a build."""
    for name in CI_ENV_VARS:
        assert robot_host_evidence(env={name: "true"}, read_file=_GO2) is None, name


def test_a_host_can_declare_itself_a_robot():
    """How the Lite3 fires this guard: nobody has measured a marker on it."""
    evidence = robot_host_evidence(env={ROBOT_HOST_ENV: "1"}, read_file=_NOTHING)
    assert evidence is not None
    assert ROBOT_HOST_ENV in evidence


def test_a_declaration_of_zero_wins_over_the_jetson_markers():
    assert robot_host_evidence(env={ROBOT_HOST_ENV: "0"}, read_file=_GO2) is None


def test_a_declaration_beats_ci_because_it_is_more_specific():
    assert robot_host_evidence(
        env={"CI": "true", ROBOT_HOST_ENV: "1"}, read_file=_NOTHING) is not None


def test_an_unparseable_declaration_does_not_disarm_the_guard():
    """A typo must fall through to the markers, not silently answer 'not a robot'.

    ``MAPPO_ROBOT_HOST=ture`` on a Go2 still refuses. An unset-is-a-value bug in the other
    direction would be a gate that a single keystroke turns off.
    """
    assert robot_host_evidence(env={ROBOT_HOST_ENV: "ture"}, read_file=_GO2) is not None
    assert robot_host_evidence(env={ROBOT_HOST_ENV: ""}, read_file=_GO2) is not None


# ------------------------------------------------------------------------ the decision

def test_it_refuses_a_live_run_from_the_system_python_on_the_go2():
    decision = evaluate(**_go2_live())
    assert decision.refuse is True
    assert "system Python on a robot" in decision.reason


def test_the_same_run_inside_a_virtualenv_is_allowed():
    decision = evaluate(**_go2_live(prefix="/home/unitree/robotics-connect-envs/armwaheed"))
    assert decision.refuse is False
    assert decision.evidence is not None, "it still knows it is on a robot"


def test_a_run_that_reaches_no_hardware_is_never_refused():
    """A shadow run, a telemetry replay and --help all reach this with a robot underneath."""
    decision = evaluate(**_go2_live(reaching_hardware=False))
    assert decision.refuse is False
    assert decision.evidence is None


def test_the_offline_suites_are_not_reddened_by_a_live_argument_namespace():
    """robot_bindings' own tests build --live namespaces on a laptop and in CI.

    Intent alone must never be enough, or adding this guard would have turned every one of
    those green tests red on a machine with no robot anywhere near it.
    """
    for env in ({}, {"GITHUB_ACTIONS": "true"}):
        decision = evaluate(**_go2_live(env=env, read_file=_NOTHING))
        assert decision.refuse is False, env
    decision = evaluate(**_go2_live(env={"GITHUB_ACTIONS": "true"}))
    assert decision.refuse is False, "a Jetson CI runner must still build"


def test_require_virtualenv_raises_system_exit_carrying_the_whole_message():
    try:
        require_virtualenv(**_go2_live())
    except SystemExit as exc:
        text = str(exc)
    else:
        raise AssertionError("require_virtualenv returned instead of refusing")
    assert "REFUSING TO RUN" in text
    assert "-m venv --system-site-packages" in text


def test_require_virtualenv_returns_the_decision_when_it_does_not_refuse():
    decision = require_virtualenv(**_go2_live(reaching_hardware=False))
    assert isinstance(decision, Decision)
    assert decision.refuse is False


# ------------------------------------------------------------------------- the message

def test_the_message_names_the_interpreter_to_build_a_venv_from():
    """Not 'use a venv' -- the actual command, with this machine's own python in it.

    The Go2 has 3.8.10 and 3.9.5 and the vendor stack is installed for 3.8, so 'the newest
    python on the box' would be actively wrong advice. The interpreter that is running is
    the one that can import what was installed for it.
    """
    text = refusal_message("mappo_drive", "NVIDIA Jetson", "/usr/bin/python3.8", "/usr")
    assert "/usr/bin/python3.8 -m venv --system-site-packages" in text
    assert "~/robotics-connect-envs/$USER" in text


def test_the_message_lists_the_venvs_that_already_exist():
    text = refusal_message(
        "mappo_drive", "NVIDIA Jetson", "/usr/bin/python3", "/usr",
        venvs=[("/home/unitree/robotics-connect-envs/armwaheed", "3.8.10")])
    assert "source /home/unitree/robotics-connect-envs/armwaheed/bin/activate" in text
    assert "3.8.10" in text


def test_the_message_forbids_the_two_things_an_agent_reaches_for():
    """pip install into the system Python, and installing a newer Python."""
    text = refusal_message("x", "y", "/usr/bin/python3", "/usr")
    assert "DO NOT pip install into the system Python" in text
    assert "python3.11" in text
    assert "finding to report, not a dependency to add" in text


def test_the_message_says_what_to_do_when_the_machine_is_not_a_robot():
    text = refusal_message("x", "y", "/usr/bin/python3", "/usr")
    assert f"{ROBOT_HOST_ENV}=0" in text


def test_there_is_no_documented_way_to_bypass_the_refusal():
    """An escape hatch printed next to a refusal is an instruction to use it.

    The only lever the message offers is a claim about the HOST, and the honest answer to
    'but my stack is in the system Python' is --system-site-packages, which the text gives.
    """
    text = refusal_message("x", "y", "/usr/bin/python3", "/usr")
    assert "ALLOW_SYSTEM_PYTHON" not in text
    assert "--system-site-packages is not optional" in text


# ------------------------------------------------------- visibility of a silent gate

def test_a_gate_that_is_not_enforcing_says_so_out_loud():
    """Detection defaults to 'not a robot', which is the shape of a gate that never fires.

    The live paths print this line every run, so the Lite3 not having a measured marker is
    something an operator reads rather than something nobody discovers.
    """
    note = describe(evaluate(**_go2_live(read_file=_NOTHING)))
    assert "not enforced" in note
    assert ROBOT_HOST_ENV in note


def test_describe_names_the_evidence_when_the_guard_is_armed_and_satisfied():
    note = describe(evaluate(**_go2_live(prefix="/home/unitree/env")))
    assert "pass" in note
    assert JETSON_RELEASE_FILE in note


def test_describe_reports_a_refusal_in_one_line():
    note = describe(evaluate(**_go2_live()))
    assert "REFUSED" in note
    assert "\n" not in note


# ------------------------------------------------------------------- venv discovery

def test_find_virtualenvs_reads_the_version_from_pyvenv_cfg():
    """Measured shape: the Go2's env, whose bin/python is a symlink to a symlink.

    Reading bin/python would report whatever /usr/bin/python3 points at today, which is
    not what built the env.
    """
    config = "home = /usr/bin\ninclude-system-site-packages = true\nversion = 3.8.10\n"
    found = find_virtualenvs(
        globs=("~/envs/*",),
        expanduser=lambda p: p.replace("~", "/home/unitree"),
        iglob=lambda p: ["/home/unitree/envs/armwaheed", "/home/unitree/envs/notavenv"],
        read_file=_fs(**{"/home/unitree/envs/armwaheed/pyvenv.cfg": config}))
    assert found == [("/home/unitree/envs/armwaheed", "3.8.10")]


def test_find_virtualenvs_never_raises_on_a_machine_with_nothing_there():
    assert find_virtualenvs(globs=("~/nope/*",), expanduser=lambda p: p,
                            iglob=lambda p: [], read_file=_NOTHING) == []


# ⚠️ Keep this at the END of the file. A `__main__` block placed mid-file stops every test
# below it from being collected, which cost this repository ten tests across two files.
if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"venv_guard: {len(tests)}/{len(tests)} passed")
