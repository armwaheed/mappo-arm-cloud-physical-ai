#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for peercap.py, plus one pin on every script in this directory.

The pin is the point. `test_no_module_here_evaluates_an_absolute_path` is what stops the
defect this directory shipped for two days coming back: nine scripts hard-coding a
scratchpad path, which is how 2,800 frames came to be declared lost in issue #77. It walks
the AST of every module here and fails if any *evaluates* a string that begins at a
filesystem root. Quoting the dead path in prose is fine and deliberate — `peercap.py` does
it, because deleting the string would lose the lesson (#86 kept it under `source.was` for
the same reason). Evaluating one is not.

The other tests are about refusing well. A tool that cannot find its corpus used to raise
`IndexError` on `f[0]` twenty lines after an empty glob, which reads as a broken script
rather than as a corpus that is somewhere else.

These tests do not import the nine scripts. They cannot: every one resolves its paths at
import and refuses when `PEERCAP_FRAMES` is unset, which is the behaviour being tested.
They are read and parsed as source instead.

Run: ``python3 test_peercap.py``. Needs neither cv2 nor numpy.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import peercap

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

SELF = os.path.basename(__file__)

#: Every runnable script here, i.e. everything but peercap.py and this file.
SCRIPTS = sorted(p for p in HERE.glob("*.py") if p.name not in {"peercap.py", SELF})

#: Everything this directory ships, scripts and support alike.
EVERY_MODULE = [*SCRIPTS, HERE / "peercap.py", HERE / SELF]

#: A rooted path, ASSEMBLED rather than written, so the scan below does not flag its own
#: positive control. That control is what stops the scan degrading into a gate that never
#: fires: with no offenders in the tree, a predicate hard-wired to False passes forever.
A_ROOTED_PATH = os.sep + os.path.join("private", "tmp", "scratchpad", "peercap") + os.sep

#: Which peercap accessor each name is allowed to come from.
ACCESSOR = {"SRC": "frames_dir", "WORK": "work_dir", "OUT": "labelled_dir"}


@contextlib.contextmanager
def env(**values):
    """Set/clear env vars and restore EXACTLY what was there, absence included.

    Restoring an unset variable by setting it to "" would leave the next test looking at a
    value rather than at nothing, and "" and unset are handled by the same branch here only
    because that branch was written to treat them alike.
    """
    before = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def a_capture(*names):
    """A temporary directory holding `names` as empty .jpg files."""
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            (Path(tmp) / name).write_bytes(b"")
        yield Path(tmp)


def _refusal(**environment):
    """``(exit_code, stdout, stderr)`` from one `frames_dir()` call."""
    out, err = io.StringIO(), io.StringIO()
    with env(**environment):
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                peercap.frames_dir()
        except SystemExit as exc:
            return exc.code, out.getvalue(), err.getvalue()
    raise AssertionError("frames_dir() returned where it should have refused")


def test_an_unset_frames_variable_refuses_rather_than_globbing_nothing():
    code, out, err = _refusal(PEERCAP_FRAMES=None)
    assert code == 2, code
    assert "PEERCAP_FRAMES is not set" in err, err
    assert out == "", "the refusal must not go to stdout, where a pipe would eat it"


def test_the_refusal_names_the_locations_and_the_issue_that_paid_for_them():
    """A refusal that only says "not found" sends the reader looking for the frames again.
    Every location the corpus is reported to be in has to be in the message itself."""
    _, _, err = _refusal(PEERCAP_FRAMES=None)
    for expected in ("arm-seattle-spark-02:~/go2-peer-dataset-20260824/",
                     "go2-peer-detection",
                     "detector/labels/peer_go2wheel_20260824.json",
                     "issue #77",
                     "NEITHER has been verified from this repository"):
        assert expected in err, f"{expected!r} missing from:\n{err}"


def test_a_frames_directory_that_does_not_exist_refuses_and_says_which_one():
    with tempfile.TemporaryDirectory() as tmp:
        gone = str(Path(tmp) / "not-here")
    code, _, err = _refusal(PEERCAP_FRAMES=gone)
    assert code == 2 and "is not a directory" in err, err
    assert gone in err, err


def test_an_empty_frames_directory_refuses_instead_of_scoring_zero():
    """The failure mode that costs the most: a directory that exists but holds nothing.
    Every downstream count comes out zero and reads as a measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        code, _, err = _refusal(PEERCAP_FRAMES=tmp)
    assert code == 2 and "holds no .jpg" in err, err


def test_a_real_capture_resolves_and_keeps_its_trailing_separator():
    """Every call site is ``SRC + name``. Without the separator the concatenation is a
    valid string naming nothing, and the glob comes back empty — silently."""
    with a_capture("p1_close_broadside_0000.jpg") as frames, env(PEERCAP_FRAMES=str(frames)):
        resolved = peercap.frames_dir()
        assert resolved == str(frames) + "/", resolved
        assert (Path(resolved) / "p1_close_broadside_0000.jpg").is_file()


def test_the_work_and_output_directories_default_beside_the_frames_and_are_created():
    """Naming the one durable location has to be enough to run the whole pipeline."""
    with a_capture("p1_close_broadside_0000.jpg") as frames, \
            env(PEERCAP_FRAMES=str(frames), PEERCAP_WORK=None, PEERCAP_LABELLED=None):
        work, labelled = peercap.work_dir(), peercap.labelled_dir()
    assert work == str(frames.parent / "peercap_work") + "/", work
    assert labelled == str(frames.parent / "peercap_labelled") + "/", labelled
    assert Path(work).is_dir() and Path(labelled).is_dir()


def test_the_work_and_output_directories_can_be_moved_off_the_frames_volume():
    """The frames may be a read-only or shared dataset mount; the intermediates are not."""
    with a_capture("p1_close_broadside_0000.jpg") as frames, tempfile.TemporaryDirectory() as tmp:
        elsewhere = str(Path(tmp) / "w")
        with env(PEERCAP_FRAMES=str(frames), PEERCAP_WORK=elsewhere, PEERCAP_LABELLED=None):
            assert peercap.work_dir() == elsewhere + "/"
            assert Path(elsewhere).is_dir(), "an override still has to be created"


def _is_a_location(value: str) -> bool:
    """A rooted path with at least one component. A bare separator is not one.

    ``"/"`` is what ``rsplit("/", 1)`` and ``str(path) + "/"`` pass around all over this
    directory; treating it as a location would make the pin below unfailable-in-practice,
    which is worse than not having it.
    """
    return (value.startswith("/") and len(value) > 1) or \
           (value.startswith("~/") and len(value) > 2)


def test_no_module_here_evaluates_an_absolute_path():
    """The regression pin for the whole defect class.

    Nine scripts in this directory carried ``SCRATCH = ("/private/tmp/..." "...")`` — one
    string, evaluated, pointing at a machine-local scratchpad. Reading that string as the
    corpus's only location is why the Aug-24 capture was written off in #77. A docstring
    that *quotes* it is what keeps the lesson; a constant that *is* it is the defect.
    """
    assert _is_a_location(A_ROOTED_PATH), "the predicate cannot fire; this pin checks nothing"
    assert not _is_a_location(os.sep), "a bare separator is not a location"

    offenders = []
    for path in EVERY_MODULE:
        tree = ast.parse(path.read_text())
        docstrings = {id(ast.get_docstring(n, clean=False)) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef,
                                        ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node.value) in docstrings:
                continue
            if _is_a_location(node.value):
                offenders.append(f"{path.name}:{node.lineno} {node.value[:60]!r}")
    assert not offenders, "absolute path evaluated:\n  " + "\n  ".join(offenders)


def test_every_script_takes_its_directories_from_peercap():
    """Not just "no literal": the names the scripts read must come from the accessors.

    A path rebuilt out of ``os.environ`` in one script would pass the test above and still
    put this directory back to nine independent answers about where the corpus is.
    """
    for path in SCRIPTS:
        tree = ast.parse(path.read_text())
        assert any(isinstance(n, ast.Import) and any(a.name == "peercap" for a in n.names)
                   for n in tree.body), f"{path.name} does not import peercap"
        bound = 0
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in ACCESSOR:
                continue
            bound += 1
            call = node.value
            assert isinstance(call, ast.Call), f"{path.name}: {target.id} is not a call"
            assert isinstance(call.func, ast.Attribute), f"{path.name}: {target.id}"
            assert isinstance(call.func.value, ast.Name) and call.func.value.id == "peercap", \
                f"{path.name}: {target.id} does not come from peercap"
            assert call.func.attr == ACCESSOR[target.id], \
                f"{path.name}: {target.id} = peercap.{call.func.attr}()"
        assert bound, f"{path.name} binds none of {sorted(ACCESSOR)}"


def test_every_script_still_carries_the_licence_header():
    """148 of the repository's 160 modules do; these nine were the gap."""
    for path in EVERY_MODULE:
        head = path.read_text().splitlines()[:3]
        assert any("SPDX-License-Identifier: Apache-2.0" in line for line in head), path.name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"peercap: {len(tests)}/{len(tests)} passed")
    sys.exit(0)
