#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every module in this directory has to IMPORT under the interpreter running this suite.

WHAT THIS EXISTS FOR. Five modules here -- `blobs.py`, `build.py`, `checks.py`,
`p4plate.py` and `track.py` -- annotated with PEP 585 builtin generics (`-> list[str]`,
`tuple[int, ...]`) and carried no `from __future__ import annotations`, under a
`ruff.toml` declaring `target-version = "py38"`. Measured on CPython 3.8.20 at `900788b`,
every one of them died like this, at its first `def`, before doing any work:

    File "blobs.py", line 26, in <module>
      def register(plate: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, float]:
    TypeError: 'type' object is not subscriptable

Two things had to be true for that to survive in a repository with a 3.8 CI leg, and this
file is the second of them:

  * ruff's `target-version = "py38"` SUPPRESSES UP006, the rule that would rewrite
    `List[str]` into `list[str]`. It has no rule that calls the modern form too new for
    the declared target, so the config permitted precisely what it was written to forbid.
    That half is closed in `../ruff.toml` and `../../ruff.toml`, which now select `FA`.
  * `test_peercap.py` parses these modules as source and never imports them, so no test
    in this directory had ever executed a `def` in one. That half is this file.

The two halves are not redundant. `FA102` is static, sees only annotations, and runs on
every leg. This one is dynamic, sees everything an import can hit -- a stdlib name that
does not exist on 3.8, a `match` statement, a runtime `X | Y` -- and only tells you about
the interpreter it runs on. CI's 3.8 leg is the one that matters here; on 3.11 it still
pins that this package imports at all.

WHY IT COULD NOT BE WRITTEN BEFORE. Each script called `main()` at module level, so
importing one ran the whole tool. Those calls now sit behind `if __name__ == "__main__":`,
which changes nothing about running them as scripts and makes importing one resolve its
directories, define its functions, and stop -- which is exactly the part that has to work
on the robot's Python.

Needs `cv2` and `numpy`: nine of these modules import them at the top, and stubbing either
one out would hide the thing being checked. CI installs both on both legs.

Run: ``python3 test_pipeline_imports.py``
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

# The modules probed below do `import peercap`, a sibling. Running this file as a script
# puts its directory on `sys.path` already, which is how the suite is run and how
# `test_peercap.py` resolves the same import; this line makes the probe work from any
# working directory as well. It sits BELOW the imports on purpose -- there are no sibling
# imports in this file for `ruff --fix` to hoist above it, which is the failure AGENTS.md
# warns about and the reason nothing here is imported by name.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: Everything this directory ships except the test modules, which the runner already
#: imports by running them. Globbed rather than listed: a module added tomorrow is covered
#: without anybody remembering to add it here.
EVERY_MODULE = sorted(p for p in HERE.glob("*.py") if not p.name.startswith("test_"))

#: One frame name from the real capture, so the temporary directory below satisfies the
#: same `*.jpg` pre-flight `peercap.frames_dir()` puts in front of every consumer.
A_FRAME = "p1_close_broadside_0000.jpg"

#: A module body carrying the exact defect, used as this file's positive control. It is
#: NOT a copy of a rule: it is source, and the probe below runs it the same way it runs
#: the real modules. Under 3.9+ it imports cleanly, which is the whole reason five modules
#: shipped this shape; under 3.8 it raises. So the control that has to hold on every
#: interpreter is the one further down -- that the probe REPORTS a module that raises.
THE_DEFECT = "def annotated() -> list[int]:\n    return []\n"


@contextlib.contextmanager
def a_capture():
    """A temporary frames directory plus its work and output siblings, all cleaned up.

    `PEERCAP_WORK` and `PEERCAP_LABELLED` are set explicitly rather than left to default
    beside the frames, so that `test_importing_a_module_writes_nothing` can look at a
    directory nothing else has touched.
    """
    before = {key: os.environ.get(key)
              for key in ("PEERCAP_FRAMES", "PEERCAP_WORK", "PEERCAP_LABELLED")}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames"
        frames.mkdir()
        (frames / A_FRAME).write_bytes(b"")
        try:
            os.environ["PEERCAP_FRAMES"] = str(frames)
            os.environ["PEERCAP_WORK"] = str(root / "work")
            os.environ["PEERCAP_LABELLED"] = str(root / "labelled")
            yield root
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _import_error(path: Path):
    """Import `path` as a module of its own; return the exception it raised, or None.

    Loaded from the file rather than by module name so the probe can be pointed at a
    temporary file too, and so nothing here depends on what a previous test left in
    `sys.modules`. The module name is deliberately not `"__main__"`, which is what keeps
    each script's `main()` from running.
    """
    name = "_pipeline_import_probe_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:  # reporting it, rather than raising, IS the job
        return error
    finally:
        sys.modules.pop(name, None)
    return None


def test_the_probe_found_every_module_in_this_directory():
    """A glob that matched nothing would make the next test pass over an empty list."""
    names = {path.name for path in EVERY_MODULE}
    assert "peercap.py" in names, names
    assert len(names) >= 10, f"only {len(names)} modules discovered: {sorted(names)}"
    assert not any(name.startswith("test_") for name in names), names


def test_the_probe_reports_a_module_that_cannot_import():
    """The control. Without it, a probe that swallowed every exception would pass forever.

    Two modules are fed to it: one whose body raises, and one carrying THE_DEFECT. The
    first must be reported on any interpreter. The second is only an error on 3.8 -- it is
    asserted conditionally for that reason, and the version test is the honest way to say
    that this file's real verdict is delivered by CI's 3.8 leg.
    """
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.py"
        broken.write_text("raise RuntimeError('this module cannot import')\n")
        error = _import_error(broken)
        assert isinstance(error, RuntimeError), f"the probe returned {error!r}"
        assert "cannot import" in str(error), error

        defective = Path(tmp) / "defective.py"
        defective.write_text(THE_DEFECT)
        error = _import_error(defective)
        if sys.version_info < (3, 9):
            assert isinstance(error, TypeError), f"3.8 must reject PEP 585: {error!r}"
            assert "not subscriptable" in str(error), error
        else:
            assert error is None, f"PEP 585 is legal on this interpreter: {error!r}"


def test_every_module_in_this_directory_imports():
    """The gate. Five modules failed this on 3.8.20 before `from __future__ import
    annotations` was added to them, and none failed it on 3.11."""
    with a_capture():
        failures = []
        for path in EVERY_MODULE:
            error = _import_error(path)
            if error is not None:
                failures.append(f"{path.name}: {type(error).__name__}: {error}")
    assert not failures, (
        f"{len(failures)} of {len(EVERY_MODULE)} modules do not import under Python "
        f"{'.'.join(str(v) for v in sys.version_info[:3])}:\n  " + "\n  ".join(failures)
    )


def test_importing_a_module_writes_nothing():
    """Importing a tool must not RUN it.

    Every script here used to call `main()` at module level; `build.py`'s writes
    `annotations.json`. If a guard is ever dropped the test above goes red first, but it
    goes red with whatever that script happened to fail on, which is not a message that
    names the cause. This one names it.
    """
    with a_capture() as root:
        for path in EVERY_MODULE:
            _import_error(path)
        written = sorted(p.name for p in root.rglob("*")
                         if p.is_file() and p.name != A_FRAME)
    assert not written, f"importing this directory wrote {written}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"pipeline imports: {len(tests)}/{len(tests)} passed")
    sys.exit(0)
