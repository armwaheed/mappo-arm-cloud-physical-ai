#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``start-dashboard.sh``: what it refuses, and that it takes its children with it.

**The interpreter and package checks are tested against a STUB interpreter, not against
this machine's.** A test that asserts "Python 3.9 is refused" by running the local
``/usr/bin/python3`` passes on a Mac and does nothing at all on a CI runner whose
``python3`` is 3.11 — the condition under test would be supplied by the environment rather
than by the test. The stub is a real interpreter with ``sys.version_info`` overwritten and
a meta-path finder that refuses named modules, so the launcher's own probes run unchanged
and the answer they get is the one this file chose. It also means these tests do not need
``device-connect-edge`` installed, which CI does not have.

The teardown test is the exception and is deliberately end-to-end: three real child
processes, a real SIGTERM, and an assertion that all three are gone. "Three processes
started together stop together" is not a claim a stub can make.

Pure stdlib. ``python3 test_start_dashboard.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "start-dashboard.sh"

#: What the launcher must be able to import before it starts anything, as (import, pip).
REQUIRED = (("device_connect_edge", "device-connect-edge"),
            ("device_connect_agent_tools", "device-connect-agent-tools"),
            ("aiohttp", "aiohttp"),
            ("numpy", "numpy"))


def _stub_interpreter(directory: Path, *, version=(3, 11, 0), blocked=()) -> Path:
    """A real interpreter that reports ``version`` and refuses to import ``blocked``.

    Everything else about it is this interpreter, so the launcher's probes — a ``-c`` that
    prints ``sys.version_info``, a ``-c`` that exits on a comparison, and a ``-c import X``
    per package — run for real against an answer this file controls.
    """
    stub = directory / "stub-python"
    stub.write_text(textwrap.dedent(f'''\
        #!{sys.executable}
        import collections, runpy, sys

        _V = collections.namedtuple(
            "version_info", "major minor micro releaselevel serial")
        sys.version_info = _V({version[0]}, {version[1]}, {version[2]}, "final", 0)

        _BLOCKED = {set(blocked)!r}


        class _Refuse:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in _BLOCKED:
                    raise ImportError(f"No module named {{fullname!r}}")
                return None


        sys.meta_path.insert(0, _Refuse())

        argv = sys.argv[1:]
        if argv and argv[0] == "-c":
            sys.argv = ["-c"] + argv[2:]
            exec(compile(argv[1], "<stub -c>", "exec"), {{"__name__": "__main__"}})
        else:
            sys.argv = argv
            runpy.run_path(argv[0], run_name="__main__")
        '''))
    stub.chmod(0o755)
    return stub


def _package(directory: Path) -> Path:
    """The smallest thing the launcher accepts as a policy package."""
    package = directory / "package"
    (package / "models").mkdir(parents=True)
    (package / "config.json").write_text(json.dumps({"model_path": "models/x.npz"}))
    return package


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(*args, cwd=None, env=None, timeout=60):
    """Run the launcher. Ports default to FREE ones, never to 8080/8800.

    A test that left the defaults in place refused on "port 8080 already in use" — because
    the machine it was written on had a dashboard running, which is the launcher working
    correctly and the test asking the wrong question.
    """
    argv = [str(a) for a in args]
    if "--port" not in argv:
        argv += ["--port", str(_free_port())]
    if "--model-port" not in argv:
        argv += ["--model-port", str(_free_port())]
    return subprocess.run([str(LAUNCHER), *argv],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=cwd or str(HERE), env=env)


def _driver_line(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("robot_driver "):
            return line
    raise AssertionError(f"no robot_driver line in a --dry-run:\n{stdout}")


# --------------------------------------------------------------------------------------
# 1. the interpreter

def test_an_interpreter_below_311_is_refused_and_the_message_names_the_version():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, version=(3, 9, 6))
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "3.9" in done.stderr, done.stderr
    assert ">= 3.11" in done.stderr, done.stderr
    # The failure it must NOT look like: a traceback about a module.
    assert "Traceback" not in done.stderr, done.stderr


def test_the_refusal_names_an_interpreter_that_would_work():
    """Naming the problem is half of it; the operator still has to know what to type."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, version=(3, 9, 6))
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
    stderr = done.stderr
    if "Run instead" in stderr:
        assert "--python python3" in stderr, stderr
    else:
        # No >= 3.11 on this PATH at all, which is the other half of the same message.
        assert "install" in stderr.lower(), stderr
        assert "virtualenv will not help" in stderr, stderr


def test_a_venv_is_not_offered_as_a_way_round_the_version():
    """It is the wrong answer, and it is the one people reach for on a robot.

    A venv is built FROM an interpreter and cannot supply a version the machine does not
    have. AGENTS.md forbids installing a newer Python on a robot for this reason, so the
    launcher must not suggest the thing that looks like it would work.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, version=(3, 8, 10))
        # PATH is PREPENDED rather than replaced: emptying it would take dirname, mktemp
        # and sed with it, and the script would die for a reason that is not the one under
        # test. Every name the launcher searches for resolves to the 3.8 stub instead.
        shadow = tmp / "path"
        shadow.mkdir()
        for name in ("python3", "python3.11", "python3.12", "python3.13", "python3.14"):
            (shadow / name).symlink_to(stub)
        env = dict(os.environ, PATH=f"{shadow}{os.pathsep}{os.environ['PATH']}")
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp), env=env)
    assert done.returncode == 2, done.stdout
    assert "cannot supply a version the machine does not have" in done.stderr, done.stderr


# --------------------------------------------------------------------------------------
# 2. the packages

def test_a_missing_package_is_reported_as_a_pip_line_and_not_as_a_traceback():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, blocked=[name for name, _ in REQUIRED])
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
    assert done.returncode == 2, done.stdout
    assert "Traceback" not in done.stderr, done.stderr
    assert f"{stub} -m pip install" in done.stderr, done.stderr
    for _, pip_name in REQUIRED:
        assert pip_name in done.stderr, f"{pip_name} missing from:\n{done.stderr}"


def test_the_pip_line_names_only_what_is_actually_missing():
    """So the operator can paste it at an environment that is half set up already."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, blocked=["aiohttp", "numpy"])
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
    assert done.returncode == 2, done.stdout
    line = next(ln for ln in done.stderr.splitlines() if "-m pip install" in ln)
    assert "aiohttp" in line and "numpy" in line, line
    assert "device-connect-edge" not in line, line


def test_aiohttp_and_numpy_are_checked_at_all():
    """They are the two nothing else installs, and the line in circulation omitted both.

    ``device-connect-agent-tools`` depends only on ``device-connect-edge``, which depends
    on eclipse-zenoh, nats-py, nkeys, pydantic and pyyaml. MEASURED in a clean 3.11 venv:
    installing the two Device Connect packages plus eclipse-zenoh leaves ``server.py``
    dying at ``No module named 'aiohttp'`` and ``robot_driver.py`` at ``numpy``.
    """
    for blocked, expected in (("aiohttp", "aiohttp"), ("numpy", "numpy")):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stub = _stub_interpreter(tmp, blocked=[blocked])
            done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
        assert done.returncode == 2, f"{blocked} was not checked:\n{done.stdout}"
        assert expected in done.stderr, done.stderr


def test_eclipse_zenoh_is_not_asked_for_because_something_else_installs_it():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp, blocked=[name for name, _ in REQUIRED])
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp))
    assert "eclipse-zenoh" not in done.stderr, \
        "eclipse-zenoh is a dependency of device-connect-edge; naming it teaches the wrong line"


# --------------------------------------------------------------------------------------
# 3. motion

def test_motion_is_off_unless_the_flag_is_typed():
    """No environment variable may enable it. This is the launcher's --live."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp)
        env = dict(os.environ, ALLOW_MOTION="1", MAPPO_ALLOW_MOTION="1",
                   MAPPO_LIVE="1", ALLOW_MOTION_DEFAULT="1")
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp), env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--allow-motion" not in _driver_line(done.stdout), done.stdout


def test_the_flag_reaches_the_driver_when_it_is_typed():
    """The complement, so the test above cannot pass by the flag never working at all."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp)
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp),
                    "--allow-motion")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--allow-motion" in _driver_line(done.stdout), done.stdout


# --------------------------------------------------------------------------------------
# 4. the arguments an operator gets wrong

def test_robot_is_shorthand_for_a_camera_url_a_go2_and_a_simulated_pose():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp)
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp),
                    "--robot", "10.1.2.3")
    line = _driver_line(done.stdout)
    assert "--camera-url http://10.1.2.3:8801/" in line, line
    assert "--platform go2" in line, line
    # --simulate is not a detail: it is the difference between a real pose and a bench
    # double's, and the README's claim about which half is real depends on it.
    assert "--simulate" in line, line


def test_robot_and_camera_url_together_are_refused_rather_than_one_winning():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp)
        done = _run("--python", stub, "--dry-run", "--package", _package(tmp),
                    "--robot", "10.1.2.3", "--camera-url", "http://elsewhere/")
    assert done.returncode == 2, done.stdout
    assert "--robot and --camera-url" in done.stderr, done.stderr


def test_a_directory_that_is_not_a_policy_package_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = _stub_interpreter(tmp)
        (tmp / "empty" / "models").mkdir(parents=True)
        done = _run("--python", stub, "--dry-run", "--package", tmp / "empty")
    assert done.returncode == 2, done.stdout
    assert "no config.json" in done.stderr, done.stderr


def test_an_option_that_swallowed_the_next_option_is_refused():
    """`--robot --port 8090` would otherwise build http://--port:8801/ and fail much later."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp), "--robot", "--simulate")
    assert done.returncode == 2, done.stdout
    assert "--robot was given '--simulate'" in done.stderr, done.stderr


def test_a_bind_that_is_not_loopback_says_there_is_no_login():
    """--host 0.0.0.0 puts an unauthenticated motion pad on the demo LAN."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp), "--host", "0.0.0.0")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NO LOGIN" in done.stdout, done.stdout


def test_loopback_does_not_carry_that_warning():
    """So the one above is a signal and not decoration on every run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NO LOGIN" not in done.stdout, done.stdout


def test_an_unknown_argument_is_refused_rather_than_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp), "--allow-motions")
    assert done.returncode == 2, done.stdout
    assert "--allow-motions" in done.stderr, done.stderr


# --------------------------------------------------------------------------------------
# 5. the ports

def test_a_port_already_in_use_is_refused_before_anything_starts():
    """The failure this replaces arrives from inside a Python server, minutes later.

    Detection is a bind attempt rather than lsof, so this holds on an image with no lsof.
    """
    with tempfile.TemporaryDirectory() as tmp, socket.socket() as held:
        tmp = Path(tmp)
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp), "--port", port,
                    "--model-port", _free_port())
    assert done.returncode == 2, done.stdout
    assert f"port {port}" in done.stderr and "already in use" in done.stderr, done.stderr


def test_the_checkpoint_servers_port_is_checked_too_and_not_only_the_dashboards():
    with tempfile.TemporaryDirectory() as tmp, socket.socket() as held:
        tmp = Path(tmp)
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        done = _run("--python", _stub_interpreter(tmp), "--dry-run",
                    "--package", _package(tmp), "--port", _free_port(),
                    "--model-port", port)
    assert done.returncode == 2, done.stdout
    assert "the checkpoint server" in done.stderr, done.stderr


# --------------------------------------------------------------------------------------
# 6. and the one that has to be end-to-end

def _children_of(pid: int) -> list:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == str(pid):
            kids.append(int(parts[0]))
    return kids


def test_all_three_children_stop_when_the_launcher_does():
    """A demo that leaves processes on 8080 and 8800 makes the NEXT demo fail to start.

    Deliberately not a stub assertion: three real processes are started, the launcher is
    signalled, and the process table is read. Nothing about a trap being present in the
    source would prove this — the first version of this script stopped all three and then
    resumed its own watchdog loop, which is a different bug the source also looked fine for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # A stub whose "python" runs the given script for real. server.py is replaced by
        # something that binds the port, so the launcher's readiness probe succeeds and the
        # test does not wait out the timeout.
        stub = _stub_interpreter(tmp)
        fake = tmp / "fake"
        fake.mkdir()
        (fake / "model_server.py").write_text("import time\nwhile True: time.sleep(0.2)\n")
        # The driver stands in for the one whose motion worker damps on SIGTERM. It records
        # that it got a SIGTERM, so "the children are gone" cannot be satisfied by a bare
        # SIGKILL — which would leave a walking robot's last velocity latched on the bus,
        # because SportClient.Move has no dead-man timeout.
        (fake / "robot_driver.py").write_text(textwrap.dedent(f'''\
            import signal, sys, time
            def _damp(*a):
                open({str(tmp / "term.marker")!r}, "w").write("SIGTERM")
                sys.exit(0)
            signal.signal(signal.SIGTERM, _damp)
            while True: time.sleep(0.2)
            '''))
        (fake / "server.py").write_text(textwrap.dedent('''\
            import http.server, socketserver, sys
            port = int(sys.argv[sys.argv.index("--port") + 1])
            class H(http.server.BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def do_GET(self):
                    self.send_response(200); self.send_header("Content-Length", "2")
                    self.end_headers(); self.wfile.write(b"ok")
            socketserver.TCPServer.allow_reuse_address = True
            socketserver.TCPServer(("127.0.0.1", port), H).serve_forever()
            '''))
        launcher = fake / LAUNCHER.name
        shutil.copy2(LAUNCHER, launcher)
        term_marker = tmp / "term.marker"

        proc = subprocess.Popen(
            [str(launcher), "--python", str(stub), "--package", str(_package(tmp)),
             "--port", str(_free_port()), "--model-port", str(_free_port()),
             "--log-dir", str(tmp / "logs")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            kids = []
            for _ in range(200):
                kids = _children_of(proc.pid)
                if len(kids) >= 3:
                    break
                time.sleep(0.1)
            assert len(kids) >= 3, f"only {len(kids)} children started: {kids}"

            proc.send_signal(signal.SIGTERM)
            out = proc.communicate(timeout=30)[0]

            for _ in range(100):
                alive = [k for k in kids if _is_alive(k)]
                if not alive:
                    break
                time.sleep(0.1)
            # Read INSIDE the with: the marker lives in the temporary directory, and
            # asserting on it after the block deletes it is a test that can only fail.
            got_sigterm = term_marker.exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    assert not alive, f"the launcher exited and left {alive} running"
    # And it must not report its own clean shutdown as a crash. A single
    # `trap cleanup EXIT INT TERM` stopped all three and then RESUMED the watchdog loop,
    # which found the children it had just killed and printed twenty lines of a log.
    assert "Stopping the other two" not in out, out
    assert "exited during startup" not in out, out
    assert got_sigterm, \
        "the driver was killed without a SIGTERM first, so its motion damp never ran"


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie answers signal 0. ps tells the two apart.
    state = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"start_dashboard: {len(tests)}/{len(tests)} passed")
