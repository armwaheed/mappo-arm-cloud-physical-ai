#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint inventory: what may be armed, what may be deleted, and why not.

Every test here is a way to leave the robot in a state the next live run cannot start from.
That is the whole job of this module — the swap itself is one line of JSON.

The last test is the one that keeps this file honest: the shape rule is re-stated in
``model_store`` rather than imported from the vendored policy package, so it is fed the same
files as ``physical_ai_mappo._Actor`` and asserted to reach the same verdict. A duplicated
rule that is checked against its original is a different thing from one that drifts.

Needs numpy. ``python3 test_model_store.py``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_store import (
    ModelStore,
    ModelStoreError,
    inspect_model,
    safe_model_name,
)

REPO = Path(__file__).resolve().parent.parent
REAL_CHECKPOINT = REPO / "policy" / "models" / "mappo_actor_3agent_1910000.npz"


# ── fixtures ─────────────────────────────────────────────────────────────────
def _actor_arrays(lidar_range=0.35, rays=12, obs=18):
    layout = ["x", "y", "vx", "vy", "dx", "dy"] + [f"lidar_{i}" for i in range(rays)]
    return {
        "W1": np.zeros((256, obs), dtype=np.float32),
        "b1": np.zeros((256,), dtype=np.float32),
        "W2": np.zeros((256, 256), dtype=np.float32),
        "b2": np.zeros((256,), dtype=np.float32),
        "W3": np.zeros((4, 256), dtype=np.float32),
        "b3": np.zeros((4,), dtype=np.float32),
        "metadata_json": np.array(json.dumps({
            "actor_input_dim": obs,
            "observation_layout": layout,
            "training_lidar_range_vmas": lidar_range,
            "training_agent_radius_vmas": 0.1,
        })),
    }


def _write(path, **overrides):
    arrays = _actor_arrays(**overrides)
    np.savez(path, **arrays)
    return Path(path)


def _package(tmp, lidar_range=0.35):
    """A policy package: config.json plus one armed checkpoint."""
    package = Path(tmp)
    (package / "models").mkdir(parents=True, exist_ok=True)
    _write(package / "models" / "armed.npz", lidar_range=lidar_range)
    (package / "config.json").write_text(json.dumps({
        "model_path": "models/armed.npz",
        "meters_per_vmas_unit": 2.5,
        "lidar_range_vmas": lidar_range,
    }, indent=2))
    return ModelStore(package)


# ── names ────────────────────────────────────────────────────────────────────
def test_a_name_that_escapes_the_models_directory_is_rejected_not_sanitised():
    """The name comes from an S3 key or a URL, i.e. from outside the robot.

    Rejecting rather than rewriting is the point: ``Path(name).name`` would turn
    ``../../authorized_keys`` into a plausible filename and lose the fact that something
    tried to escape.
    """
    for hostile in ("../evil.npz", "../../etc/passwd.npz", "/abs/path.npz",
                    "models/nested.npz", ".hidden.npz", "sub\\win.npz"):
        try:
            safe_model_name(hostile)
        except ModelStoreError:
            continue
        raise AssertionError(f"{hostile!r} was accepted as a checkpoint name")


def test_a_name_that_is_not_an_npz_is_rejected():
    for wrong in ("actor.pt", "actor", "actor.npz.txt"):
        try:
            safe_model_name(wrong)
        except ModelStoreError:
            continue
        raise AssertionError(f"{wrong!r} was accepted")
    assert safe_model_name("actor_v2-final.npz") == "actor_v2-final.npz"


# ── inspection ───────────────────────────────────────────────────────────────
def test_the_real_delivered_checkpoint_inspects_clean():
    """Run against the checkpoint the demo actually ships, not only synthetic ones."""
    if not REAL_CHECKPOINT.is_file():
        print("  skip  real checkpoint not present")
        return
    report = inspect_model(REAL_CHECKPOINT)
    assert report.loadable, report.problems
    assert report.rays == 12, report.rays
    assert report.trained_lidar_range_vmas == 0.35
    assert report.size_bytes == 268063, report.size_bytes
    assert report.sha256 == (
        "7327f72401adfdfa1931a516e85aeee62b5bee0e06e976c13600515ca2d2ca11"), report.sha256


def test_a_wrong_shaped_network_is_not_loadable():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wrong.npz"
        np.savez(path, W1=np.zeros((8, 8), dtype=np.float32))
        report = inspect_model(path)
        assert not report.loadable
        assert any("missing array" in p for p in report.problems), report.problems


def test_a_file_that_is_not_an_npz_at_all_reports_rather_than_raises():
    """A corrupt download is an ordinary outcome, so it must come back as a report.

    If this raised, the driver would have to catch an exception to build the same message it
    already builds from ``problems`` — and a caller that forgot would 500 a web request.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "junk.npz"
        path.write_bytes(b"this is not a zip file")
        report = inspect_model(path)
        assert not report.loadable
        assert any("could not be read" in p for p in report.problems), report.problems


def test_a_checkpoint_with_a_different_ray_count_is_refused():
    """Issue #29 asks for a 24-ray retrain. This adapter casts 12 and cannot drive it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "wide.npz", rays=24, obs=30)
        report = inspect_model(path)
        assert not report.loadable
        assert any("lidar features" in p for p in report.problems), report.problems


# ── the swap ─────────────────────────────────────────────────────────────────
def test_arming_a_checkpoint_rewrites_only_model_path():
    """A config someone hand-tuned must not be reformatted or reordered by a button press."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        _write(store.models_dir / "second.npz")
        before = store.config()
        result = store.select("second.npz")
        after = store.config()

        assert result["active"] == "second.npz"
        assert result["previous"] == "armed.npz"
        assert after["model_path"] == "models/second.npz"
        assert {k: v for k, v in after.items() if k != "model_path"} == \
               {k: v for k, v in before.items() if k != "model_path"}


def test_a_checkpoint_trained_at_a_different_lidar_range_is_refused():
    """The failure this prevents is silent at swap time and fatal at the start of the run.

    ``MappoController._check_against_checkpoint`` raises on exactly this mismatch. Catching
    it here turns a crash in the arena into a refusal at the click.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp, lidar_range=0.35)
        _write(store.models_dir / "other.npz", lidar_range=0.50)
        try:
            store.select("other.npz")
        except ModelStoreError as exc:
            assert "0.5" in str(exc) and "0.35" in str(exc), str(exc)
            assert store.active_model() == "armed.npz", "the config was written anyway"
            return
        raise AssertionError("a mismatched training range was armed")


def test_the_armed_checkpoint_cannot_be_deleted():
    """Deleting it points config.json at a file that is not there.

    The failure would surface at the start of the next live run — with the operator already
    in the arena — rather than here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        try:
            store.remove("armed.npz")
        except ModelStoreError as exc:
            assert "armed" in str(exc)
            assert (store.models_dir / "armed.npz").is_file()
            return
        raise AssertionError("the armed checkpoint was deleted")


def test_a_non_armed_checkpoint_can_be_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        _write(store.models_dir / "spare.npz")
        result = store.remove("spare.npz")
        assert result["removed"] == "spare.npz"
        assert result["freed_bytes"] > 0
        assert not (store.models_dir / "spare.npz").exists()


def test_installing_a_bad_checkpoint_never_lands_it():
    """A file that cannot be driven must never appear in a list someone selects from."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        bad = Path(tmp) / "incoming.npz"
        np.savez(bad, W1=np.zeros((3, 3), dtype=np.float32))
        try:
            store.install("incoming.npz", bad)
        except ModelStoreError:
            assert not (store.models_dir / "incoming.npz").exists()
            assert [m["name"] for m in store.list_models()] == ["armed.npz"]
            return
        raise AssertionError("an undrivable checkpoint was installed")


def test_installing_over_the_armed_checkpoint_is_refused():
    """Replacing the bytes under the config's feet is a swap no event records."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        incoming = _write(Path(tmp) / "incoming.npz")
        try:
            store.install("armed.npz", incoming, overwrite=True)
        except ModelStoreError as exc:
            assert "armed" in str(exc)
            return
        raise AssertionError("the armed checkpoint's file was replaced")


def test_the_listing_flags_an_incompatible_checkpoint_without_refusing_to_show_it():
    """An operator must be able to see a checkpoint they cannot currently arm.

    Hiding it would make "why is my checkpoint not there" the question, and the answer is
    more useful than the absence.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp, lidar_range=0.35)
        _write(store.models_dir / "other.npz", lidar_range=0.50)
        listing = {m["name"]: m for m in store.list_models()}
        assert set(listing) == {"armed.npz", "other.npz"}
        assert listing["other.npz"]["loadable"] is True       # the file is a valid network
        assert listing["other.npz"]["compatible_with_config"] is False  # ...but not here
        assert listing["armed.npz"]["active"] is True


def test_a_config_write_is_atomic_and_leaves_no_temp_file():
    with tempfile.TemporaryDirectory() as tmp:
        store = _package(tmp)
        _write(store.models_dir / "second.npz")
        store.select("second.npz")
        leftovers = [p.name for p in store.package_dir.glob("*.tmp")]
        assert not leftovers, leftovers
        json.loads(store.config_path.read_text())    # still valid JSON


def test_a_package_without_a_config_is_refused_at_construction():
    """Pointing --package at the wrong directory is the one way a swap silently does nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ModelStore(tmp)
        except ModelStoreError as exc:
            assert "config.json" in str(exc)
            return
        raise AssertionError("a directory with no config.json was accepted")


# ── the equivalence that keeps the duplicated rule honest ─────────────────────
def test_this_modules_verdict_matches_the_policy_packages_actor_loader():
    """Feed the same files to both implementations and assert they agree.

    ``model_store`` re-states the network contract rather than importing the vendored
    package's private ``_Actor``. That is a deliberate choice — see the module docstring —
    and it is only defensible while this test exists.
    """
    policy_dir = REPO / "policy"
    if not (policy_dir / "physical_ai_mappo.py").is_file():
        print("  skip  policy package not present")
        return
    sys.path.insert(0, str(policy_dir))
    try:
        from physical_ai_mappo import _Actor
    except ImportError as exc:
        print(f"  skip  policy package not importable ({exc})")
        return

    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "good.npz": _write(Path(tmp) / "good.npz"),
            "wide.npz": _write(Path(tmp) / "wide.npz", rays=24, obs=30),
        }
        junk = Path(tmp) / "junk.npz"
        np.savez(junk, W1=np.zeros((8, 8), dtype=np.float32))
        cases["junk.npz"] = junk
        if REAL_CHECKPOINT.is_file():
            cases["real.npz"] = REAL_CHECKPOINT

        for name, path in cases.items():
            mine = inspect_model(path).loadable
            try:
                _Actor(path)
                theirs = True
            except Exception:
                theirs = False
            assert mine == theirs, (
                f"{name}: model_store says loadable={mine}, physical_ai_mappo._Actor "
                f"says {theirs}. The duplicated shape rule has drifted.")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"model_store: {len(tests)}/{len(tests)} passed")
