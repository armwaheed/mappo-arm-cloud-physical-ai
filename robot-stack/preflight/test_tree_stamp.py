#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the deployed-tree stamp.

Two halves, and the first is the one that matters most. The tree id this module computes
is worthless unless it is *git's* tree id -- an id that is self-consistent but matches
nothing git ever wrote would verify every deploy and resolve to no commit, and it would
fail only months later when somebody tried to look one up. So the format is pinned by a
known-answer vector taken from a real ``git`` (recorded below with the commands that
produced it), and the vector is chosen to *discriminate*: a plain-name sort of its four
entries produces a different, stable, wrong id, which
``test_entries_sort_a_directory_as_if_it_ended_in_a_slash`` asserts.

The second half asks the question AGENTS.md insists on -- what would make this fail? -- of
the guard itself. A stamp check that only compares files to a manifest can be defeated by
editing the manifest, so there is a test that edits it, and it must still refuse.

Everything runs in a temporary directory. Nothing here needs git, a robot, or a network.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from tree_stamp import (
    MODE_EXEC,
    MODE_FILE,
    MODE_TREE,
    STAMP_NAME,
    Verdict,
    blob_id,
    build_stamp,
    describe,
    file_mode,
    hash_paths,
    parse_stamp,
    refusal_message,
    require_stamped_tree,
    root_tree_id,
    tree_id,
    verify,
    walk_tree,
)

# ---------------------------------------------------------------- known-answer vector
#
# Produced by a real git 2.39 in an empty repository, 2026-08-26:
#
#     printf 'a' > a.txt; printf 'b' > lib/b.txt; printf 'c' > lib-extra/c.txt
#     printf '#!/bin/sh\n' > run.sh; chmod +x run.sh; git add -A; git commit
#     git ls-tree -r HEAD           -> the four entries below
#     git rev-parse HEAD^{tree}     -> KAT_TREE
#     git rev-parse HEAD:lib        -> KAT_SUBTREE
#
# `lib-extra` and `lib` are here on purpose. Git orders a tree entry by its name with a
# `/` appended when the entry is a directory, so `lib-extra` sorts BEFORE `lib`
# ('-' is 0x2D, '/' is 0x2F) -- the reverse of what sorting the bare names gives.
KAT_MANIFEST = {
    "a.txt": (MODE_FILE, "2e65efe2a145dda7ee51d1741299f848e5bf752e"),
    "lib-extra/c.txt": (MODE_FILE, "3410062ba67c5ed59b854387a8bc0ec012479368"),
    "lib/b.txt": (MODE_FILE, "63d8dbd40c23542e740659a7168a0ce3138ea748"),
    "run.sh": (MODE_EXEC, "1a2485251c33a70432394c93fb89330ef214bfc9"),
}
KAT_TREE = "d1ccc4650aa630666af696378ada393d1e9da50a"
KAT_SUBTREE = "bc4d1181aca5a33673d7c5d4c209d09ce1cfabd7"
#: What a plain-name sort of KAT_MANIFEST produces. Recorded so the vector is known to
#: discriminate rather than merely to pass.
KAT_TREE_WRONG_ORDER = "98724d0f49bfaad463b79405cf87b19981b681bd"


# ------------------------------------------------------------------------- fixtures

class Deployment:
    """A throwaway directory with a stamp, and the levers to corrupt it."""

    def __init__(self, files):
        self.root = tempfile.mkdtemp(prefix="tree-stamp-test-")
        for path, data in files.items():
            self.write(path, data)
        self.restamp(sorted(files))

    def write(self, path, data):
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode())

    def restamp(self, paths, commit="0" * 40, ref="main"):
        manifest, _missing = hash_paths(self.root, paths)
        self.put_stamp(build_stamp(commit, ref, manifest))

    def put_stamp(self, stamp):
        with open(os.path.join(self.root, STAMP_NAME), "w") as handle:
            json.dump(stamp, handle)

    def stamp(self):
        with open(os.path.join(self.root, STAMP_NAME)) as handle:
            return json.load(handle)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


_TREE = {"integration/mappo_drive.py": "print('drive')\n",
         "policy/config.json": '{"scale": 2.5}\n',
         "README.md": "# repo\n"}


def _clean():
    return Deployment(dict(_TREE))


# --------------------------------------------------------------- the git object format

def test_blob_id_matches_git_hash_object():
    assert blob_id(b"a") == "2e65efe2a145dda7ee51d1741299f848e5bf752e"
    assert blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_root_tree_id_matches_a_real_git_tree():
    assert root_tree_id(KAT_MANIFEST) == KAT_TREE


def test_nested_subtree_matches_a_real_git_tree():
    """The recursion is checked at a subtree too, not only at the root.

    A root-only assertion passes for an implementation that hashes the flat path strings,
    which would agree with git on nothing deeper than one level.
    """
    assert root_tree_id({"b.txt": KAT_MANIFEST["lib/b.txt"]}) == KAT_SUBTREE


def test_entries_sort_a_directory_as_if_it_ended_in_a_slash():
    """The vector discriminates: the naive ordering yields a different, wrong id."""
    assert KAT_TREE != KAT_TREE_WRONG_ORDER
    entries = [(MODE_TREE, "lib", KAT_SUBTREE),
               (MODE_TREE, "lib-extra", "653c8359fc980eb3a393a41a1f1cbe4e8ce458f8"),
               (MODE_FILE, "a.txt", KAT_MANIFEST["a.txt"][1]),
               (MODE_EXEC, "run.sh", KAT_MANIFEST["run.sh"][1])]
    assert tree_id(entries) == KAT_TREE
    assert tree_id(list(reversed(entries))) == KAT_TREE


def test_tree_mode_has_no_leading_zero():
    """``git cat-file -p`` PRINTS ``040000``; it STORES ``40000``, and the id is over the
    stored bytes. Padding it produces an id git cannot resolve."""
    assert MODE_TREE == "40000"


def test_root_tree_id_refuses_an_empty_manifest():
    try:
        root_tree_id({})
    except ValueError:
        return
    raise AssertionError("an empty manifest produced a tree id")


def test_root_tree_id_refuses_a_path_that_is_both_file_and_directory():
    try:
        root_tree_id({"a": (MODE_FILE, blob_id(b"x")), "a/b": (MODE_FILE, blob_id(b"y"))})
    except ValueError:
        return
    raise AssertionError("a path that is both a file and a directory was accepted")


# ------------------------------------------------------------------ reading from disk

def test_file_mode_reports_one_executable_bit():
    deployment = _clean()
    try:
        plain = os.path.join(deployment.root, "README.md")
        assert file_mode(plain) == MODE_FILE
        os.chmod(plain, 0o755)
        assert file_mode(plain) == MODE_EXEC
    finally:
        deployment.close()


def test_walk_tree_prunes_pycache_and_git():
    deployment = _clean()
    try:
        deployment.write("integration/__pycache__/mappo_drive.cpython-38.pyc", b"\x00")
        deployment.write(".git/HEAD", "ref: refs/heads/main\n")
        found = walk_tree(deployment.root)
        assert not [p for p in found if "__pycache__" in p or p.startswith(".git/")]
        assert "integration/mappo_drive.py" in found
    finally:
        deployment.close()


def test_hash_paths_separates_missing_from_changed():
    deployment = _clean()
    try:
        os.remove(os.path.join(deployment.root, "README.md"))
        manifest, missing = hash_paths(deployment.root, sorted(_TREE))
        assert missing == ["README.md"]
        assert "README.md" not in manifest
    finally:
        deployment.close()


# ------------------------------------------------------------------------ stamp shape

def test_build_stamp_derives_the_tree_rather_than_accepting_one():
    """There is no parameter by which a caller can assert a tree id it did not compute."""
    stamp = build_stamp("c" * 40, "main", KAT_MANIFEST)
    assert stamp["tree"] == KAT_TREE
    assert stamp["file_count"] == 4
    assert stamp["files"]["run.sh"] == f"{MODE_EXEC} {KAT_MANIFEST['run.sh'][1]}"


def test_parse_stamp_rejects_text_that_is_not_json():
    for text in ("", "not json", "[]"):
        try:
            parse_stamp(text)
        except ValueError:
            continue
        raise AssertionError(f"{text!r} parsed as a stamp")


def test_parse_stamp_rejects_a_stamp_that_lists_no_files():
    try:
        parse_stamp(json.dumps({"commit": "a" * 40, "tree": KAT_TREE, "files": {}}))
    except ValueError:
        return
    raise AssertionError("a stamp with no files parsed")


def test_parse_stamp_rejects_a_malformed_entry():
    try:
        parse_stamp(json.dumps({"commit": "a" * 40, "tree": KAT_TREE,
                                "files": {"a.txt": "no-mode"}}))
    except ValueError:
        return
    raise AssertionError("an entry without a mode parsed")


# ---------------------------------------------------------------------------- verify

def test_verify_accepts_a_clean_deploy():
    deployment = _clean()
    try:
        verdict = verify(deployment.root)
        assert verdict.ok, verdict.reason
        assert verdict.tree == verdict.expected_tree
    finally:
        deployment.close()


def test_verify_refuses_a_modified_file_and_names_it():
    deployment = _clean()
    try:
        deployment.write("integration/mappo_drive.py", "print('drive'); import os\n")
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.changed == ["integration/mappo_drive.py"]
        message = refusal_message("mappo_drive", verdict)
        assert "integration/mappo_drive.py" in message
        # The tree the disk ACTUALLY holds is reported, not withheld. It is what an
        # operator looks up, and the first version of this said "files missing" here --
        # on a tree with nothing missing.
        assert verdict.tree is not None and verdict.tree != verdict.expected_tree
        assert verdict.tree in message and "missing" not in message
    finally:
        deployment.close()


def test_verify_refuses_a_file_that_only_changed_MODE():
    """chmod +x edits no byte of content and still changes the commit's tree."""
    deployment = _clean()
    try:
        os.chmod(os.path.join(deployment.root, "integration/mappo_drive.py"), 0o755)
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.changed == ["integration/mappo_drive.py"]
    finally:
        deployment.close()


def test_verify_refuses_a_missing_file():
    deployment = _clean()
    try:
        os.remove(os.path.join(deployment.root, "policy/config.json"))
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.missing == ["policy/config.json"]
        assert verdict.tree is None
    finally:
        deployment.close()


def test_verify_refuses_an_unlisted_py_because_it_can_shadow_an_import():
    deployment = _clean()
    try:
        deployment.write("integration/visual_nav.py", "raise SystemExit('shadowed')\n")
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.extra_py == ["integration/visual_nav.py"]
    finally:
        deployment.close()


def test_verify_tolerates_run_output_and_counts_it():
    """A .jsonl and a .mp4 are what a run PRODUCES. Refusing on those makes the guard
    something people delete, and the deployed trees are full of them."""
    deployment = _clean()
    try:
        deployment.write("smoke.jsonl", '{"tick": 0}\n')
        deployment.write("integration/mappo_drive.py.bak-preberth", "old\n")
        verdict = verify(deployment.root)
        assert verdict.ok, verdict.reason
        assert verdict.extra_other == 2
        assert "2 unlisted non-.py" in describe(verdict)
    finally:
        deployment.close()


def test_verify_refuses_a_stamp_EDITED_to_match_a_modified_file():
    """The attack the per-file comparison alone does not stop.

    Change a deployed file, then rewrite that file's entry in the stamp so the manifest
    agrees with the disk again. Every per-file check now passes. The run must still
    refuse, because the tree id in the stamp is derived from the ORIGINAL bytes and the
    one recomputed from disk is not.
    """
    deployment = _clean()
    try:
        deployment.write("integration/mappo_drive.py", "print('drive'); DRIFT = 1\n")
        stamp = deployment.stamp()
        manifest, _missing = hash_paths(deployment.root, sorted(_TREE))
        mode, blob = manifest["integration/mappo_drive.py"]
        stamp["files"]["integration/mappo_drive.py"] = f"{mode} {blob}"
        deployment.put_stamp(stamp)

        verdict = verify(deployment.root)
        assert verdict.changed == [], "the per-file check was expected to be satisfied"
        assert verdict.missing == []
        assert not verdict.ok, "editing the manifest defeated the guard"
        assert verdict.reason == "tree id does not match the stamp"
        assert verdict.tree != verdict.expected_tree
    finally:
        deployment.close()


def test_verify_refuses_when_only_the_stamp_TREE_was_edited():
    """The mirror of the test above: content untouched, claim rewritten."""
    deployment = _clean()
    try:
        stamp = deployment.stamp()
        stamp["tree"] = "0" * 40
        deployment.put_stamp(stamp)
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.reason == "tree id does not match the stamp"
    finally:
        deployment.close()


def test_verify_reports_an_unstamped_tree():
    deployment = _clean()
    try:
        os.remove(os.path.join(deployment.root, STAMP_NAME))
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.reason == "unstamped"
    finally:
        deployment.close()


def test_verify_reports_an_unreadable_stamp_rather_than_crashing():
    deployment = _clean()
    try:
        deployment.write(STAMP_NAME, "{ truncated")
        verdict = verify(deployment.root)
        assert not verdict.ok
        assert verdict.reason.startswith("unreadable stamp")
    finally:
        deployment.close()


# ------------------------------------------------------------------------- the refusal

def test_require_stamped_tree_returns_the_verdict_on_a_clean_tree():
    deployment = _clean()
    try:
        assert require_stamped_tree("mappo_drive", deployment.root, on_robot=True).ok
    finally:
        deployment.close()


def test_require_stamped_tree_refuses_a_mismatch_even_off_a_robot():
    """A mismatch is a finding about a tree somebody deployed, wherever it is read."""
    deployment = _clean()
    try:
        deployment.write("README.md", "# edited\n")
        try:
            require_stamped_tree("mappo_drive", deployment.root, on_robot=False)
        except SystemExit as exc:
            assert "REFUSING" in str(exc)
            return
        raise AssertionError("a modified deployed tree was allowed to run")
    finally:
        deployment.close()


def test_require_stamped_tree_allows_an_unstamped_CHECKOUT():
    """Otherwise the guard fires on every developer clone and every CI job, and a gate
    that always fires is a gate somebody removes."""
    deployment = _clean()
    try:
        os.remove(os.path.join(deployment.root, STAMP_NAME))
        verdict = require_stamped_tree("mappo_drive", deployment.root, on_robot=False)
        assert not verdict.ok and verdict.reason == "unstamped"
    finally:
        deployment.close()


def test_require_stamped_tree_refuses_an_unstamped_tree_ON_A_ROBOT():
    deployment = _clean()
    try:
        os.remove(os.path.join(deployment.root, STAMP_NAME))
        try:
            require_stamped_tree("mappo_drive", deployment.root, on_robot=True)
        except SystemExit as exc:
            assert "cannot say which commit" in str(exc)
            return
        raise AssertionError("an unstamped tree ran on a robot")
    finally:
        deployment.close()


def test_refusal_message_names_the_files_and_not_only_a_count():
    verdict = Verdict(False, "/srv/tree", "1 modified, 0 missing",
                      expected_tree="a" * 40, tree="b" * 40, commit="c" * 40,
                      changed=["integration/mappo_drive.py"])
    message = refusal_message("mappo_drive", verdict)
    assert "integration/mappo_drive.py" in message
    assert "a" * 40 in message and "b" * 40 in message
    assert "push-to-robot.sh" in message


def test_describe_names_the_tree_id_even_when_it_refuses():
    verdict = Verdict(False, "/srv/tree", "1 modified, 0 missing", expected_tree="a" * 40,
                      tree="b" * 40, commit="c" * 40, ref="main")
    line = describe(verdict)
    assert "REFUSED" in line and "b" * 12 in line and "c" * 12 in line


def test_describe_says_so_when_the_deploy_was_dirty():
    verdict = Verdict(True, "/srv/tree", "clean", expected_tree="a" * 40, tree="a" * 40,
                      commit="c" * 40, ref="main", dirty=True)
    assert "deployed dirty" in describe(verdict)


# ⚠️ Keep this at the END of the file. A `__main__` block placed mid-file stops every test
# below it from being collected, which cost this repository ten tests across two files.
if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"tree_stamp: {len(tests)}/{len(tests)} passed")
