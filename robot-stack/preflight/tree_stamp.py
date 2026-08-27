#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Make a deployed tree name its own commit, and refuse to run when it cannot.

None of the trees on the lab Go2 is a git checkout -- no ``.git``, so no branch, no
commit, and no ``git status``. AGENTS.md has said "a live run does not tell you which code
produced it" since 2026-08-26; this module is that sentence turned into a check.

The identity it records is **git's own root tree object id, recomputed from the bytes on
disk**. That choice is the whole design, and it is worth being explicit about what it buys
and what it does not.

## Why a tree id rather than a commit string in a file

Writing ``commit=1cb2d69`` into a text file at deploy time produces a value that is true
for exactly as long as nobody touches the tree. The first ``vi`` on a ``.py`` at 11pm
makes it a lie, and nothing anywhere notices -- which is how a robot comes to have ten
sibling directories named after guesses. A stamp has to be *derived from the content it
describes*, so that changing the content changes the answer.

Git already has such a function, it is already the id this repository indexes by, and it
needs no git to compute::

    blob id = sha1(b"blob %d\\x00" % len(bytes) + bytes)
    tree id = sha1(b"tree %d\\x00" % len(body) + body)
       body = concat over entries sorted by name, directories sorted as name + "/", of
              b"%s %s\\x00" % (mode, name) + 20 raw bytes of the child's id

So :func:`root_tree_id` reads the deployed files and returns a 40-hex string that any
clone can resolve: ``git rev-parse <commit>^{tree}``, or ``git log --format='%T %H'``.
Two consequences follow, and only the first is a security property:

* **Change a byte in any deployed file and the root tree id changes.** There is no edit to
  a *content* field that makes an edited file pass. This closes the failure that actually
  happens -- accidental drift, a debug print left in, a hand-patched constant -- completely,
  on the robot, with no git and no network.
* **Someone who edits the stamp too can still lie.** They can recompute the tree id with
  this very file and paste it in. What they cannot do is make the pair *(commit, tree)*
  agree, because ``stamp["tree"]`` is claimed to be the tree OF ``stamp["commit"]`` and
  that link lives in the repository, not on the robot. ``tree_stamp.py audit`` checks it in
  one command from any clone. **This is not tamper-proofing and must not be sold as such.**
  Tamper-proofing needs a key, and a key on a robot that every agent can read is a key
  every agent can sign with. What this gets you is that a false claim is *falsifiable from
  the run log*, which is the property the published measurements actually need.

## What the run prints

The guard prints the tree id on every run, refusing or not, for the reason ``venv_guard``
prints its verdict: a check that is quiet when it passes is indistinguishable from a check
that is not running. A telemetry line carrying ``287242e...`` is evidence somebody else can
resolve six months later; "deployed from main" is not.

## Extra files are two different questions

A deployed tree accumulates output -- ``.jsonl``, ``.mp4``, ``__pycache__`` -- and refusing
on those would make the guard something people route around within a day. But an extra
``.py`` is not output: it is on ``sys.path`` and it can shadow a module the run imports.
So extras are split. An unlisted ``.py`` refuses; anything else is counted and named in the
verdict line and does not. The Go2's ``~/mappo-main`` held
``integration/mappo_drive.py.bak-preberth`` and
``robot-stack/unitree/go2/visual_nav/visual_nav.py.orig`` on 2026-08-26 -- neither is
importable, both are somebody's hand-edit, and a check that says so out loud without
blocking a run is the right treatment for them.

Python 3.8, standard library only, no imports from this repository -- same constraint as
``venv_guard``: it has to be importable by the interpreter it is about to refuse.

``python3 test_tree_stamp.py``. Run ``python3 tree_stamp.py verify <dir>`` to check a tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
import sys

#: Written at the root of a deployed tree. Dot-prefixed so it sorts with the other
#: metadata and is not mistaken for something the run reads.
STAMP_NAME = ".mappo-stamp.json"

#: Git's file modes. Only ``100644`` and ``100755`` have ever appeared in this repository
#: (measured across all 160 commits on 2026-08-26), but a symlink is one ``ln -s`` away and
#: hashing it as a regular file would silently produce the wrong tree id, so it is handled.
MODE_FILE = "100644"
MODE_EXEC = "100755"
MODE_LINK = "120000"
#: Git writes a tree's mode WITHOUT the leading zero -- ``40000``, not ``040000``. Padding
#: it changes the hashed bytes and therefore the id, and nothing else in the format catches
#: the error.
MODE_TREE = "40000"

#: Never part of the identity: git ignores it and it is regenerated by whichever
#: interpreter ran last, so including it would make the tree id depend on whether the run
#: has happened yet.
PRUNED_DIRS = ("__pycache__", ".git")

_PREFIX = "[tree-stamp]"


def blob_id(data: bytes) -> str:
    """The git object id of ``data`` as a blob. ``git hash-object -t blob`` with no git."""
    # SHA-1 because git's object id is SHA-1. This is an identity that must equal
    # git's, not a security digest, and swapping it for SHA-256 would make every id
    # here resolve to nothing.
    digest = hashlib.sha1()
    digest.update(b"blob %d\x00" % len(data))
    digest.update(data)
    return digest.hexdigest()


def _entry_sort_key(entry):
    """Git's tree ordering: by name, with a directory compared as if it ended in ``/``.

    This is the one rule in the format that is not obvious, and getting it wrong produces
    a tree id that is stable, self-consistent, and matches nothing git ever wrote -- so it
    fails only when someone tries to resolve the id, which is long after the deploy.
    """
    mode, name, _ = entry
    return (name + "/").encode() if mode == MODE_TREE else name.encode()


def tree_id(entries) -> str:
    """The git object id of one tree. ``entries`` is ``(mode, name, hex_id)`` triples."""
    body = b"".join(
        f"{mode} {name}".encode() + b"\x00" + bytes.fromhex(child)
        for mode, name, child in sorted(entries, key=_entry_sort_key))
    digest = hashlib.sha1()  # SHA-1: see blob_id
    digest.update(b"tree %d\x00" % len(body))
    digest.update(body)
    return digest.hexdigest()


def root_tree_id(manifest) -> str:
    """The git root tree id for ``{posix path: (mode, blob id)}``.

    Rebuilds the directory nesting that ``git ls-tree -r`` flattened, then hashes bottom
    up. An empty manifest is an error rather than the id of the empty tree: every caller
    here got its manifest from a real tree, so an empty one means the scan found nothing,
    and returning a valid-looking id for that would let a deploy of zero files verify.
    """
    if not manifest:
        raise ValueError("empty manifest: nothing to identify")
    root: dict = {}
    for path, (mode, blob) in manifest.items():
        parts = path.split("/")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{path!r}: {part!r} is both a file and a directory")
        if parts[-1] in node and isinstance(node[parts[-1]], dict):
            raise ValueError(f"{path!r} is both a file and a directory")
        node[parts[-1]] = (mode, blob)
    return _hash_dir(root)


def _hash_dir(node) -> str:
    entries = []
    for name, value in node.items():
        if isinstance(value, dict):
            entries.append((MODE_TREE, name, _hash_dir(value)))
        else:
            entries.append((value[0], name, value[1]))
    return tree_id(entries)


def file_mode(path, lstat=None) -> str:
    """Git's mode for a path on disk. Git records one executable bit, not nine."""
    lstat = os.lstat if lstat is None else lstat
    st = lstat(path)
    if stat_module.S_ISLNK(st.st_mode):
        return MODE_LINK
    return MODE_EXEC if st.st_mode & 0o111 else MODE_FILE


def _read_path(path):
    """Bytes of a path. A symlink hashes as its target text, which is what git stores."""
    if os.path.islink(path):
        return os.readlink(path).encode()
    with open(path, "rb") as handle:
        return handle.read()


def hash_paths(root, paths, read_path=None, mode_of=None):
    """``{path: (mode, blob id)}`` for ``paths`` under ``root``. Missing paths are skipped.

    Returns the manifest and the sorted list of paths that were not there, because
    "changed" and "deleted" are different findings and a caller that conflates them
    reports a missing file as a modified one.
    """
    read_path = _read_path if read_path is None else read_path
    mode_of = file_mode if mode_of is None else mode_of
    manifest = {}
    missing = []
    for path in paths:
        full = os.path.join(root, path.replace("/", os.sep))
        try:
            data = read_path(full)
            manifest[path] = (mode_of(full), blob_id(data))
        except OSError:
            missing.append(path)
    return manifest, sorted(missing)


def walk_tree(root, listdir=None, isdir=None):
    """Every file under ``root`` as posix-relative paths, pruning ``__pycache__``/``.git``.

    Its own ``os.walk``: the deployed trees are the only thing this ever reads and the
    prune list has to be applied to directory names before descending, not filtered out of
    the results afterwards -- a ``__pycache__`` under a venv is thousands of paths.
    """
    listdir = os.listdir if listdir is None else listdir
    isdir = os.path.isdir if isdir is None else isdir
    found = []
    stack = [""]
    while stack:
        rel = stack.pop()
        base = os.path.join(root, rel) if rel else root
        try:
            names = sorted(listdir(base))
        except OSError:
            continue
        for name in names:
            child = f"{rel}/{name}" if rel else name
            if isdir(os.path.join(base, name)) and not os.path.islink(
                    os.path.join(base, name)):
                if name not in PRUNED_DIRS:
                    stack.append(child)
            else:
                found.append(child)
    return sorted(found)


def build_stamp(commit, ref, manifest, deployed_at=None, deployed_by=None, source=None,
                dirty=False):
    """The stamp document. ``tree`` is derived here and never passed in.

    ``dirty`` records that the deploy came from a working tree with uncommitted changes.
    It is a flag on an honest stamp, not a refusal: refusing here would only teach people
    to commit noise, and the tree id still describes exactly what was copied -- it just
    will not resolve to anything in the repository, which is precisely the fact worth
    recording.
    """
    return {
        "commit": commit,
        "ref": ref,
        "tree": root_tree_id(manifest),
        "dirty": bool(dirty),
        "deployed_at": deployed_at,
        "deployed_by": deployed_by,
        "source": source,
        "file_count": len(manifest),
        "files": {path: f"{mode} {blob}" for path, (mode, blob) in sorted(manifest.items())},
    }


def parse_stamp(text):
    """Load a stamp, raising ``ValueError`` on anything a verifier cannot use.

    Checked rather than trusted because the stamp is the one input that arrives from
    outside: a truncated ``scp`` produces valid JSON surprisingly often, and a stamp
    missing ``files`` would otherwise verify an empty manifest against itself.
    """
    try:
        stamp = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"stamp is not valid JSON: {exc}") from exc
    if not isinstance(stamp, dict):
        raise ValueError("stamp is not a JSON object")
    for key in ("commit", "tree", "files"):
        if key not in stamp:
            raise ValueError(f"stamp has no {key!r}")
    if not isinstance(stamp["files"], dict) or not stamp["files"]:
        raise ValueError("stamp lists no files")
    manifest = {}
    for path, spec in stamp["files"].items():
        parts = str(spec).split()
        if len(parts) != 2:
            raise ValueError(f"stamp entry for {path!r} is not '<mode> <blob>'")
        manifest[path] = (parts[0], parts[1])
    stamp["manifest"] = manifest
    return stamp


class Verdict:
    """The finding, separated from the act of refusing so both halves are testable.

    Same split as ``venv_guard.Decision`` and for the same reason: a caller that wants to
    log the tree id without dying uses :func:`verify`, and only
    :func:`require_stamped_tree` raises.
    """

    __slots__ = ("changed", "commit", "dirty", "expected_tree", "extra_other", "extra_py",
                 "missing", "ok", "reason", "ref", "root", "tree")

    def __init__(self, ok, root, reason, expected_tree=None, tree=None, commit=None,
                 ref=None, dirty=False, changed=(), missing=(), extra_py=(),
                 extra_other=0):
        self.ok = ok
        self.root = root
        self.reason = reason
        self.expected_tree = expected_tree
        self.tree = tree
        self.commit = commit
        self.ref = ref
        self.dirty = dirty
        self.changed = list(changed)
        self.missing = list(missing)
        self.extra_py = list(extra_py)
        self.extra_other = extra_other

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Verdict(ok={self.ok!r}, tree={self.tree!r}, reason={self.reason!r})"


def verify(root, read_stamp=None, hash_fn=None, walk=None):
    """Check the tree at ``root`` against its own stamp. Never raises; returns a Verdict.

    Three independent findings, reported together rather than short-circuiting, because an
    operator who fixes the first of three and re-runs learns about the second one deploy
    later:

    * every stamped path is present and hashes to the recorded blob,
    * the root tree id recomputed from disk equals the one the stamp claims,
    * no unlisted ``.py`` sits anywhere under the tree.

    The second is not redundant with the first. The per-file pass compares the manifest to
    disk; the tree id compares the manifest to the *commit*, and a stamp whose ``files``
    were edited to match an edited tree passes the first and fails the second.
    """
    read_stamp = _read_text if read_stamp is None else read_stamp
    hash_fn = hash_paths if hash_fn is None else hash_fn
    walk = walk_tree if walk is None else walk

    text = read_stamp(os.path.join(root, STAMP_NAME))
    if text is None:
        return Verdict(False, root, "unstamped")
    try:
        stamp = parse_stamp(text)
    except ValueError as exc:
        return Verdict(False, root, f"unreadable stamp: {exc}")

    expected = stamp["manifest"]
    found, missing = hash_fn(root, sorted(expected))
    changed = sorted(path for path, spec in found.items() if spec != expected[path])

    on_disk = set(walk(root))
    unlisted = on_disk - set(expected) - {STAMP_NAME}
    extra_py = sorted(p for p in unlisted if p.endswith(".py"))

    # Computed whenever every stamped path is PRESENT, modified or not. A tree whose
    # files were edited still has a tree id, and that id is the most useful thing an
    # operator can be handed: it is what the run would have been, and it is what to look
    # up. Only a missing file makes it uncomputable, because the manifest then describes
    # a tree that does not exist on this disk.
    actual = root_tree_id(found) if not missing else None
    tree_matches = actual == stamp["tree"]

    ok = not missing and not changed and not extra_py and tree_matches
    if ok:
        reason = "clean"
    elif missing or changed:
        reason = f"{len(changed)} modified, {len(missing)} missing"
    elif not tree_matches:
        reason = "tree id does not match the stamp"
    else:
        reason = f"{len(extra_py)} unlisted .py"
    return Verdict(ok, root, reason, expected_tree=stamp["tree"], tree=actual,
                   commit=stamp.get("commit"), ref=stamp.get("ref"),
                   dirty=bool(stamp.get("dirty")), changed=changed, missing=missing,
                   extra_py=extra_py, extra_other=len(unlisted) - len(extra_py))


def _read_text(path):
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8")
    except OSError:
        return None
    except UnicodeDecodeError as exc:
        raise ValueError(f"stamp is not UTF-8: {exc}") from exc


def describe(verdict) -> str:
    """One line, printed on every run whether it refuses or not.

    Names the tree id even when the verdict is bad, because that id is the only thing that
    lets somebody reading the log afterwards find out what actually ran.
    """
    if verdict.reason == "unstamped":
        return (f"{_PREFIX} {verdict.root}: no stamp -- this tree cannot name its commit. "
                f"On a robot that refuses; in a checkout, ask git instead.")
    short = (verdict.commit or "?")[:12]
    tree = (verdict.tree or verdict.expected_tree or "?")[:12]
    tag = "clean" if verdict.ok else f"REFUSED ({verdict.reason})"
    dirty = ", deployed dirty" if verdict.dirty else ""
    extra = f", {verdict.extra_other} unlisted non-.py" if verdict.extra_other else ""
    return (f"{_PREFIX} {tag}: commit {short} ({verdict.ref or 'no ref'}), "
            f"tree {tree}{dirty}{extra}")


def refusal_message(component, verdict) -> str:
    """What the operator sees instead of a run. Names files, not a count."""
    lines = [f"{_PREFIX} {component}: REFUSING -- {verdict.root} does not match its stamp.",
             ""]
    if verdict.reason == "unstamped":
        lines += [f"There is no {STAMP_NAME} here, so this tree cannot say which commit it",
                  "is. That is the state every deployed tree on the lab Go2 was in on",
                  "2026-08-26, and it is why no run before this one can be attributed to a",
                  "commit.", ""]
    else:
        disk = verdict.tree or f"(uncomputable: {len(verdict.missing)} stamped file(s) absent)"
        lines += [f"stamp says   commit {verdict.commit}  tree {verdict.expected_tree}",
                  f"disk gives   tree {disk}",
                  ""]
        for label, paths in (("modified", verdict.changed), ("missing", verdict.missing),
                             ("unlisted .py", verdict.extra_py)):
            if paths:
                lines.append(f"{len(paths)} {label}:")
                lines += [f"    {p}" for p in paths[:20]]
                if len(paths) > 20:
                    lines.append(f"    ... and {len(paths) - 20} more")
                lines.append("")
    lines += ["Re-deploy rather than editing here:",
              "    bash deploy/push-to-robot.sh <host> <dest>",
              "",
              "Editing on the robot is what produced ten sibling trees named after",
              "guesses. If the edit is worth keeping, it is worth a commit."]
    return "\n".join(lines)


def require_stamped_tree(component, root, on_robot=False, verdict=None) -> Verdict:
    """Raise ``SystemExit`` unless ``root`` may be run. Returns the Verdict when it may.

    ``on_robot`` is the one thing that makes this enforceable rather than merely annoying,
    and it exists because the obvious rule -- "refuse an unstamped tree" -- is a gate that
    fires on every developer clone and every CI job, and a gate that always fires gets
    commented out within a week. A clone is not a deployment. So:

    * **stamped and matching** -> run, and print the tree id.
    * **stamped and NOT matching** -> refuse, anywhere, including a laptop. Somebody
      deployed this tree and then changed it; that is the finding regardless of host.
    * **no stamp, not a robot** -> run. This is a checkout, and the git it is sitting in
      answers the question better than any stamp could.
    * **no stamp, on a robot** -> refuse. That is exactly the ungoverned state every tree
      on the lab Go2 was in on 2026-08-26, and it is the state this module exists to end.

    The caller supplies ``on_robot`` rather than this module deciding it, so the two
    guards keep one definition of "robot" between them instead of two that can drift --
    ``venv_guard.robot_host_evidence`` is that definition, and it is positive-only.

    ``verdict`` is injected by the tests so the refusal text can be asserted on without a
    filesystem. There is deliberately no environment variable that disarms this, for the
    reason ``venv_guard`` gives at length: an escape hatch documented beside a refusal is
    an instruction to use it, and there is always a correct action here -- re-deploy.
    """
    verdict = verify(root) if verdict is None else verdict
    if verdict.ok:
        return verdict
    if verdict.reason == "unstamped" and not on_robot:
        return verdict
    raise SystemExit(refusal_message(component, verdict))


# ------------------------------------------------------------------------------ CLI


def _git(args, cwd):
    import subprocess
    done = subprocess.run(["git", "-C", cwd, *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{_PREFIX} git {' '.join(args)} failed: "
                         f"{done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout.decode("utf-8")


def _cmd_stamp(argv):
    """Write a stamp for a git checkout. Runs where git is -- never on the robot."""
    root = os.path.abspath(argv[0] if argv else ".")
    commit = _git(["rev-parse", "HEAD"], root).strip()
    ref = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip()
    dirty = bool(_git(["status", "--porcelain"], root).strip())
    manifest = {}
    for line in _git(["ls-tree", "-r", commit], root).splitlines():
        meta, path = line.split("\t", 1)
        mode, _kind, blob = meta.split()
        manifest[path] = (mode, blob)
    import datetime
    stamp = build_stamp(
        commit, ref, manifest,
        deployed_at=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        deployed_by=os.environ.get("USER") or "unknown", source=root, dirty=dirty)
    expected = _git(["rev-parse", f"{commit}^{{tree}}"], root).strip()
    if stamp["tree"] != expected:
        raise SystemExit(f"{_PREFIX} recomputed tree {stamp['tree']} != git's {expected}")
    sys.stdout.write(json.dumps(stamp, indent=1, sort_keys=True) + "\n")
    return 0


def _cmd_verify(argv):
    root = os.path.abspath(argv[0] if argv else ".")
    verdict = verify(root)
    print(describe(verdict))
    if not verdict.ok:
        print("", file=sys.stderr)
        print(refusal_message("tree_stamp verify", verdict), file=sys.stderr)
        return 1
    return 0


def _cmd_audit(argv):
    """Close the loop the robot cannot: does ``stamp.tree`` really belong to ``stamp.commit``?

    This is the half that makes a hand-edited stamp falsifiable, and it needs the
    repository, so it runs in a clone and never on the robot.
    """
    if len(argv) < 2:
        raise SystemExit("usage: tree_stamp.py audit <stamp.json> <clone>")
    stamp = parse_stamp(_read_text(argv[0]) or "")
    clone = os.path.abspath(argv[1])
    tree = _git(["rev-parse", f"{stamp['commit']}^{{tree}}"], clone).strip()
    print(f"stamp commit {stamp['commit']}")
    print(f"stamp tree   {stamp['tree']}")
    print(f"repo  tree   {tree}")
    if tree != stamp["tree"]:
        print("MISMATCH: this stamp's tree is not the tree of the commit it names.")
        return 1
    print("OK: the commit and the tree agree, so the claim is the repository's, not the "
          "stamp author's.")
    return 0


def _cmd_id(argv):
    """The root tree id of an arbitrary directory, stamp or no stamp -- the forensic mode."""
    root = os.path.abspath(argv[0] if argv else ".")
    paths = [p for p in walk_tree(root) if p != STAMP_NAME]
    manifest, _missing = hash_paths(root, paths)
    print(root_tree_id(manifest))
    return 0


_COMMANDS = {"stamp": _cmd_stamp, "verify": _cmd_verify, "audit": _cmd_audit, "id": _cmd_id}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        print(f"usage: tree_stamp.py {{{'|'.join(sorted(_COMMANDS))}}} [args]\n\n"
              "  stamp <clone>            write a stamp for a git checkout (needs git)\n"
              "  verify <dir>             check a deployed tree against its stamp (no git)\n"
              "  audit <stamp> <clone>    check the stamp's commit and tree agree\n"
              "  id <dir>                 root tree id of any directory, stamp or not",
              file=sys.stderr)
        return 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
