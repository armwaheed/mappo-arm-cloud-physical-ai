#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The checkpoints on this robot: what is here, which one is armed, and what may replace it.

A "model swap" in this demo is one field in ``policy/config.json`` — ``model_path``. That is
the whole mechanism, and it is worth being precise about what it does and does not do,
because the obvious reading is wrong in a useful way.

**A swap takes effect on the NEXT run, never on the one in progress.**
``MappoController.__init__`` reads the config and loads the weights once, at construction.
A live ``mappo_drive`` process has its actor in memory and will not re-read the file. So
this store cannot yank the network out from under a robot that is walking — not because it
checks, but because there is no code path by which it could. That is a safety property
worth stating rather than a limitation to apologise for, and it is why the dashboard says
"armed for the next run" instead of "loaded".

**What this store must therefore catch is a swap that breaks the next run.** Two failures
are silent at swap time and fatal at load time, several minutes later, with the operator
already standing in the arena:

  * ``lidar_range_vmas``. ``MappoController._check_against_checkpoint`` refuses to run if
    the config's value disagrees with the checkpoint's ``training_lidar_range_vmas``, and
    it is right to: the observation's proximity convention is measured against that range,
    so a mismatch hands the network a lidar vector on a scale it never trained on. Every
    number stays finite; the robot just steers wrongly. :meth:`select` runs the same
    comparison BEFORE writing the config, so the answer arrives while someone is looking at
    a screen instead of at a robot.
  * The observation width and ray count. A checkpoint built for a different fan — which is
    exactly what issue #29 asks Sagar for, a 24-ray retrain — cannot be driven by this
    adapter at all. :func:`inspect_model` reports it and :meth:`select` refuses it.

**The shape check here is deliberately a second implementation, not a call into the policy
package.** ``policy/`` is a vendored deliverable (see ``policy/PROVENANCE.md``); reaching
into its private ``_Actor`` would couple the dashboard to a name its author never published.
So the contract is re-stated here, and :func:`test_model_store` pins the two together by
feeding the same files to both and asserting they agree — a duplicated rule that is checked
against its original is a different thing from a duplicated rule that drifts.

Pure stdlib + numpy. ``python3 test_model_store.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The delivered network: 18 -> 256 -> 256 -> 4, tanh throughout. Stated in
#: ``policy/PROVENANCE.md`` and enforced by the policy package's actor loader.
EXPECTED_SHAPES = {
    "W1": (256, 18),
    "b1": (256,),
    "W2": (256, 256),
    "b2": (256,),
    "W3": (4, 256),
    "b3": (4,),
}

#: The observation this adapter builds, and the fan it casts. Both come from
#: ``policy/physical_ai_mappo.py`` (``OBS_DIM``, ``N_RAYS``) and both are properties of the
#: ADAPTER, so a checkpoint that wants different ones cannot be driven by this code.
OBS_DIM = 18
N_RAYS = 12

#: A checkpoint file is an ``.npz``. Anything else is not refused for being unsafe — it is
#: refused for not being the thing.
MODEL_SUFFIX = ".npz"

#: Model names are filenames in one flat directory, never paths. A name that survives this
#: cannot traverse out of the models directory, cannot be absolute, and cannot be a dotfile.
#: The regex is the whole defence and it is deliberately narrow: a checkpoint arriving from
#: an S3 key or an HTTP URL is attacker-influenced input in the ordinary sense, and
#: ``models/`` sits next to a config file this store writes.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ModelStoreError(RuntimeError):
    """A request that would leave the robot in a state the next run cannot start from."""


def safe_model_name(name: str) -> str:
    """Return ``name`` if it is a legal checkpoint filename, else raise.

    Rejects anything with a separator, a parent reference, a leading dot, or a suffix other
    than ``.npz``. ``Path(name).name`` is NOT used to sanitise: silently rewriting
    ``../../x.npz`` to ``x.npz`` turns an attack into a surprise, and a caller that meant a
    path deserves to be told it cannot have one.
    """
    if not isinstance(name, str) or not _SAFE_NAME.match(name):
        raise ModelStoreError(
            f"{name!r} is not a legal checkpoint name: use letters, digits, dot, dash and "
            f"underscore only, starting with a letter or digit, and no directory separators"
        )
    if not name.endswith(MODEL_SUFFIX):
        raise ModelStoreError(f"{name!r} must end in {MODEL_SUFFIX}")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ModelReport:
    """What a file on disk is, and whether this adapter could drive it."""

    name: str
    path: Path
    size_bytes: int
    sha256: str
    loadable: bool
    #: Empty when :attr:`loadable`. Each entry is a complete sentence naming the field that
    #: disagrees and what it disagrees with, because these are read by an operator and not
    #: by a parser.
    problems: list = field(default_factory=list)
    #: The checkpoint's own ``metadata_json``, or ``{}`` for one that predates the field.
    metadata: dict = field(default_factory=dict)

    @property
    def trained_lidar_range_vmas(self):
        return self.metadata.get("training_lidar_range_vmas")

    @property
    def trained_agent_radius_vmas(self):
        return self.metadata.get("training_agent_radius_vmas")

    @property
    def rays(self):
        """The number of lidar features the checkpoint's layout declares, or ``None``."""
        layout = self.metadata.get("observation_layout")
        if layout is None:
            return None
        return sum(1 for term in layout if str(term).startswith("lidar"))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "loadable": self.loadable,
            "problems": list(self.problems),
            "trained_lidar_range_vmas": self.trained_lidar_range_vmas,
            "trained_agent_radius_vmas": self.trained_agent_radius_vmas,
            "rays": self.rays,
            "training_frames": self.metadata.get("training_frames"),
            "training_n_agents": self.metadata.get("training_n_agents"),
        }


def _metadata_problems(metadata: dict) -> list:
    """What the checkpoint says about itself that this ADAPTER cannot satisfy.

    Split out of :func:`inspect_model` so that the array-shape reading and the
    metadata reading are two things rather than one long one — and so this half can be
    exercised against a metadata dict with no file behind it.

    Note the direction of every check: the checkpoint is not wrong, the adapter simply
    cannot drive it. A 24-ray retrain (issue #29) will land here, and the right response to
    that message is to widen the adapter, not to fix the checkpoint.
    """
    problems = []
    input_dim = metadata.get("actor_input_dim")
    if input_dim is not None and input_dim != OBS_DIM:
        problems.append(
            f"checkpoint expects a {input_dim}-value observation; this adapter builds "
            f"{OBS_DIM}")
    layout = metadata.get("observation_layout")
    if layout is not None:
        rays = sum(1 for term in layout if str(term).startswith("lidar"))
        if rays != N_RAYS:
            problems.append(
                f"checkpoint has {rays} lidar features; this adapter casts {N_RAYS} rays")
    return problems


def inspect_model(path: str | Path) -> ModelReport:
    """Read a candidate checkpoint and say whether this adapter could drive it.

    Never raises for a bad file — a corrupt download is an ordinary outcome of a cloud
    fetch, and the caller wants a report it can show, not an exception it has to catch to
    build the same report. It raises only if ``path`` cannot be read at all.
    """
    path = Path(path)
    size = path.stat().st_size
    problems: list = []
    metadata: dict = {}

    try:
        # allow_pickle stays False. An .npz is a zip of arrays and enabling pickle would
        # make a downloaded checkpoint arbitrary code execution on the robot.
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            missing = sorted(set(EXPECTED_SHAPES) - names)
            if missing:
                problems.append(f"missing array(s) {missing}; not a MAPPO actor checkpoint")
            for key, expected in EXPECTED_SHAPES.items():
                if key in names and data[key].shape != expected:
                    problems.append(
                        f"{key} has shape {tuple(data[key].shape)}, this adapter needs "
                        f"{expected}")
            if "metadata_json" in names:
                try:
                    metadata = json.loads(str(data["metadata_json"]))
                except ValueError:
                    problems.append("metadata_json is not valid JSON")
                if not isinstance(metadata, dict):
                    problems.append("metadata_json is not a JSON object")
                    metadata = {}
    except Exception as exc:
        problems.append(f"could not be read as an .npz: {exc}")

    problems.extend(_metadata_problems(metadata))

    return ModelReport(
        name=path.name,
        path=path,
        size_bytes=size,
        sha256=sha256_file(path),
        loadable=not problems,
        problems=problems,
        metadata=metadata,
    )


class ModelStore:
    """The checkpoints in one policy package, and which of them the next run will use.

    Args:
        package_dir: the policy package — the directory holding ``config.json`` and
            ``models/``. This is the same directory ``mappo_drive.py --package`` takes, and
            pointing the two at different places is the one way to make a swap appear to do
            nothing.
    """

    def __init__(self, package_dir: str | Path, config_name: str = "config.json") -> None:
        self.package_dir = Path(package_dir).resolve()
        self.config_path = self.package_dir / config_name
        if not self.config_path.is_file():
            raise ModelStoreError(
                f"{self.config_path} does not exist; --package must point at the policy "
                f"package (the directory holding config.json and models/)")

    # ── config ───────────────────────────────────────────────────────────────
    def config(self) -> dict:
        data = json.loads(self.config_path.read_text())
        if not isinstance(data, dict):
            raise ModelStoreError(f"{self.config_path}: expected a JSON object")
        return data

    @property
    def models_dir(self) -> Path:
        """Where checkpoints live: the parent of whatever ``model_path`` points at.

        Derived from the config rather than hard-coded to ``models/`` so that a package
        laid out differently still works — and so that a store never writes to a directory
        the configured model is not already in.
        """
        configured = Path(self.config().get("model_path", "models/x.npz"))
        if configured.is_absolute():
            return configured.parent
        return (self.package_dir / configured).parent

    def active_model(self) -> str | None:
        """The filename the config currently points at, or ``None`` if it points nowhere."""
        configured = self.config().get("model_path")
        return Path(configured).name if configured else None

    def active_path(self) -> Path | None:
        configured = self.config().get("model_path")
        if not configured:
            return None
        path = Path(configured)
        return path if path.is_absolute() else self.package_dir / path

    # ── inventory ────────────────────────────────────────────────────────────
    def list_models(self) -> list:
        """Every ``.npz`` in the models directory, inspected, active one flagged.

        Sorted by name so the dashboard's list does not reorder itself between polls for
        reasons the operator cannot see.
        """
        active = self.active_model()
        directory = self.models_dir
        if not directory.is_dir():
            return []
        out = []
        for path in sorted(directory.glob(f"*{MODEL_SUFFIX}")):
            report = inspect_model(path).as_dict()
            report["active"] = (path.name == active)
            report["compatible_with_config"] = not self._incompatibilities(
                inspect_model(path))
            out.append(report)
        return out

    def path_for(self, name: str) -> Path:
        return self.models_dir / safe_model_name(name)

    # ── the swap ─────────────────────────────────────────────────────────────
    def _incompatibilities(self, report: ModelReport) -> list:
        """Why this checkpoint and the CURRENT config cannot be run together.

        Separate from :func:`inspect_model`'s ``problems``, which are properties of the file
        alone. This is the pairwise check, and it is the one that turns a crash at the start
        of a live run into a refusal at the moment of the click.
        """
        problems = list(report.problems)
        trained = report.trained_lidar_range_vmas
        if trained is not None:
            configured = self.config().get("lidar_range_vmas")
            if configured is not None and not _close(float(trained), float(configured)):
                problems.append(
                    f"{report.name} was trained with lidar_range_vmas={trained} but "
                    f"config.json says {configured}. The proximity convention is measured "
                    f"against this range, so the run would be refused at load. Change "
                    f"lidar_range_vmas to {trained}, or pick another checkpoint — do not "
                    f"change meters_per_vmas_unit, which is a different number.")
        return problems

    def select(self, name: str) -> dict:
        """Arm ``name`` for the next run by rewriting ``model_path``.

        Refuses a checkpoint this adapter cannot drive, or one whose training constants
        disagree with the config it would run under. Rewrites only ``model_path`` and keeps
        every other key and its order, so a config someone hand-tuned is not reformatted by
        a button press.
        """
        name = safe_model_name(name)
        path = self.models_dir / name
        if not path.is_file():
            raise ModelStoreError(f"{name} is not on this robot")

        problems = self._incompatibilities(inspect_model(path))
        if problems:
            raise ModelStoreError(
                f"refusing to arm {name}: " + " ".join(problems))

        previous = self.active_model()
        data = self.config()
        configured = Path(data.get("model_path", ""))
        # Preserve the config's own idiom: if it referred to the model relatively, keep it
        # relative. An absolute path written into a config that is copied to another robot
        # is a path that does not exist there.
        if configured.is_absolute():
            data["model_path"] = str(path)
        else:
            data["model_path"] = str(Path(configured.parent) / name)
        self._write_config(data)
        return {"active": name, "previous": previous,
                "takes_effect": "next run — a live run holds its weights in memory"}

    def _write_config(self, data: dict) -> None:
        """Write the config as one atomic replace.

        A half-written ``config.json`` is a robot that cannot start, and the window is real:
        this is a Jetson writing to an SD card while a demo is being set up. Write a
        sibling temp file and rename it, which is atomic within a filesystem.
        """
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(self.config_path)

    # ── install / remove ─────────────────────────────────────────────────────
    def install(self, name: str, source: str | Path, *, overwrite: bool = False) -> dict:
        """Move a downloaded file into the models directory under ``name``.

        Validates BEFORE it lands: a file that cannot be driven never appears in the
        inventory, so it can never be selected by someone reading a list. Refuses to
        overwrite the armed checkpoint outright — replacing the bytes under the config's
        feet is a swap that no event records and that nothing can undo.
        """
        name = safe_model_name(name)
        source = Path(source)
        report = inspect_model(source)
        if not report.loadable:
            raise ModelStoreError(
                f"refusing to install {name}: " + " ".join(report.problems))

        destination = self.models_dir / name
        if destination.exists():
            if name == self.active_model():
                raise ModelStoreError(
                    f"{name} is the armed checkpoint; arm a different one before replacing "
                    f"its file, or install under another name")
            if not overwrite:
                raise ModelStoreError(f"{name} is already on this robot; pass overwrite")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Replace rather than copy+unlink so a reader never sees a partial file, and so an
        # existing checkpoint survives a failed install.
        source.replace(destination)
        installed = inspect_model(destination).as_dict()
        installed["active"] = False
        installed["compatible_with_config"] = not self._incompatibilities(
            inspect_model(destination))
        return installed

    def remove(self, name: str) -> dict:
        """Delete a checkpoint from the robot.

        Refuses the armed one. Deleting it would leave ``config.json`` pointing at a file
        that is not there, and the failure would surface as a missing-file error at the
        start of the next live run rather than here.
        """
        name = safe_model_name(name)
        if name == self.active_model():
            raise ModelStoreError(
                f"{name} is armed for the next run; arm a different checkpoint first, "
                f"otherwise config.json would point at a file that is not there")
        path = self.models_dir / name
        if not path.is_file():
            raise ModelStoreError(f"{name} is not on this robot")
        size = path.stat().st_size
        path.unlink()
        return {"removed": name, "freed_bytes": size}


def _close(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1.0)
