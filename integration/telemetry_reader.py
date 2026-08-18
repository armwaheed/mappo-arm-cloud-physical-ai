# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read the control stack's telemetry: header, ticks, outcome.

The robot writes one JSON object per control tick (``visual_nav.py --telemetry``). This
is the consumer side, kept deliberately small — it validates the schema major, splits
the three record types, and gets out of the way.

WHY NOT PARSE THE CONSOLE LOG. Because it does not contain what it appears to. Measured
over a 107-tick live run of this stack, the console printed the robot's pose **once**, in
a start-up banner, and no camera data at all — ``lat=235ms`` is a frame's AGE, not a
frame. The commands are there and the goal appears only as a scalar distance. An
observation vector cannot be built from it. The console format also changes to stay
readable (``people=0`` became ``obst=[binx1,personx1]`` inside a week), which is correct
for prose and fatal for a parser.

Pure stdlib. ``python3 test_observation.py`` covers this and ``observation.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Schema majors this reader understands. Refuse anything else rather than guess — a
#: renamed field that silently reads as None is worse than a loud failure.
SUPPORTED_MAJORS = (1,)


class SchemaError(RuntimeError):
    """The file is not a telemetry stream this reader can interpret."""


@dataclass
class Run:
    """One recorded run: its configuration, its ticks, and how it ended."""

    header: dict
    ticks: list = field(default_factory=list)
    outcome: dict | None = None

    @property
    def schema(self) -> str:
        return self.header.get("schema", "")

    @property
    def video(self) -> str | None:
        """Path the matching annotated MP4 was written to, if any."""
        return self.header.get("video")

    @property
    def completed(self) -> bool:
        """Whether the run wrote an outcome line.

        False means the file was truncated — the process was killed, or is still
        running. Worth checking before treating a run as a finished episode, since the
        writer flushes per line specifically so a killed run still yields data.
        """
        return self.outcome is not None

    def moving_ticks(self) -> list:
        """Ticks that commanded a non-zero velocity.

        Each component is read with a default because a ``command`` block can be present
        and PARTIAL: ``mappo_policy.tick_from_state`` builds ``{"reason": ...}`` with no
        velocity in it, and indexing that raised ``KeyError`` from a public method of the
        reader. A tick that records only a reason commanded no velocity this reader can
        see, which is "not moving" — the same answer as an absent block.
        """
        return [t for t in self.ticks
                if t.get("command")
                and any(abs(t["command"].get(k) or 0.0) > 0.0
                        for k in ("vx", "vy", "wz"))]


def _schema_major(schema: str) -> int:
    try:
        return int(schema.rsplit("/", 1)[1].split(".")[0])
    except (IndexError, ValueError) as exc:
        raise SchemaError(f"cannot read a schema version out of {schema!r}") from exc


def read_run(path: str | Path) -> Run:
    """Parse a ``.jsonl`` telemetry file into a :class:`Run`.

    Raises :class:`SchemaError` on a missing or unsupported header. Blank lines are
    skipped; a trailing partial line (the process died mid-write) is tolerated, because
    the alternative is discarding a whole run over its last few bytes.
    """
    header = None
    ticks: list = []
    outcome = None
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Only ever acceptable for the LAST line, which is the one a kill can
                # truncate. Anything earlier means real corruption.
                remaining = handle.read().strip()
                if remaining:
                    raise SchemaError(f"{path}:{number}: corrupt mid-file") from None
                break
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "tick":
                ticks.append(record)
            elif kind == "outcome":
                outcome = record

    if header is None:
        raise SchemaError(f"{path}: no header record — not a telemetry stream")
    major = _schema_major(header.get("schema", ""))
    if major not in SUPPORTED_MAJORS:
        raise SchemaError(
            f"{path}: schema major {major}, this reader understands "
            f"{SUPPORTED_MAJORS}. A major bump means a field was renamed or removed.")
    return Run(header=header, ticks=ticks, outcome=outcome)
