# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pull one representative still out of a run, for a scene capture per run.

The still is a recorded frame, so the plan-view radar ``overlay.draw_plan_view``
burns into the top-right corner comes with it — goal, mapped objects and range
rings, already drawn. Nothing here re-renders that; re-rendering would invite the
capture and the run to disagree about what the robot believed.

WHICH frame: the first one whose tick had acquired a goal, because a capture taken
before acquisition shows an empty radar and says nothing about where the robot
thought it was going. A run that never acquired a goal falls back to the first
recorded frame, and the filename says ``nogoal`` so the two are never confused.

Ticks carry ``perception.video_frame``, the index of the matching ``--record``
frame. That mapping is the only reliable way across: the recorder writes one frame
per perception cycle, not per control tick, so frame N is NOT tick N and a run
recorded 58 frames for 81 ticks.

Usage: scene_capture.py RUN.mp4 RUN.jsonl OUT_DIR
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _ticks(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue          # a run killed mid-write leaves a half-line; skip it
        if record.get("type") == "tick":
            out.append(record)
    return out


def pick_frame(ticks: list[dict]) -> tuple[int, dict | None]:
    """``(video_frame_index, tick)`` for the capture, and whether a goal was held.

    Falls back to frame 0 when nothing acquired a goal or when no tick carries a
    frame index — a run recorded without ``--record`` has ``video_frame: null``
    on every tick, and treating that as frame 0 is better than failing.
    """
    for tick in ticks:
        if not tick.get("goal"):
            continue
        index = (tick.get("perception") or {}).get("video_frame")
        if isinstance(index, int):
            return index, tick
    return 0, None


def capture(mp4: Path, jsonl: Path, out_dir: Path) -> Path:
    ticks = _ticks(jsonl)
    index, tick = pick_frame(ticks)
    suffix = "goal" if tick else "nogoal"
    out = out_dir / f"{mp4.stem}--frame{index:03d}-{suffix}.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select=eq(n\\,{index})", "-vsync", "0", "-frames:v", "1",
         "-q:v", "2", str(out), "-y"],
        check=True,
    )
    goal = (tick or {}).get("goal") or {}
    objects = (tick or {}).get("obstacles") or []
    print(f"{out}")
    print(f"    frame {index}   goal={goal.get('distance_m', 'none')}   "
          f"objects={[o.get('label') for o in objects]}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    mp4, jsonl, out_dir = Path(argv[1]), Path(argv[2]), Path(argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    capture(mp4, jsonl, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
