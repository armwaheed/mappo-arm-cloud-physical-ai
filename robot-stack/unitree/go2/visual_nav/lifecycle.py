# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small lifecycle helpers shared by the robot-specific run entry points."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable


def run_cleanup(
    prefix: str,
    steps: Iterable[tuple[str, Callable[[], None] | None]],
) -> None:
    """Attempt every cleanup step without hiding the exception that ended the run.

    A camera teardown failure must not skip the final locomotion stop. If normal work
    succeeded, the first cleanup error is still raised after all steps have had a chance;
    if another exception is already unwinding, cleanup failures are reported without
    replacing the more useful original cause.
    """
    already_failing = sys.exc_info()[0] is not None
    first_error: Exception | None = None
    for label, callback in steps:
        if callback is None:
            continue
        try:
            callback()
        except Exception as exc:
            print(f"[{prefix}] cleanup failed in {label}: {exc!r}", file=sys.stderr)
            if first_error is None:
                first_error = exc
    if first_error is not None and not already_failing:
        raise first_error
