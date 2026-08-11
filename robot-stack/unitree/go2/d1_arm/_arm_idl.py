# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored CycloneDDS IDL for the Unitree D1 arm, isolated in its own module.

This file MUST NOT use ``from __future__ import annotations``. cyclonedds 0.10.2
resolves a dataclass's field annotations *by name against the class's module* when it
populates the topic type (``IdlStruct.__idl__.populate()``). Under stringified
(PEP 563 / future) annotations, a field typed ``str`` is stored as the string
``"str"``, which cyclonedds then fails to resolve — it looks up ``str`` as a module
attribute, not a builtin — raising ``TypeError: Type str ... cannot be resolved``.

``d1_arm.py`` itself needs future annotations (it uses ``list[float]`` / ``X | None``
on Python 3.8), so the IDL type is defined here, in a module deliberately left on
eager annotations, and imported by ``d1_arm._vendored_armstring()``.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct


@dataclass
class ArmString_(IdlStruct, typename="unitree_arm::msg::dds_::ArmString_"):
    """The D1 ``unitree_arm/msg/ArmString`` — a single string payload (JSON command).

    The ``::``-separated typename is LOAD-BEARING: it must match the C++ DDS type
    ``unitree_arm::msg::dds_::ArmString_`` exactly, or the arm's command subscriber
    silently drops our messages (no ack, no motion). A ``.``-separated name still
    receives feedback but is not accepted for commands. Verified on-hardware 2026-07-08
    against the D1Py reference (github.com/omar-mostafa81/D1Py)."""
    data_: str = ""
