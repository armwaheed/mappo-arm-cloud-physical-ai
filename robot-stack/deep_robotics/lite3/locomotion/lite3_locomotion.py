# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deep Robotics Lite3 Venture locomotion through the vendor ROS 2 bridge.

``Lite3_ROS`` exposes the high-level gait controller as ``/cmd_vel`` and publishes
measured body velocity and odometry on ``/leg_odom2``.  This adapter deliberately uses
that path instead of ``Lite3_MotionSDK``: MotionSDK takes over the twelve joints and
would require this repository to supply a balance controller, while the ROS bridge keeps
the manufacturer's gait and balance controller in charge.

An external vendor interface remains the authority for posture and control mode. Before
a live run the operator stands the robot and enables its high-level navigation mode,
then supplies ``--operator-ready``. The posture methods below validate that
acknowledgement and do not embed firmware-dependent simple-command codes.
"""

from __future__ import annotations

import time
from collections.abc import Callable


def _ros2_locomotion(**kwargs):
    """Construct the shared ROS 2 Twist binding without importing ROS off-robot."""
    try:
        from arm_dc_robotkit.ros2_twist_locomotion import Ros2TwistLocomotion
    except ImportError:
        # The robotkit can also be deployed flat with its ``lib`` directory on
        # PYTHONPATH; deploy/install.sh supports both layouts.
        from ros2_twist_locomotion import Ros2TwistLocomotion

    return Ros2TwistLocomotion(**kwargs)


class Lite3Locomotion:
    """The navigator's locomotion interface over Lite3 ROS ``Twist``/``Odometry``.

    The implementation is composed rather than subclassed so this module remains
    importable on a workstation without ROS 2 or the shared robotkit.  Both are loaded
    only by :meth:`connect`, which also makes the pure lifecycle behaviour testable with
    a small injected implementation.
    """

    def __init__(self, *, cmd_vel_topic: str = "/cmd_vel",
                 odom_topic: str = "/leg_odom2", operator_ready: bool = False,
                 implementation_factory: Callable = _ros2_locomotion,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._cmd_vel_topic = cmd_vel_topic
        self._odom_topic = odom_topic
        self._operator_ready = operator_ready
        self._implementation_factory = implementation_factory
        self._sleep = sleep
        self._impl = None
        self._manual_posture_reported = False

    def connect(self) -> None:
        """Connect to the documented Lite3 ROS bridge and wait for odometry."""
        if self._impl is not None:
            raise RuntimeError("locomotion is already connected")
        self._impl = self._implementation_factory(
            cmd_vel_topic=self._cmd_vel_topic,
            odom_topic=self._odom_topic,
            stamped=False,
            node_name="mappo_lite3_locomotion",
        )
        try:
            self._impl.connect()
        except Exception:
            self.shutdown()
            raise

    def prepare_motion(self) -> None:
        """Require the operator-controlled posture and auto-mode transition.

        The public Lite3 bridge leaves this transition to an external vendor interface;
        its exact mechanism differs by firmware. Failing here is preferable to
        publishing velocities that the robot silently ignores in manual mode.
        """
        if not self._operator_ready:
            raise RuntimeError(
                "the Lite3 operator has not confirmed STANDING + high-level navigation "
                "mode; use the vendor-approved interface, keep the emergency stop in "
                "hand, then pass --operator-ready"
            )
        if self._impl is None:
            raise RuntimeError("connect() first")

    def recover(self) -> None:
        """Validate readiness; posture itself remains under operator/app control."""
        self.prepare_motion()
        self._report_manual_posture()

    def stand(self) -> None:
        """Validate readiness; posture itself remains under operator/app control."""
        self.prepare_motion()
        self._report_manual_posture()

    def stand_down(self) -> None:
        """Stop motion; the operator returns the Lite3 to manual/prone in the app."""
        self.stop()

    def _report_manual_posture(self) -> None:
        if not self._manual_posture_reported:
            print("[Lite3Locomotion] posture is operator-controlled; standing + auto "
                  "mode acknowledged")
            self._manual_posture_reported = True

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if self._impl is None:
            raise RuntimeError("connect() first")
        self._impl.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        """Publish three zero commands so one best-effort ROS sample cannot be lost."""
        if self._impl is None:
            return
        self._stop_implementation(self._impl)

    def _stop_implementation(self, implementation) -> None:
        for index in range(3):
            implementation.stop()
            if index < 2:
                self._sleep(0.02)

    def pose(self):
        if self._impl is None:
            raise RuntimeError("connect() first")
        return self._impl.pose()

    def velocity(self) -> tuple[float, float, float]:
        if self._impl is None:
            raise RuntimeError("connect() first")
        return self._impl.velocity()

    def battery_level(self) -> float:
        """Return the battery percentage when the selected transport reports it."""
        if self._impl is None:
            raise RuntimeError("connect() first")
        battery = getattr(self._impl, "battery_level", None)
        if battery is None:
            raise RuntimeError("selected Lite3 transport does not report battery level")
        return battery()

    def mode(self) -> tuple:
        """Return the vendor state tuple when the selected transport exposes it."""
        if self._impl is None:
            raise RuntimeError("connect() first")
        mode = getattr(self._impl, "mode", None)
        if mode is None:
            raise RuntimeError("selected Lite3 transport does not expose vendor state mode")
        return mode()

    def assert_axis_state_ready(self) -> None:
        """Require the selected transport's documented simple-axis state gate."""
        if self._impl is None:
            raise RuntimeError("connect() first")
        gate = getattr(self._impl, "assert_axis_state_ready", None)
        if gate is None:
            raise RuntimeError("selected Lite3 transport is not a simple-axis transport")
        gate()

    def shutdown(self) -> None:
        """Stop first, then release the shared ROS binding. Safe after partial setup."""
        implementation, self._impl = self._impl, None
        if implementation is None:
            return
        try:
            self._stop_implementation(implementation)
        finally:
            implementation.shutdown()
