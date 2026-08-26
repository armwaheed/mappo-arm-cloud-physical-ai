#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mirror validated Lite3 telemetry without changing the motion host's destination setting.

The Lite3 motion host sends state to one configured receiver. Run this on that existing receiver
to forward valid raw state datagrams to a separate application port, for example a host-local
port used by a navigation process. This module never encodes or targets a motion command.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROBOT_STACK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROBOT_STACK))

from deep_robotics.lite3.commissioning.lite3_state_probe import (  # noqa: E402
    DEFAULT_PORT,
    DecodeError,
    decode_frame,
)

DEFAULT_RELAY_PORT = 43898
MOTION_COMMAND_PORT = 43893
DEFAULT_SECONDS = 30.0
MAX_SECONDS = 3600.0


@dataclass(frozen=True)
class RelaySummary:
    """Counts from a bounded telemetry relay session."""

    received: int
    sent: int
    rejected: int
    ignored: int
    elapsed_s: float


class Lite3StateRelay:
    """Send only datagrams accepted by the shared Lite3 state decoder.

    UDP ``sendto`` confirms local enqueue, not target delivery. Treat ``sent`` counts as local
    sender evidence only; a separate receiver capture must prove an end-to-end telemetry path.
    """

    def __init__(self, *, listen_host: str = "0.0.0.0", listen_port: int = DEFAULT_PORT,
                 target_host: str, target_port: int = DEFAULT_RELAY_PORT,
                 source_address: str | None = None,
                 socket_factory=socket.socket, clock=time.monotonic) -> None:
        if not 0 <= listen_port <= 65535:
            raise ValueError("--listen-port must be within 0..65535")
        if not 1 <= target_port <= 65535:
            raise ValueError("--target-port must be within 1..65535")
        if target_port == MOTION_COMMAND_PORT:
            raise ValueError(
                f"refusing to relay telemetry to Lite3 command port {MOTION_COMMAND_PORT}"
            )
        if target_port == listen_port:
            raise ValueError("target port must differ from the relay listen port")
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._target_host = target_host
        self._target_port = target_port
        self._source_address = source_address
        self._socket_factory = socket_factory
        self._clock = clock
        self._receiver: socket.socket | None = None
        self._sender: socket.socket | None = None
        self._received = 0
        self._sent = 0
        self._rejected = 0
        self._ignored = 0

    @property
    def listen_port(self) -> int:
        """The actual bound source port, including when zero requested an ephemeral port."""
        return self._listen_port

    def start(self) -> None:
        """Bind the existing telemetry destination and open a separate relay sender."""
        if self._receiver is not None:
            raise RuntimeError("state relay is already running")
        receiver = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sender = None
        try:
            receiver.bind((self._listen_host, self._listen_port))
            sender = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            if self._source_address is not None:
                sender.bind((self._source_address, 0))
        except OSError:
            receiver.close()
            if sender is not None:
                sender.close()
            raise
        self._receiver = receiver
        self._sender = sender
        self._listen_port = receiver.getsockname()[1]

    def shutdown(self) -> None:
        """Release sockets. Safe after failed or partial startup."""
        for attribute in ("_receiver", "_sender"):
            sock = getattr(self, attribute)
            setattr(self, attribute, None)
            if sock is not None:
                sock.close()

    def forward_once(self, timeout_s: float) -> bool:
        """Receive and optionally enqueue one datagram; return whether it was locally sent."""
        if timeout_s < 0.0:
            raise ValueError("timeout must not be negative")
        receiver = self._receiver
        sender = self._sender
        if receiver is None or sender is None:
            raise RuntimeError("start() first")
        receiver.settimeout(timeout_s)
        try:
            payload, _source = receiver.recvfrom(2048)
        except socket.timeout:
            return False
        self._received += 1
        try:
            frame = decode_frame(payload)
        except DecodeError:
            self._rejected += 1
            return False
        if frame["kind"] != "robot_state":
            self._ignored += 1
            return False
        sender.sendto(payload, (self._target_host, self._target_port))
        self._sent += 1
        return True

    def run_for(self, seconds: float) -> RelaySummary:
        """Relay for one bounded interval and return receipt/forwarding counts."""
        if not 0.0 < seconds <= MAX_SECONDS:
            raise ValueError(f"--seconds must be within 0..{MAX_SECONDS:.0f}")
        started = self._clock()
        while True:
            remaining = seconds - (self._clock() - started)
            if remaining <= 0.0:
                break
            self.forward_once(min(0.2, remaining))
        return RelaySummary(
            received=self._received,
            sent=self._sent,
            rejected=self._rejected,
            ignored=self._ignored,
            elapsed_s=self._clock() - started,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward validated Lite3 state telemetry to a non-command UDP port.",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, default=DEFAULT_RELAY_PORT)
    parser.add_argument(
        "--source-address",
        help="bind relay output to the Mac Ethernet address accepted by the motion host",
    )
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        relay = Lite3StateRelay(
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            target_host=args.target_host,
            target_port=args.target_port,
            source_address=args.source_address,
        )
        relay.start()
        try:
            summary = relay.run_for(args.seconds)
        finally:
            relay.shutdown()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[lite3_state_relay] FAILED: {error}", file=sys.stderr)
        return 2
    print(
        "[lite3_state_relay] "
        f"received={summary.received} sent={summary.sent} "
        f"rejected={summary.rejected} ignored={summary.ignored} "
        f"elapsed_s={summary.elapsed_s:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
