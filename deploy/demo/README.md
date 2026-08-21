<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The demo host

The dashboard, a simulated fleet, a replayed camera and a self-hosted checkpoint server, on
one machine with no robots attached to it.

## ⚠️ What is real here and what is not

This matters more than anything else on this page, because the whole demo is a set of claims
made to people who cannot check them from the room.

| | |
| --- | --- |
| **The Device Connect mesh** | REAL. Drivers announce themselves, are discovered, and are invoked over Zenoh exactly as a robot is. |
| **The dashboard** | REAL — the same code that runs against hardware, unmodified. |
| **The robots** | **SIMULATED.** Every one says so on its fleet row, in its device identity (`Go2 EDU (SIMULATED)`) and in `get_capabilities`. |
| **The gait floors, posture rules and refusals** | REAL. A simulated Go2 refuses 0.21 m/s with the same message and the same measured number as the robot did. |
| **The travelled distances** | **NOT real.** The bench double delivers exactly what it is commanded — `delivered_fraction` is 1.00, which is precisely the number a real robot never produces. |
| **The camera** | **A RECORDED RUN**, replayed. Every frame is labelled in the pixels, not by the page, because a screenshot keeps the pixels and loses the caption. |
| **The checkpoint server's location** | **A CLAIM.** `--location` is a caption; the index reports it as `location_claimed` alongside `simulated: true`. |

A demo fleet that is indistinguishable from a real one is a hazard rather than a better demo:
somebody eventually presses a key believing a robot is on the other end of it.

## The standing deployment

| | |
| --- | --- |
| **Dashboard** | <http://10.241.11.4:8090/> |
| **Checkpoint server** | <http://10.241.11.4:9000/index.json> |
| Host | `waheedbrown-learning-paths`, Ubuntu 24.04, **aarch64** (`Standard_D2ps_v6`, Arm Neoverse), eastus |
| Fleet | `demo-go2-01`, `demo-go2-02`, `demo-lite3-01`, `demo-lite3-02` — all simulated |
| Survives a reboot | yes, via a user crontab `@reboot`; no sudo, nothing system-wide |
| Control | `~/mappo-demo-src/deploy/demo/run_demo.sh start|status|stop` |

⚠️ **That address is as shareable as it gets, and it is not very.** The VM has **no public
IP** — `publicIpAddress` is empty in its instance metadata — so `10.241.11.4` is reachable
only from inside the Arm network. No DNS name resolves for it either, internal or external.
For an audience joining from outside, this needs a public IP and an NSG rule on the
subscription, which is a change to someone's cloud estate rather than something the demo can
arrange for itself. Worth settling **before** the room, not in it.

⚠️ **The host has another tenant.** `papa-web` holds `0.0.0.0:80` and `0.0.0.0:8080`, and a
Hugo server holds `1313`. Nothing here touches them; the ports were chosen around them.

## Install

```bash
./install_demo.sh                 # a venv under ~/.mappo-demo; nothing system-wide
```

A venv is not optional on current Ubuntu — 24.04 marks the system Python externally-managed
(PEP 668) and `pip install` refuses. `--break-system-packages` on a host running other
services is how you break the other services.

Then populate the three asset directories and start:

```bash
~/.mappo-demo/policy/          # config.json + models/  (the policy package)
~/.mappo-demo/served-models/   # .npz the checkpoint server offers
~/.mappo-demo/frames/          # .jpg the camera replays

./run_demo.sh start | status | stop
```

## The fleet

Two Go2s and two Lite3s, all simulated. **One Lite3 runs without `--allow-motion` on
purpose**, so the demo always has a robot whose motion keys are correctly refused — a gate is
easier to believe when you can watch it hold.

The simulation keeps each platform's *rules* while driving the bench double: `--platform`
carries the gait floors and posture semantics, `--backend sim` decides what is actually
driven. That split is load-bearing. Collapsing them hands the demo the bench double's floors
of zero, and the refusal that is this stack's most characteristic behaviour silently stops
firing.

## The camera

Frames come from a recorded Go2 run, pre-extracted to JPEG:

```bash
ffmpeg -i run.mp4 -vf "scale=-2:480,crop=640:480" -q:v 6 frames/f%04d.jpg
```

Pre-extracted rather than decoded live, so the demo host needs neither OpenCV nor ffmpeg for
a job `ffmpeg` already did once. They loop, because a demo outlives an 11-second recording
and a feed frozen on its last frame looks exactly like a camera that has died.

## The checkpoint server

`model_server.py` serves a JSON index and the `.npz` files beside it. **This required no new
capability in the dashboard** — `cloud_models` has accepted an `http(s)://` source since the
first commit. What was missing was that a server could be *browsed* the way a bucket could,
so the UI implied a bucket was the real answer. Both are peers now, and the server is listed
first.

```bash
python3 model_server.py --dir ./checkpoints --port 9000 \
    --label "Arm Neoverse CPU server" --location "Tokyo, Japan"
```

Pass `--real` only when it is genuinely running where `--location` says. Without it the index
reports `simulated: true`, and the dashboard shows what the server says about itself.

## Ports, and the host's other tenants

Nothing here binds 80 or 8080 — a demo that takes down the machine's existing services is not
a demo. Defaults are **8090** (dashboard) and **9000** (checkpoint server), overridable with
`MAPPO_DASH_PORT` and `MAPPO_MODEL_PORT`.

## ⚠️ The dashboard has no authentication

It binds `0.0.0.0` so it can be reached from a laptop, which means anyone who can reach the
port can drive any motion-enabled robot on that mesh. On this host every robot is simulated,
so the blast radius is a log line. **Do not point this arrangement at a mesh with a real
robot on it** without putting something in front of it.
