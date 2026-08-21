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
| **Dashboard** | <http://mappo.10-241-11-4.nip.io:8090/> |
| **Checkpoint sources** | Arm: <http://models.10-241-11-4.nip.io:9000/index.json> · S3 stand-in: <http://models.10-241-11-4.nip.io:9001/index.json> |
| Same thing by IP | <http://10.241.11.4:8090/> · <http://10.241.11.4:9000/index.json> |
| Host | `waheedbrown-learning-paths`, Ubuntu 24.04, **aarch64** (`Standard_D2ps_v6`, Arm Neoverse), eastus |
| Fleet | `demo-go2-01`, `demo-go2-02`, `demo-lite3-01`, `demo-lite3-02` — all simulated |
| Survives a reboot | yes, via a user crontab `@reboot`; no sudo, nothing system-wide |
| Control | `~/mappo-demo-src/deploy/demo/run_demo.sh start|status|stop` |

### The hostname

`nip.io` is wildcard DNS that decodes an IP out of the name it is asked for:
`mappo.10-241-11-4.nip.io` resolves to `10.241.11.4`, and any label in front is free. So a
readable, self-describing name costs no DNS record, no ticket and no infrastructure — which
is worth knowing, because the obvious conclusion after checking reverse DNS, the
Azure-provided internal FQDN and Azure Private DNS is that a name is impossible here. It is
not; those are just the wrong three places to look.

Two caveats, neither serious:

* **It depends on a third party.** If `nip.io` is unreachable the name stops working and the
  IP still does. `mappo.10-241-11-4.sslip.io` is the same trick from a different operator and
  resolves identically, so there are two independent fallbacks behind one IP.
* **The lookup is public.** Resolving this tells the service's operator that somebody asked
  about `10.241.11.4`. That is an RFC1918 address belonging to millions of networks, so it
  identifies nothing — but the query does leave the building, and on a host holding anything
  sensitive that is a reason to use the IP instead.

⚠️ **The VM has no public IP.** `publicIpAddress` is empty in its instance metadata, so this
is reachable from inside the Arm network and nowhere else — the hostname is a convenience,
not a route. An audience joining from outside needs a public IP and an NSG rule on the
subscription: a change to somebody's cloud estate, not something the demo can arrange for
itself. Worth settling **before** the room, not in it.

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

## The two checkpoint sources

Every robot advertises where its checkpoints can come from, so the dashboard offers a named
**choice** — `Arm Neoverse CPU server — Tokyo, Japan` or `AWS S3 — cn-north-1, Beijing` — and
nobody has to remember a URL. The list is advertised by the **robot**, not configured in the
dashboard, because it is a property of the deployment the robot sits in; two robots on one
mesh can legitimately pull from different places, which a dashboard-level setting cannot say.

⚠️ **Neither source is what its name says, and both say so.** The first is this VM in
`eastus`, not a CPU server in Tokyo. The second is the *same program* with a different label
and different contents — **it is not AWS, there is no bucket, and nothing here speaks the S3
API.** They exist so the demo can show the *choice between two places*, which is the actual
subject. `simulated: true` travels with each into the dashboard, which prints it under the
picker.

The real S3 path is still there and still works — pick **custom address…** and give a bucket,
and `cloud_models.list_s3` does the genuine thing with boto3 and real credentials.

⚠️ **The addresses are what the ROBOT must reach, not what a browser must.** The download runs
on the robot. On this host they are the same machine, so loopback would work — which is
exactly the trap for anyone copying this file to a deployment where the robot is elsewhere.

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
