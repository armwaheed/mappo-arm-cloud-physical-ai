<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Demo network — one isolated LAN, no tethers / 演示网络 —— 独立局域网，无网线牵绊

**EN** — Every robot and the operator's laptop on one airgapped WiFi LAN, so the robots walk
untethered and the Device Connect dashboard can actually discover them.
**中文** —— 所有机器人与操作笔记本接入同一个隔离的 WiFi 局域网：机器人无需拖线行走，
Device Connect 面板也才能发现它们。

## Why not just use the Ethernet cable

Three reasons, and only the first is about tidiness.

1. **A cable runs through the lane the robot is meant to walk down.** Every run on
   2026-09-01 ended with one.
2. **A tether is not a LAN.** Device Connect's D2D discovery is **multicast**. Two robots on
   two point-to-point cables to one laptop share no broadcast domain, so the fleet table
   cannot fill however correct the dashboard is. One WiFi LAN fixes this by construction.
3. **The laptop's routing is at risk.** macOS ranks Ethernet above Wi-Fi, so a router
   offering a default gateway captures the laptop's traffic. It happened twice.

## The topology

```
      corp WiFi ───────── en0 ┐
      (internet, unchanged)   ├── operator laptop
                              │
  ┌──────────────────┐  en9 ──┘   manual IP, NO gateway, NO DNS
  │ NETGEAR RAX50    │◄─── LAN port (optional; WiFi works too)
  │ 192.168.1.1      │
  │ WAN: EMPTY       │◄··· WiFi ···► robot 1  192.168.1.120
  └──────────────────┘                robot 2  192.168.1.121
         airgapped                     one broadcast domain
```

## Router settings

| setting | value | why |
| --- | --- | --- |
| SSID | `NETGEAR93` (+ `-5G`) | both bands: 5 GHz is faster, 2.4 GHz reaches further in a hall |
| Security | **WPA2-PSK** | not WPA3 — WPA2 is what the robot's NetworkManager was verified against |
| LAN | `192.168.1.0/24`, router `.1` | matches the robots' existing addressing; nothing on the robot needed changing |
| Address reservation | robot MAC → fixed IP | the robots reboot often; a moving IP mid-demo is an avoidable failure |
| Client isolation | **off** | this model exposes no such control on the main SSID; verified by test, not by checkbox |
| WAN port | **EMPTY** | airgap from the Arm network, enforced by an unpopulated socket |

⚠️ **Change the admin password before the venue.** `admin`/`password` is tolerable on an
office island and not in a hotel where the SSID is visible to everyone in range.

## Laptop — the setting that stops it losing the internet

Set the Ethernet service to a **manual address with the router field EMPTY**, and clear its
DNS. With no gateway it is structurally incapable of becoming the default route, whatever
the service order says.

```sh
sudo networksetup -setmanual "USB 10/100 LAN" 192.168.1.50 255.255.255.0 ""
sudo networksetup -setdnsservers "USB 10/100 LAN" Empty
netstat -rn -f inet | grep default      # MUST still name the Wi-Fi interface
```

The empty final argument is the whole fix. DHCP does not work here: the RAX50 advertises
itself as a gateway, macOS ranks Ethernet above Wi-Fi, and the laptop's internet stops.

If the browser bounces to `routerlogin.net` and fails, corp DNS is resolving it to the real
internet host. Point it locally:

```sh
sudo sh -c 'echo "192.168.1.1 www.routerlogin.net routerlogin.net" >> /etc/hosts'
```

## Robot

```sh
sudo nmcli device wifi connect NETGEAR93 password <passphrase> ifname wlan0
sudo nmcli connection modify NETGEAR93 connection.autoconnect-priority 100
sudo nmcli connection modify GuestAccess connection.autoconnect no
```

**Disabling other saved networks is not tidiness.** `GuestAccess` is in range at the office,
was set to autoconnect, and would compete for `wlan0` on every boot — and these robots reboot
several times an hour during setup. A robot that silently joins the wrong network at the venue
is a demo that cannot be driven.

Audio, if the robot is to speak — the account must be in the `audio` group or PulseAudio
falls back to a null sink and `aplay` exits 0 while making no sound:

```sh
sudo usermod -aG audio "$USER"     # then start a NEW session
```

## Addressing, both robots

Two Lite3s ship with the **same** factory address, `192.168.1.120`, so the second one to
join a shared LAN collides with the first. Give each robot's wired interface a distinct
address before it ever meets the other.

| | robot 1 (LITE3-A) | robot 2 |
| --- | --- | --- |
| `wlan0` (demo path) | `192.168.1.120` | `192.168.1.2` |
| `wlan0` MAC | `54:ef:33:9e:88:2a` | `54:ef:33:9e:1a:8d` |
| `eth1` (cable fallback) | `192.168.1.119` | `192.168.1.118` |
| DHCP reservation | done | **still to do** |

⚠️ **`jy_exe` sends robot state to exactly one address**, set in
`/home/ysc/jy_exe/conf/network.toml`. It ships as `192.168.1.120`, so **moving the wired
interface off `.120` silently breaks the drive path** — `mappo_drive` then dies with *"no
Lite3 state frame arrived on 127.0.0.1:43897 within 5s"*. Robot 1 kept working only by
luck: its `wlan0` inherited `.120`, so the vendor's target still resolved.

Set it to `127.0.0.1`. The drive runs **on** the robot, so localhost is both correct and
immune to any future address change:

```toml
ip = '127.0.0.1'
target_port = 43897
local_port = 43893
```

It takes effect when `jy_exe` restarts. **Restarting it is a motion-controller restart** —
do it with an operator present, not remotely on an unattended robot.

## Bootstrapping a robot with no internet

The robots are airgapped, so Python dependencies cannot be fetched on the robot, and
`python3 -m venv` on this image cannot `ensurepip`. Neither is a problem: **wheels are zip
files**, so cross-download them on a laptop and extract them onto a `PYTHONPATH` directory.
The venv still exists because `preflight/venv_guard.py` requires one for `--live`; it simply
holds no packages.

```sh
# on the laptop
pip download --only-binary=:all: --platform manylinux2014_aarch64     --python-version 38 --implementation cp --abi cp38     "numpy<1.25" "opencv-python-headless<4.10" -d wheels/

# on the robot, after copying wheels/ across
cd ~/mappo-lite3-stage/python && for w in ~/mappo-lite3-stage/wheels/*.whl; do
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$w"
done
export PYTHONPATH=$HOME/mappo-lite3-stage/python
```

**This installs nothing into the system interpreter**, which `AGENTS.md` forbids outright.

The detector model is not in this repository and must be copied across too: `--model-dir`
expects `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`, the **stock**
published pair. ⚠️ Checksum a newly-fetched copy against a working robot's before trusting
it — this repository's detector measurements are tied to specific weights.

## ⚠️ The trap that cost the most time

**Do not let two interfaces hold the same address.** With a cable in *and* WiFi up, `eth1`
and `wlan0` both carried `192.168.1.120`. Linux keeps one route per subnet and picks it by
metric, so:

- **ARP worked**, because ARP replies leave by the interface the request arrived on — the
  laptop happily resolved the robot's MAC;
- **ICMP and TCP did not**, because replies left by whichever interface had the lower metric,
  which was repeatedly the one that had just been unplugged.

It presents exactly like client isolation. **When ARP succeeds and everything above it fails,
read the routing table before blaming the access point.**

`eth1` now sits on `192.168.1.119` so the two no longer collide. The better fix, still to do,
is putting `eth1` on a **different subnet** (`192.168.137.0/24`, the vendor default) so no
metric arbitrates between them at all.

## Verifying it, in order

Do these **before** removing the cable — the whole point is to prove the new path works while
the old one is still there.

```sh
netstat -rn -f inet | grep default          # 1. laptop default route unchanged
ping 192.168.1.120                          # 2. robot answers
arp -a | grep 192.168.1.120                 # 3. and the MAC is the WIRELESS one
ssh user@192.168.1.120 'echo ok'            # 4. SSH, which is what actually matters
ssh user@192.168.1.120 \
  'curl -m 6 --interface wlan0 https://github.com'   # 5. MUST fail — airgap intact
```

Step 3 matters: with a cable also attached, step 2 can pass over Ethernet and tell you
nothing about the WiFi.

## If the WiFi fails at the venue

`eth1` keeps `192.168.1.119` and `192.168.137.120`, so a direct laptop-to-robot cable still
works as a fallback. Set the laptop's Ethernet to `192.168.1.50` (as above) and connect to
`192.168.1.119`.
