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
