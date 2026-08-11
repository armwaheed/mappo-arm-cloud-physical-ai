---
name: unitree-go2-connect
description: >-
  Put a control host on the Unitree Go2's network and configure CycloneDDS so unitree_sdk2py sees the
  robot's DDS topics (rt/lowstate, rt/sportmodestate, rt/utlidar/*, rt/arm_Command, ...). The Go2 EDU
  exposes its DDS on the 192.168.123.0/24 wired net; connect the host to that subnet, point
  CYCLONEDDS_URI at unitree/go2/cyclonedds.xml, and pass the Go2-facing interface to
  ChannelFactoryInitialize. Use this first, before locomotion / d1_arm / lidar_sight / deploy.
  Also covers putting the robot's own Jetson on WiFi — which needs a USB dongle, because the Go2
  EDU's Orin NX ships with no WiFi radio.
metadata:
  tags: [unitree-go2, networking, cyclonedds, dds, connect, setup, wifi, wlan]
---

# Unitree Go2 — Connect (host ↔ robot networking)

Get a control host onto the Go2's DDS network. Once connected, every other Go2 skill
([locomotion](../locomotion/SKILL.md), [d1_arm](../d1_arm/SKILL.md),
[lidar_sight](../lidar_sight/SKILL.md), [deploy](../deploy/)) just works over `unitree_sdk2py`.

## The network

The Go2 EDU's computers live on **`192.168.123.0/24`** (wired):

| Host | Role |
|---|---|
| `192.168.123.161` | sport/motion MCU — `rt/lowstate`, `rt/sportmodestate`, `rt/api/sport/request` |
| `192.168.123.18` | onboard nav/AI Jetson — LiDAR graph-SLAM + VIO (the usual SSH host) |
| `192.168.123.100` | expansion dock (piggyback) — bridges the D1 arm onto DDS |

Connect your control host to this subnet one of two ways:

1. **Run on the robot's onboard Jetson** (`ssh unitree@192.168.123.18`) — it is already on the net;
   `iface="eth0"`. Simplest for bring-up.
2. **Wire an external host** into the Go2's dock/switch (or bridge it onto `192.168.123.0/24`). Give the
   host a `192.168.123.x` address and use its Go2-facing NIC as `iface`.

## Configure DDS

```bash
export CYCLONEDDS_URI="file://$PWD/unitree/go2/cyclonedds.xml"   # edit the NIC name inside first
```

Then initialise the channel factory with the SAME interface name:

```python
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0, "eth0")   # every Go2 binding does this for you via connect()
```

Confirm the link is live with a read-only probe (no motion):

```bash
python3 unitree/go2/locomotion/go2_locomotion.py --iface eth0 --seconds 5   # prints mode + measured pose
```

## Put the robot's own Jetson on WiFi

Goal: reach the Go2's compute over WiFi with **no ethernet cable** — i.e. SSH `unitree@<wifi-ip>`
instead of `unitree@192.168.123.18`. The board to put on WiFi is the **onboard Orin (`192.168.123.18`)**,
because that is the SSH host and where every robotics-connect process runs. (The expansion dock and the
sport MCU are *not* shortcuts to this — see [below](#the-expansion-dock-is-not-a-shortcut).)

**The Go2 EDU's Orin NX has no WiFi radio. A USB WiFi dongle is required.** This is a hardware fact,
not a configuration gap — no amount of driver or NetworkManager work brings up `wlan0` without one.
Unitree's own guidance agrees: *"The Orin in the GO2 generally doesn't have a WiFi module installed,
and it is recommended to use a dongle."*

Confirm the absence in three commands on `192.168.123.18` — all three must be empty:

```bash
lsusb                    # only the two root hubs (1d6b:0002 / 1d6b:0003) => nothing plugged in
rfkill list              # EMPTY output => zero radios registered (not blocked — absent)
ls /sys/class/net/       # docker0 dummy0 eth0 lo — no wlan0
```

Boot dmesg shows the tell cleanly: `usbcore: registered new interface driver rtl8852bu` (the driver
loads fine) with **no** matching `new USB device found`, and `tegra-xusb ... entering ELPG done` —
the xHCI controller power-gates itself because nothing is attached.

### The driver is already staged

The Realtek **RTL8852BU** driver ships pre-installed and loaded on the Go2 Jetson
(`/lib/modules/5.10.104-tegra/updates/8852bu.ko`), so an 8852BU-class dongle is plug-and-play. Other
chipsets (e.g. the commonly recommended TL-WN722N) generally work on Ubuntu without extra drivers.
Insert the dongle **before powering the robot on**.

### Trap: `go2-ap` outranks your network

The factory ships a NetworkManager profile named **`go2-ap`** that claims `wlan0` in **AP mode**
(SSID `GO2-DIRECT`, `ipv4.method=shared`, `192.168.50.1/24`) at **`autoconnect-priority 100`**. It is
invisible today because `wlan0` doesn't exist — but the moment a dongle enumerates it will win, and
the robot will *broadcast its own access point instead of joining your network*. Any station profile
must outrank it:

```bash
sudo nmcli con add type wifi con-name <SSID> ifname wlan0 ssid <SSID> \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk '<password>' \
    connection.autoconnect yes connection.autoconnect-priority 200 \
    ipv4.method auto
nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY con show   # station must sit above go2-ap
```

Leaving `go2-ap` at 100 rather than disabling it gives a sensible fallback: join the known network
when in range, otherwise fall back to the robot's own AP. Adding the profile is safe with no radio
present — it stays inert until `wlan0` appears, and it does not disturb `eth0`. `wpa-psk` covers
WPA2-PSK; if the network is WPA3-SAE-only the connect fails and the profile needs
`wifi-sec.key-mgmt sae` instead.

### Bring it up and find its address (cable-free bootstrap)

Do the first bring-up **while still on ethernet**, so you can read the new WiFi address before you pull
the cable. Insert the dongle, then on `192.168.123.18`:

```bash
nmcli device wifi rescan
nmcli -t -f DEVICE,STATE dev | grep wlan0        # expect wlan0 ... connected
ip -4 -br addr show wlan0                         # <-- this is your new cable-free SSH address
```

Then from your workstation, `ssh unitree@<that-ip>` — and only now unplug ethernet. On an air-gapped
network with no convenient lease table, a small NetworkManager dispatcher (installed this session at
`/etc/NetworkManager/dispatcher.d/90-ausrd-log`) appends the address to `/var/log/ausrd-wlan0.log` on
every `wlan0` up-event, so it is still recoverable after the cable is gone (read it back over the
robot's own AP, or on the next ethernet session). sshd already binds `0.0.0.0:22`, so SSH answers on
`wlan0` with no further config. Nothing about this removes the wired path — `eth0` stays up and the
`192.168.123.0/24` DDS bus is unaffected.

### The expansion dock is not a shortcut

The board on the Go2's back that the D1 arm plugs into is the **expansion dock (`192.168.123.100`)** —
and it *is* a full Ubuntu 20.04 host (OpenSSH + vsFTPd), not a dumb bridge, so it could in principle
carry its own radio. It is still the wrong target for cable-free access:

- Even if the dock joined the WiFi, the **Orin** would not become reachable over it unless the dock
  also routed/NATed `192.168.123.0/24` — a fragile extra hop versus simply dongling the Orin.
- The dock uses **separate credentials** from the Orin's `unitree/123` (it is reached only over DDS by
  the [d1_arm](../d1_arm/SKILL.md) skill, never SSH), so its interior — and whether it has WiFi at all
  — can't be inspected without those credentials in hand.

The sport MCU (`192.168.123.161`) *does* hold the factory Go2 WiFi (the `GO2_xxxx` AP and the app's
station link), but it exposes only port `9991` (no SSH), and that radio is meant to be pointed at a
network **through the Unitree app**, not to carry an SSH session to the Orin. Neither board changes the
answer: for cable-free SSH, dongle the Orin.

### The G1's `setup-wifi.sh` does not port to the Go2

Do not reach for [`unitree/g1/install/setup-wifi.sh`](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/install/setup-wifi.sh)
here. It assumes two things the Go2 does not have:

| G1 assumption | Go2 reality |
|---|---|
| Driver `.deb` at `/home/unitree/wifi-bt-deb/` | Directory absent — the `.ko` is installed directly |
| `rtl8852bu-dkms` package installed | Not a package here; nothing in `dpkg -l` |
| `systemctl enable nvwifibt.service` | Fails — `nvwifibt` is a **static** unit (no `[Install]`) |

That last one is worth spelling out: on the Go2, `nvwifibt.service` is stock NVIDIA L4T plumbing that
shells out to `brcm_patchram_plus` over `ttyTHS*` for **Broadcom Bluetooth** chips (BCM4329/4330/4324/
4354/4356). It has nothing to do with a Realtek USB WiFi device, and starting it will not produce a
`wlan0`.

Both robots use the same 8852bu chipset, but only the G1 needs the driver *installed*; the Go2 needs
the *radio* physically present.

## Notes

- The bundled [`cyclonedds.xml`](../cyclonedds.xml) uses SPDP discovery on one wired interface, mirroring
  the robot's own config. Set the `<NetworkInterface name="...">` to your host's NIC (Linux `eth0`,
  macOS `en0`).
- To reach the Go2 from a WiFi-only host through a gateway (e.g. a DGX Spark), the G1 stack's IP-forward
  + static-route scripts (`unitree/g1/configure_robot.sh` / `configure_spark.sh`) are a working template —
  substitute the Go2's `192.168.123.x` addresses. That routes a *host* onto the robot's wired subnet; to
  put the *robot* on WiFi instead, see [above](#put-the-robots-own-jetson-on-wifi) (needs a USB dongle).
- If you do run DDS over `wlan0` rather than `eth0`, the interface name is pinned in two places — the
  `<NetworkInterface name="...">` in [`cyclonedds.xml`](../cyclonedds.xml) and the `ChannelFactoryInitialize`
  argument. Change both, or discovery silently stays on the wired NIC.
- `192.168.123.161` (the sport MCU) exposes only port `9991` and no SSH, so its WiFi state is not
  inspectable from the Jetson — the robot's app-facing radio is a separate board from the Orin.
- First check the [install](../install/SKILL.md) skill if `unitree_sdk2py` isn't yet on the host.
