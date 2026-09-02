# SAFETY — operating a real humanoid under low-level / learned control

**English** | [简体中文](SAFETY_CN.md)

This document is the **non-negotiable** safety layer for running motor-level control on a
physical humanoid: velocity-walk, `rt/arm_sdk` overlays, and
especially **whole-body RL-policy deploy** (`rt/lowcmd` with the vendor balance controller
released). It is written from a real incident — read it before any of the on-hardware
control skills ([deploy-policy](https://github.com/arm/arm-dc-nvidia-isaaclab/blob/main/skills/deploy-policy/SKILL.md), [locomotion](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/locomotion/README.md),
[controller](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/controller/README.md)).

It is humanoid-general; G1 specifics are called out.

---

## 0. The cardinal rule

> **NEVER hard-kill (`kill -9` / `SIGKILL`) a process that is sending low-level motor
> commands to a real robot. A hard kill is the OPPOSITE of a stop.**

When the publisher of `rt/lowcmd` (or any motor-command topic) dies without sending a final
safe command, the robot's motor controller keeps acting on the **last command it received** —
a *position target held at the configured stiffness*. The DDS layer makes this worse: the
writer's **last sample persists** to the motor subscriber (and a buffered sample can flush as the
writer is torn down), so the stale high-gain target keeps being applied with nothing left to
update or damp it. If those were policy targets at
transfer/sim gains (on the G1, `kp` up to **150–200**), the joints drive **hard** toward that
posture with no further updates and no rescue. On a downed or unsupported robot that is a
violent runaway.

**What this looked like (2026-06-12, G1 whole-body bed-reach deploy):** a whole-body policy
run on a gantry didn't transfer; the operator pressed the controller **abort, which correctly
damped it**. The deploy process was still alive, so it was `kill -9`'d "to stop the commands."
That hard-kill left the last high-gain reach command latched on the motors with no publisher to
update or damp it — the robot **spin-kicked on the floor and broke a window.** The abort was the
right stop; the hard-kill undid it.

A `SIGKILL` cannot be caught, so a process cannot damp on the way out. That is *why* you never
use it as a stop — and why every deploy process must install a catchable-signal damp handler
(§2, `lib/safe_stop.py` (arm-dc-robotkit `lib/safe_stop.py`)).

---

## 1. The stop hierarchy (most authoritative first)

Always have the higher tiers physically in reach before any motion.

1. **Hardware e-stop / power cut** — the only *guaranteed* stop. Cuts motor power regardless of
   software state. Keep a hand on it / the battery for every run.
2. **Controller firmware damp — `L2+B`** — the handheld's firmware-level damping. Processed by
   firmware, so it works **even if your control process has hung or DDS has stalled**; sets the
   joints **compliant** (`kp→0`). This is the backstop for when software can no longer help.
3. **In-loop ANY-button abort (the primary abort during a run)** — the
   [`G1Remote`](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/controller/README.md) **any-button latch** wired into the control loop:
   the operator presses **ANY button** and the loop catches it within one ~20 ms tick, runs the
   clean `kp=0, kd≈small, tau=0` damp on **all** joints (or releases the arm overlay), then exits.
   You should **never have to hunt for a specific combo to stop a run — mash any button.** This is
   what the loop polls every tick and what `SafeStop` (arm-dc-robotkit `lib/safe_stop.py`) damps on. It is
   software-mediated, so tiers 1–2 remain the backstops if the loop itself ever stalls.
   ⚠️ **The any-button latch is a *clean* abort in Develop mode** (the vendor controller is paused,
   so buttons are not bound to gestures — they are just bits the loop reads). In **Regular / AI-Sport
   mode the `A/B/X/Y` combos fire vendor gesture routines** *in addition* to tripping the latch, so
   prefer `L2+B` / the e-stop there. Always `confirm_abort_live` (press+release, see the latch) *in
   the mode you will run in* before any motion. (Exiting Develop mode itself is a **reboot** — there
   is no software re-engage of the vendor controller.)

**A process kill is NOT on this list.** If a process must be terminated, **damp first (any button),
confirm the robot is compliant, then terminate** — and never with `-9`.

---

## 2. Signal-safe deploy (required for any `rt/lowcmd` / `rt/arm_sdk` process)

Every process that commands motors MUST guarantee a damp on **every** exit — normal return,
exception, `SIGINT` (Ctrl-C), and `SIGTERM` (`kill` without `-9`). Use
`lib/safe_stop.py` (arm-dc-robotkit `lib/safe_stop.py`):

```python
from arm_dc_robotkit.safe_stop import SafeStop

def damp():               # command compliant on ALL joints (kp=0, kd≈3, tau=0), ~1 s
    g1.publish_damp(seconds=1.0)

with SafeStop(damp):      # damps on return, exception, SIGINT, SIGTERM
    run_control_loop()    # ... your 50 Hz loop, polling the controller abort each tick
```

`SafeStop` cannot protect against `SIGKILL` (uncatchable) or a power loss — those are exactly
why tiers 1–2 above exist. It removes the *catchable* failure modes so the only ways out leave
the robot compliant.

Also keep a **standalone panic-damp** script you can run from a second shell
([`lib/safe_stop.py --panic` pattern]): it opens its own publisher and floods `kp=0` damp to all
joints. Use it to make a robot inert *without approaching it*.

---

## 3. The de-risk ladder — deploying a whole-body policy onto real legs

A first sim→real transfer of a whole-body balance policy **will fall if anything is off** (joint
order, obs scaling, the un-observable base-velocity term, the takeover-pose gap). Failures are
**fast** (sub-second, faster than a catch reflex), not gradual. So stage it; each rung gates the
next:

| Rung | What runs | Fall risk | Gate to pass |
|---|---|---|---|
| **0 — offline** | Build the live obs, run the policy, **print** obs+actions. No commands. | none | obs sane (gravity ≈ `[0,0,-1]`), **joint-order verified** (predicted pose offsets land on the named joints), actions finite/bounded |
| **1 — partial, fall-safe** | Policy runs, but apply **only a subset that can't drop the robot** (e.g. arms via `rt/arm_sdk` while the legs stay on the **vendor balance controller**). Motion-blended, rate-limited, clamped, abort-armed. | none (legs held by vendor) | smooth, bounded, abortable; IMU stays level (legs unaffected) |
| **2 — full whole-body** | All joints via `rt/lowcmd`. The operator has the robot in **low-level Develop mode** (entered by hand); the code **verifies** that and does **not** release the vendor in software. Move-to-default → blend → policy. | **HIGH — can fall** | **Develop mode verified**, **abort proven live** in that mode, **feet-on-ground + slack gantry** (NOT fully suspended), e-stop in reach, clear area |

**Strongly prefer rung 1 for any useful task the arms can do.** Arms-over-vendor-balance delivers a
reach with **zero whole-body balance risk** and needs no gantry. Only take on rung 2 if the task
**demonstrably** needs leg/CoM control the vendor balancer can't provide.

**Rung 2 requires support — but not suspension.** Use a **gantry/hoist with a few cm of slack**, the
robot's **feet on the ground**: it bears its own weight and balances, the hoist catches a fall in a
few centimetres. **Do NOT fully suspend it:** a balance policy assumes ground-reaction dynamics, so
fully off the ground it is **off-distribution — flailing is the guaranteed behavior, not a corner
case**, and you learn nothing. "Spot it by hand" is also not adequate for a sub-second leg failure.
(A vision-less policy reaches a *target coordinate*, not a perceived surface — you do **not** need the
task furniture in front of it for a balance test, and clear space makes a fall cleaner.)

---

## 4. Handover & gains (rung 2 specifics)

- **Hand off to low-level control BY HAND — operator-driven, never a software release.** The old
  `MotionSwitcher.ReleaseMode()` path took the legs while the operator believed the robot was in a
  safe mode (§0 incident). The operator drives the G1 into low-level Develop mode instead:

  | Action | Buttons (FW ≥1.4) | Notes |
  |---|---|---|
  | **Damping / e-stop** | **`L2+B`** (old: `L1+A`) | firmware-level compliant damp; the operator e-stop; works even if the control process has hung |
  | Locked standing | `L2+UP` | from damping; support the shoulders |
  | Regular / AI-Sport | `R1+X` | **buttons fire vendor gestures here** (so the in-loop any-button latch is *not clean* in this mode) |
  | **Develop / low-level `rt/lowcmd`** | **`L2+R2`** | **precondition: SUSPENDED + DAMPING first**; pauses AI-Sport; here **any button is a clean abort**; **exit = reboot** |

  **Sequence: suspend → `L2+B` (damping) → `L2+R2` (Develop).** The code only **verifies** this — but
  **NOT via `MotionSwitcher.CheckMode()`, which lies:** on the G1 EDU it returns the *configured* mode
  (`name='ai'`) in BOTH AI-Sport and a freshly-entered Develop mode, and `rt/sportmodestate` is silent
  on this variant — neither says whether the balancer is actually running (verified live; corroborated
  by unitree_sdk2_python#43). The reliable signal is **high-level service liveness**: a read-only loco
  GET-FSM RPC (`ROBOT_API_ID_LOCO_GET_FSM_ID`) **answers (code 0) when the balancer is running and
  errors/times out when it is off** (Develop → code 3102, verified live). `verify_whole_body_ready`
  gates on that: **service answers → REFUSE** (a controller is active); **service down → proceed**;
  **probe unavailable → proceed only on an explicit operator Develop assertion** (fail-safe).
  **Never let `rt/lowcmd` fight a running high-level controller — it jitters/oscillates, it does not
  cleanly take over** (unitree_sdk2_python#43/#108).
- **PREFER the fall-safe arm overlay over whole-body takeover on the G1 EDU.** `rt/lowcmd` whole-body
  is only valid when the high-level controller is fully *off* (Develop mode) — on this variant trying
  to drive all joints against the live AI controller makes the legs fight it. For arm tasks (the
  bedside reach), keep the robot in Regular/AI-Sport mode and use the **`rt/arm_sdk` overlay**: it
  *blends* with the controller (`executed = controller·(1−w) + arm_sdk·w`, weight ramped 0→1;
  unitree_sdk2_python#108) instead of fighting it, and is fall-safe + needs no gantry. Drive
  locomotion with the vendor `LocoClient.Move`. Whole-body `rt/lowcmd` is a **gantry-only, last-resort
  research path**, and on the EDU it is best treated as a stand-only experiment, not the task path.
- **lowcmd silently no-ops unless** you copy `mode_machine` from the first `LowState` into every
  `LowCmd` and set each commanded motor's `mode=1` (FOC enable) — independent of the mode gate above
  (unitree_sdk2_python#44). The G1 binding does both.
- **The abort is ANY button — prove it first.** In Develop mode the operator must be able to **mash
  any button** to abort (the in-loop latch → clean damp; §1.3), never hunt for a specific combo.
  `confirm_abort_live` enforces this before any motion: the operator presses+releases the handheld and
  the code confirms the any-button latch fires **in this mode**. (This is exactly the trap that bit us
  — in Regular mode the "abort test" buttons fired gestures and destabilized the robot, so the latch
  is only *clean* in Develop mode.)
- **`rt/lowcmd` writer QoS: explicit VOLATILE + keep-last-1.** So a torn-down writer leaves no
  history for the middleware to retransmit to the motor subscriber. (Defense-in-depth behind §0/§2 —
  the real protection is the SafeStop damp and never `kill -9`.)
- **The legs only bear load once you command the default pose at gains.** Low-level Develop mode
  starts in zero-torque/damping — the bare mode *cannot stand* (the tether holds the weight). The
  legs become load-bearing when the program drives the **default pose at the trained PD** (the
  move-to-default), NOT via the vendor balancer. This is why you can't "launch from Regular mode," and
  it dictates the startup order (mirrors `unitree_rl_gym deploy_real`: zero-torque → develop →
  move-to-default → lower hoist → policy → lower rope).
- **Kill the activation transient + blend EVERY transition** (a snap is harmless on an arm, a fall on
  a leg): (1) a **scripted, gain-ramped** move to the EXACT default pose (damping-first — kd nominal
  throughout, kp ramped up), policy out of the loop; (2) an **operator checkpoint** — hold the default
  stance and wait for the operator to lower the tether (**stage 1**, feet bear weight) and signal
  proceed; **fail-safe damp** if no proceed (never auto-start the policy); (3) start the policy on a
  **neutral command** and **ramp the command** (operator lowers the tether **stage 2** once stably
  balancing); (4) on completion, **return** by ramping the command back to neutral with the policy
  *still balancing* (never an open-loop snap), then **gently** ease the pose to default and ramp kp→0;
  (5) **first whole-body test = STAND**, add a reach only after a stable stand.
- **Match the sim PD gains** the policy trained under (G1 bed-reach: hip 100/2, knee 150/4,
  ankle 40/2, waist 200/5, arms 40/10). Wrong gains = no transfer. (Read them from the deploy
  contract so they can't drift; a missing-gains contract would publish `kp=0` → zero torque.)
- **Abort / end → low-level damp** (`kp=0, kd≈3`), not a kill (§0). Under a gantry the hoist
  then holds the robot; exiting Develop mode itself is a reboot.

---

## 5. After an incident — make it safe **without approaching**

1. **Do not approach a robot that just ran away.** Cut power first if you can.
2. From a safe distance, make it inert in software if it's still commanded: run the **panic-damp**
   (§2) — a `kp=0` flood can only make it compliant, it cannot drive a posture.
3. **Verify inert with read-only telemetry before approaching:** subscribe to `rt/lowstate` and
   confirm, over a few seconds, **max |joint velocity| ≈ 0**, **max |torque| ≈ 0** (passive, not
   driven), the **IMU frozen**, and that **no process is publishing** the command topic. Only then
   approach — ideally from behind / away from the limbs' swing arc; with motors limp the joints are
   back-driveable, not powered.
4. Power off (pull the battery) before re-rigging.

---

## 6. Pre-run checklist (every on-hardware control run)

- [ ] Hardware e-stop / battery in reach; operator's hand on it.
- [ ] Controller abort **armed** and wired into the loop ([`G1Remote`](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/controller/README.md)).
- [ ] Process uses `SafeStop` (arm-dc-robotkit `lib/safe_stop.py`) (damps on return/exception/SIGINT/SIGTERM).
- [ ] Rung 2 only: robot on a **gantry** with slack; clear area; vendor-release verified.
- [ ] You know the **panic-damp** command for a second shell.
- [ ] Verify by **what you see**, not telemetry — but use telemetry to confirm *inert* before approaching.
- [ ] **Commanded top speed is at or above the platform's gait floor.** On the Go2 that is
      ~0.35 m/s (`MIN_GAIT_COMMAND_M_S`); below it the robot stands still while being
      commanded forward and reports no fault. This is on the safety checklist rather than
      only in the nav docs because of what it does to *diagnosis*: the encoders, the state
      estimator and the stall gate all agree the robot is being physically held, so the
      operator is sent to look for a tether or an obstruction that does not exist —
      and time spent on a wrong physical cause is time spent near a live robot.

> Premises are worth challenging everywhere in this project — **except physical safety.** There,
> default to the most conservative stop and never improvise a "just kill it" shortcut mid-motion.
