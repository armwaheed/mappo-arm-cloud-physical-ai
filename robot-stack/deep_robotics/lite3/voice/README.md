<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spoken cues

Chinese first, then English, for every cue. The demo is in Shanghai and the sentence that
matters most — *"please step out of my path"* — is useless if the person it is addressed to
has to wait through an English sentence to hear it.

## What is committed here, and what is not

**Committed: `lite3_stand_request_*.wav`.** Ours, generated for this repository.

> 我还没有站起来。请在遥控器上把我切换到站立模式。
> I am not standing yet. Please set me to standing mode on the controller.

It exists because no borrowed cue said it. `assert_axis_state_ready` refuses a robot that
is not in force-control standing (`basic_state=8; axis motion requires documented
force-control state 6`), and **no amount of retrying clears that by itself** — a person
has to change the mode. That makes it the one failure where speaking is the entire remedy
rather than narration, so it needs a sentence that says what to do. The nearest borrowed
cue, `fault` — *"A fault has occurred. I have stopped."* — is true and tells nobody
anything actionable.

**Not committed: the `rrd_*` cues.** They are borrowed from the sibling repository
[`armwaheed/rrd-deep-robotics-dr02`](https://github.com/armwaheed/rrd-deep-robotics-dr02),
`assets/voice`, which is where their provenance and their `manifest.json` live. Copying
audio between repositories is how a licence question gets created out of nothing — the same
reasoning [`docs/WHITEPAPER.md` Appendix D](../../../../docs/WHITEPAPER.md) gives for
rendering vendor URDFs rather than committing them.

Fetch them onto the robot with:

```sh
gh repo clone armwaheed/rrd-deep-robotics-dr02 /tmp/rrd -- --depth 1
scp /tmp/rrd/assets/voice/rrd_{person_stop,person_thanks,resuming,fault,greeting}_{zh,en}.wav \
    user@<robot>:~/mappo-lite3-stage/voice/
```

`mission.py` names any cue file it cannot find **at start-up**, rather than discovering it
at the moment the robot needs to speak — which is the one time the information is useless.

## Format

24 kHz, mono, `pcm_s16le`, matching the RRD manifest so the two sets are interchangeable.
The committed pair was generated with macOS `say` (`Tingting` / `Samantha`) and converted
with `afconvert -f WAVE -d LEI16@24000 -c 1`.

## Cues that are deliberately unused

RRD's `arrived`, `scanning`, `walking` and `no_table` all name a **table**, because that is
what DR02 drives to. This demo drives to an ArUco marker. A robot announcing furniture it is
not looking for reads as a bug to anyone who speaks the language.
