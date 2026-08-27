<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — one robot, four detectors, and a sweep that scored a fifth

[#129](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/129) opened as *"the
scorers use 300 px and the robot uses 224"*. That is not what is wrong. Enumerated against
every launcher there is:

| launcher | `--input-size` | `--confidence` | `--classes` |
| --- | ---: | ---: | ---: |
| `deploy/run-peer-supervised.sh` | **224** | 0.25 | all **20** VOC labels |
| `run-smoke.sh`, `run-berth.sh`, `run-chair.sh` — on the robot, not in this repository | none → **300** | 0.45 | none → **`person` only** |
| a bare `visual_nav.py` | default → **300** | 0.4 | default → **`person` only** |
| **what the 2026-08-26 checkpoint sweep scored** | **300** | **0.25** | all 20 |

**The same robot runs at two squares, three score floors and two class lists depending on
which script starts it, nothing reconciles them, and no run recorded which one produced
it.** The last row is the sweep's, and it is not a launcher: it took the square from a
scorer constant and the floor from the peer launcher, and produced a combination nothing
has ever run. So the scorers' 300 was *right* for three launchers and wrong for the fourth
— and the sweep managed to be wrong for all of them at once.

⚠️ **`--classes` belongs in that table and is easy to miss.**
`PersonDetector.detect_tiered` drops any detection whose label is not in the list *before*
the tracker, the map or the planner sees it. The peer launcher passes all twenty labels
precisely because the peer is not reliably labelled `person` — this repository has it
recorded as `horse 0.28` head-on at 1.3 m. The other launchers pass none, and get
`("person",)`. So on a smoke run a peer detected perfectly as `motorbike` is **not an
obstacle at all**, and that is not a threshold effect: the box is discarded.

That is a different defect from the one #129 named, and a worse one. A number carried
between two runs launched by different scripts was silently comparing two detectors, and
`evidence/2026-08-27-89-runs-survived-14-can-be-dated/` had to establish what 89 recorded
runs computed by **reading the launchers**, because the logs could not say.

## What is in this directory

**The fix.**
[`robot-stack/unitree/go2/visual_nav/inference_profile.py`](../../robot-stack/unitree/go2/visual_nav/inference_profile.py)
declares every configuration a launcher produces, each naming the file that produces it.
There is deliberately **no `PRODUCTION` object**: there is no single production
preprocessing to point at, and pretending otherwise is how one gets picked by accident.
`deploy/run-peer-supervised.sh` takes its flags from that module rather than holding
literals; every scorer in `detector/` reads it; `--preprocessing` is **required with no
default**, so a scorer has to say which launcher it is scoring for; and a configuration no
launcher runs is refused unless a reason is given, which is then written into the results.

**The record.** `visual_nav.py`'s telemetry header now carries the resolved preprocessing
and the name of the configuration it matches — or `null`, which is a real answer and is
deliberately not rounded to the nearest name. Without that, no past number can be
attributed to a preprocessing, which is the half of #129 that no guard can fix
retrospectively.

**The measurement.** 800 candidate checkpoints — all twenty runs, all forty epochs, none
dropped — plus the incumbent and the 22-class starting point, on the 2026-08-20 cross-day
day, at **all four** configurations. Profiles sharing a square come from one forward pass,
so a difference between those rows is the score floor and cannot be an inference
difference.

```bash
python3 report.py --markdown          # every table below
python3 report.py --check-readme      # fail if this page has drifted from them
```

⚠️ **What the guard is, precisely, so it is not over-read.** Python has no way to forbid a
call. "Structurally impossible" here means three checkable things, none of which is a type
system: no module in `detector/` declares a network input size any more (a test scans the
directory for both forms and carries a negative control); no scorer has a default
preprocessing, so the choice is in the shell history; and the choice is written into the
output, so a row cannot be quoted as a run without the file contradicting it. Somebody can
still call `cv2.dnn.blobFromImage` with a literal in a script they never commit. What they
cannot do is have five committed files disagree and nobody notice.

## One detector, four configurations, on one held-out day

<!-- TABLE-INCUMBENT -->
| one detector, four configurations | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `go2-peer-supervised` — 224 px, floor 0.25, 20 VOC labels<br>`deploy/run-peer-supervised.sh` | 30/60 = 50% | 57/221 = 26% | 27 | 25 | 44 |
| `go2-run-smoke` — 300 px, floor 0.45, `person` only<br>`/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)` | 8/60 = 13% | 5/221 = 2% | 20 | 20 | 32 |
| `go2-navigator-default` — 300 px, floor 0.4, `person` only<br>`robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)` | 8/60 = 13% | 10/221 = 5% | 24 | 24 | 40 |
| `mobilenet-ssd-trained` — 300 px, floor 0.25, 20 VOC labels<br>**run by nothing** | **41/60 = 68%** | **108/221 = 49%** | **40** | **32** | 56 |
<!-- /TABLE-INCUMBENT -->

**The shipped weights are not one detector.** Peer recall runs from **13% to 68%** across
four configurations of the *same weights on the same frames* — and **the most flattering
row is the one no launcher runs.** Every "the shipped network gets 68% recall" in this
repository describes that row. No run of this robot has ever produced it.

The row that matters operationally is `go2-run-smoke`, because that is what **all 89
logged runs** computed, and it sees the peer on a fraction of the frames the sweep's
configuration does. Most of that gap is the class list, not the square: at 300 px the only
difference between it and the sweep's row is the floor and `--classes`, and dropping every
non-`person` label is what removes the peer.

<!-- TABLE-BASE -->
The 22-class starting point every candidate was grown from — the incumbent's own weights through `detector/add_class.py` — reproduces **every one of those rows exactly**, at all four configurations. That is `add_class.py`'s claim checked rather than quoted, and it is what lets a candidate row be read against an incumbent row at all.
<!-- /TABLE-BASE -->

## Lined up against the sweep's own published row

The sweep reported the manifest's `test` split. Its shipped row reads *"68% (32/47), 56%
(76/136), 22/22"* — and all three re-derive here **to the frame**, in the
`mobilenet-ssd-trained` row, from a different script on a different host. That is both the
evidence that the weights measured are the shipped network, and the identification of which
configuration the sweep used.

<!-- TABLE-SPLIT -->
| incumbent, the manifest's `test` split | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `go2-peer-supervised` — 224 px, floor 0.25, 20 VOC labels<br>`deploy/run-peer-supervised.sh` | 25/47 = 53% | 45/136 = 33% | 8 | 7 | 22 |
| `go2-run-smoke` — 300 px, floor 0.45, `person` only<br>`/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)` | 8/47 = 17% | 1/136 = 1% | 6 | 6 | 15 |
| `go2-navigator-default` — 300 px, floor 0.4, `person` only<br>`robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)` | 8/47 = 17% | 1/136 = 1% | 6 | 6 | 15 |
| `mobilenet-ssd-trained` — 300 px, floor 0.25, 20 VOC labels<br>**run by nothing** | **32/47 = 68%** | **76/136 = 56%** | **16** | **8** | 22 |
<!-- /TABLE-SPLIT -->

⚠️ **The `person` label denominator is 22 at 0.25 and 15 at the floors the logged runs use,
and it does not move with the square at all.** The sweep's `people` column counts frames
carrying a box *labelled* `person`; it is 22 at both 224 px and 300 px. The gate the robot
actually applies — `person_detector.person_shaped`, box aspect h/w >= 2.0, whatever VOC
called it — **halves**, 16 to 8, between the sweep's configuration and the peer launcher's.
A denominator insensitive to the axis under test is not a conservative choice; it is a
blind one.

## Does anything beat the incumbent? Once per configuration, because that is the question

<!-- TABLE-HEADLINE -->
| configuration | candidates beating both peer axes | ...and keeping the people | the bar: incumbent recall | hold | candidates holding for nobody |
| --- | ---: | ---: | ---: | ---: | ---: |
| `go2-peer-supervised` — 224 px, floor 0.25, 20 VOC labels<br>`deploy/run-peer-supervised.sh` | 308 of 800 | **6** | 30/60 = 50% | 27 | 117 |
| `go2-run-smoke` — 300 px, floor 0.45, `person` only<br>`/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)` | 65 of 800 | **4** | 8/60 = 13% | 20 | 41 |
| `go2-navigator-default` — 300 px, floor 0.4, `person` only<br>`robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)` | 240 of 800 | **12** | 8/60 = 13% | 24 | 40 |
| `mobilenet-ssd-trained` — 300 px, floor 0.25, 20 VOC labels<br>**run by nothing** | 30 of 800 | **2** | 41/60 = 68% | 40 | 27 |
<!-- /TABLE-HEADLINE -->

**"Does this checkpoint beat the shipped weights" has four different answers**, and they
are not close. The bar moves because the incumbent moves: at the sweep's configuration it
scores 68% peer recall and 49% false alarms — easy to beat on false alarms, hard on recall.
At `run-smoke`'s it scores 13% and **2%** — the opposite shape, and the reason only 65
checkpoints clear both axes there against 308 at the peer launcher's. A ranking taken at
one configuration and quoted at another is not a conservative approximation; it is a
different experiment, and the next table shows it does not even share its winners.

## Who clears every gate, and where

`hold` is the denominator that matters, **and it is per configuration**. It counts frames
where a detection *this launcher's `--classes` lets through* is at or above the aspect
gate. Under the peer launcher every label is let through, so the aspect gate alone
separates *hold the robot* from *route to the policy as an obstacle*; under the two 300 px
launchers only `person` is, so `hold` there is the person-labelled subset and equals
`+shaped` exactly — `report.py` refuses to build a table on a file where those two ever
disagree, which is how it checks that the class filter ran at all. `+shaped` is the label-and-shape figure regardless of configuration — the closest
thing to the sweep's `people` column that is still a shape test. `person_detector`'s own
docstring is why the bare label decides neither: *"on 12 consecutive live frames the Go2
Wheel was labelled `person` every single time"*.

A checkpoint "clears all four" when it beats the incumbent on peer recall **and** on false
alarms **and** loses none of the people on **either** denominator, all at the same
configuration.

<!-- TABLE-CLEARERS -->
**`go2-peer-supervised` — 224 px, floor 0.25, 20 VOC labels<br>`deploy/run-peer-supervised.sh`**

| clears all four | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r_pseudo01_aug/epoch002` | 37/60 = 62% | 41/221 = 19% | 37 | 33 | 58 |
| `m_bb02_d03_aug/epoch002` | 36/60 = 60% | 51/221 = 23% | 28 | 28 | 51 |
| `s_pseudo02_aug/epoch002` | 35/60 = 58% | 33/221 = 15% | 30 | 27 | 60 |
| `r_pseudo01_aug/epoch003` | 34/60 = 57% | 30/221 = 14% | 29 | 25 | 41 |
| `m_bb02_d03_aug/epoch001` | 34/60 = 57% | 50/221 = 23% | 30 | 29 | 51 |
| `k_full_pseudo03/epoch003` | 32/60 = 53% | 39/221 = 18% | 33 | 30 | 65 |
| **incumbent — the bar** | **30/60 = 50%** | **57/221 = 26%** | **27** | **25** | 44 |

**`go2-run-smoke` — 300 px, floor 0.45, `person` only<br>`/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)`**

| clears all four | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a_careful/epoch009` | 11/60 = 18% | 4/221 = 2% | 20 | 20 | 34 |
| `a_careful/epoch010` | 11/60 = 18% | 4/221 = 2% | 20 | 20 | 34 |
| `d_conv5_distil1/epoch010` | 11/60 = 18% | 4/221 = 2% | 20 | 20 | 34 |
| `d_conv5_distil1/epoch011` | 11/60 = 18% | 4/221 = 2% | 20 | 20 | 34 |
| **incumbent — the bar** | **8/60 = 13%** | **5/221 = 2%** | **20** | **20** | 32 |

**`go2-navigator-default` — 300 px, floor 0.4, `person` only<br>`robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)`**

| clears all four | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_pseudo02_aug/epoch020` | 14/60 = 23% | 5/221 = 2% | 25 | 25 | 44 |
| `s_pseudo02_aug/epoch033` | 14/60 = 23% | 7/221 = 3% | 24 | 24 | 45 |
| `s_pseudo02_aug/epoch022` | 13/60 = 22% | 7/221 = 3% | 28 | 28 | 48 |
| `s_pseudo02_aug/epoch023` | 13/60 = 22% | 7/221 = 3% | 24 | 24 | 44 |
| `s_pseudo02_aug/epoch037` | 13/60 = 22% | 7/221 = 3% | 24 | 24 | 44 |
| `s_pseudo02_aug/epoch021` | 12/60 = 20% | 5/221 = 2% | 24 | 24 | 41 |
| `s_pseudo02_aug/epoch024` | 11/60 = 18% | 6/221 = 3% | 25 | 25 | 42 |
| `s_pseudo02_aug/epoch016` | 11/60 = 18% | 8/221 = 4% | 27 | 27 | 46 |
| **incumbent — the bar** | **8/60 = 13%** | **10/221 = 5%** | **24** | **24** | 40 |

...and 4 more.

**`mobilenet-ssd-trained` — 300 px, floor 0.25, 20 VOC labels<br>**run by nothing****

| clears all four (run by nothing) | peer recall | false alarms | hold | +shaped | `person` label |
| --- | ---: | ---: | ---: | ---: | ---: |
| `r_pseudo01_aug/epoch007` | 50/60 = 83% | 35/221 = 16% | 47 | 40 | 71 |
| `s_pseudo02_aug/epoch017` | 42/60 = 70% | 34/221 = 15% | 40 | 33 | 55 |
| **incumbent — the bar** | **41/60 = 68%** | **108/221 = 49%** | **40** | **32** | 56 |
<!-- /TABLE-CLEARERS -->

## Do the winners at one configuration win at another? No — not one of them

<!-- TABLE-OVERLAP -->
| checkpoints clearing all four gates at BOTH | `go2-peer-supervised` | `go2-run-smoke` | `go2-navigator-default` | `mobilenet-ssd-trained` |
| --- | ---: | ---: | ---: | ---: |
| `go2-peer-supervised` — 224 px, floor 0.25, 20 VOC labels<br>`deploy/run-peer-supervised.sh` | — | 0 | 0 | 0 |
| `go2-run-smoke` — 300 px, floor 0.45, `person` only<br>`/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)` | 0 | — | 0 | 0 |
| `go2-navigator-default` — 300 px, floor 0.4, `person` only<br>`robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)` | 0 | 0 | — | 0 |
| `mobilenet-ssd-trained` — 300 px, floor 0.25, 20 VOC labels<br>**run by nothing** | 0 | 0 | 0 | — |

Clearing every gate at **all three deployed** configurations: **0** of 800.

Clearing every gate at **all four**, the sweep's own included: **0**.
<!-- /TABLE-OVERLAP -->

**The sets are pairwise disjoint.** Not one checkpoint of the 800 clears every gate at more
than one of this robot's configurations. So a ranking taken at one configuration is not a
conservative approximation of the ranking at another; it has no members in common with it.
That is the strongest form of the finding, and it is what makes the fix a guard rather than
a footnote: there is no configuration you can score at and then quietly quote elsewhere.

## The checkpoints the sweep ranked highest, at every configuration

<!-- TABLE-NAMED -->
| the sweep's picks | at `go2-peer-supervised` | at `go2-run-smoke` | at `go2-navigator-default` | at `mobilenet-ssd-trained` |
| --- | ---: | ---: | ---: | ---: |
| `s_pseudo02_aug/epoch020` | 27/60 = 45%<br>hold 9 of 27<br>no | 14/60 = 23%<br>hold 17 of 20<br>no | 14/60 = 23%<br>hold 25 of 24<br>**clears all four** | 35/60 = 58%<br>hold 44 of 40<br>no |
| `k_full_pseudo03/epoch022` | 23/60 = 38%<br>hold 8 of 27<br>no | 11/60 = 18%<br>hold 10 of 20<br>no | 11/60 = 18%<br>hold 12 of 24<br>no | 38/60 = 63%<br>hold 19 of 40<br>no |
| `k_full_pseudo03/epoch020` | 28/60 = 47%<br>hold 7 of 27<br>no | 13/60 = 22%<br>hold 8 of 20<br>no | 13/60 = 22%<br>hold 9 of 24<br>no | 36/60 = 60%<br>hold 17 of 40<br>no |
| `k_full_pseudo03/epoch017` | 31/60 = 52%<br>hold 8 of 27<br>beats the peer axes, loses people | 13/60 = 22%<br>hold 8 of 20<br>no | 14/60 = 23%<br>hold 9 of 24<br>no | 39/60 = 65%<br>hold 17 of 40<br>no |
| `k_full_pseudo03/epoch010` | 40/60 = 67%<br>hold 9 of 27<br>beats the peer axes, loses people | 12/60 = 20%<br>hold 12 of 20<br>no | 12/60 = 20%<br>hold 12 of 24<br>no | 45/60 = 75%<br>hold 22 of 40<br>beats the peer axes, loses people |
| `p_bb02_d01_aug/epoch020` | 33/60 = 55%<br>hold 10 of 27<br>beats the peer axes, loses people | 13/60 = 22%<br>hold 14 of 20<br>no | 14/60 = 23%<br>hold 14 of 24<br>beats the peer axes, loses people | 36/60 = 60%<br>hold 30 of 40<br>no |
| `l_full_bb02/epoch040` | 33/60 = 55%<br>hold 4 of 27<br>beats the peer axes, loses people | 8/60 = 13%<br>hold 10 of 20<br>no | 9/60 = 15%<br>hold 10 of 24<br>beats the peer axes, loses people | 35/60 = 58%<br>hold 16 of 40<br>no |
| `f_full_distil01/epoch020` | 10/60 = 17%<br>hold 0 of 27<br>no | 1/60 = 2%<br>hold 4 of 20<br>no | 1/60 = 2%<br>hold 4 of 24<br>no | 22/60 = 37%<br>hold 7 of 40<br>no |
<!-- /TABLE-NAMED -->

**`s_pseudo02_aug/epoch020` — the candidate the sweep picked — clears every gate at exactly
one configuration, and it is not the one the sweep scored.** At `go2-navigator-default` it
beats the incumbent on both peer axes and holds for 25 against 24: a clean pass. At the
sweep's own `mobilenet-ssd-trained`, where it was selected, it does not clear. At the peer
launcher's configuration it does not clear either — 45% recall against the incumbent's 50%,
and 9 held against 27.

Read that twice. The sweep selected a checkpoint through a configuration nothing runs; the
checkpoint turns out to be genuinely good, at a configuration nobody scored at and three
launchers can produce; and it is not good at either configuration a run has been recorded
through. None of that is visible from a single-configuration table, which is why there is
no longer a way to produce one by accident.

⛔ **Fine-tuning on this corpus costs the network its people, at every configuration**, and
the sweep's `people` column could not show it because it counted labels rather than the
gate. The count of checkpoints that hold for **nobody at all** — zero person-shaped boxes
in 284 frames — is in the headline table, per configuration.

⚠️ **"Fine-tuned checkpoints are silent at 224" is a Lite3 result and does not transfer
here.** [`evidence/2026-08-27-lite3-pov-clip-audit/`](../2026-08-27-lite3-pov-clip-audit/README.md)
measures every checkpoint of two Lite3 runs at **0 of 168** on the Lite3 clip at 224 px —
the new class emits no box at all, best score 0.000, while the same weights fire at
0.55–0.66 at 300. That is a real and serious finding about **that** class on **that**
clip.

It is not what the `go2wheel` fine-tunes do here. At 224 px they span **10% to 72%** peer
recall across the 800 — plenty of them well above the incumbent's 50%, and none of them
silent as a class. Two fine-tunes from the same trainer, one silent
at 224 and one not, is exactly the "non-monotonic in input size" the launch script's own
comment warns about — and repeating the Lite3 number as a general claim about fine-tuning
would be this issue's mistake in a new place. What both results support is the narrow
claim: **a checkpoint's behaviour at one square tells you nothing about the other, so it
has to be measured at the configuration it will be launched under.**

⛔ **A sub-threshold score is not a near miss.** Patch the `DetectionOutput` layer to 0.01
and the 224 px path reports a `person` in **535 of 535** frames of walk 3, including the
~300 that show only cardboard —
[`evidence/2026-08-27-89-runs-survived-14-can-be-dated/`](../2026-08-27-89-runs-survived-14-can-be-dated/README.md).
Below ~0.25 the score is noise, so "it almost fired" is not a reading anything here
supports, and a table built from patched-floor detections is a table about noise.

## Epochs 1–9 have never appeared in any table this project has published

Take the union of every epoch this project has published a row for: `sweep_all.json` holds
10/20/30/40, `wave6_wholeday.json` adds 15 and 25, and the checkpoint-sweep page's fine
pass over the winning run adds 17 and 22. That is **{10, 15, 17, 20, 22, 25, 30, 40}**, and
it is checkable from the two committed files. **No epoch below 10 has ever been scored, at
any configuration.**

That region is not empty. Every checkpoint that clears all four gates at the peer
launcher's configuration is **epoch 1, 2 or 3**, and the four that clear them at
`run-smoke`'s are epochs 9, 10 and 11 — all but three of those outside the published grid
entirely. It is not a general rule that early epochs win: at `go2-navigator-default` the
twelve that clear sit between epochs 12 and 37. The rule is the one this whole page is
about — **where the good checkpoints are depends on which configuration you ask about** —
and the published grid answered for a configuration nothing runs, on an epoch axis that
skipped the region where two of the three real answers live.

## Coverage — everything that was scored, and everything that was not

<!-- TABLE-COVERAGE -->
|  |  |
| --- | ---: |
| candidate checkpoints scored | **800** |
| `.caffemodel` files matching the inventory glob | 800 |
| not scored | **0** |
| configurations each was scored at | **4** |
| forward passes per checkpoint | 2 |
<!-- /TABLE-COVERAGE -->

`--inventory-glob` is what produces the "not scored" list: `score_crossday.py` is told what
*could* have been scored and records the difference in its own output, so a partial sweep
cannot be reported as a complete one. That is the error the 2026-08-26 sweep made when 627
of 640 checkpoints went unscored and unmentioned; the list here is empty, and it is empty
because it was computed rather than because nobody looked.

⚠️ **The count is 800, not the 804 that has been passed around.** `find ~/ssdft/runs -name
'*.caffemodel'` returns exactly 800 — twenty runs at forty epochs, no stragglers and no
odd-named extras — and `find ~/ssdft` returns 801, the extra being
`base/mnssd22.caffemodel`, which is scored too: the incumbent grown to 22 classes by
`detector/add_class.py`, i.e. epoch 0 of all twenty runs. The epoch grid is complete and
nothing was dropped to reach 800.

## Provenance

<!-- TABLE-PROVENANCE -->
| what | value |
| --- | ---: |
| frames scored | 284 |
| manifest sha256 | `1e13ffcd28ecc8f2…` |
| pixel digest | `95f18062c9f0aa2f…` |
| candidate prototxt sha256 | `c0e5b4cc70bf215f…` |
| incumbent weights sha256 | `761c86fbae3d8361…` |
| hold gate | person_detector.PERSON_ASPECT_MIN = 2.0 |
| prototxt DetectionOutput floor | 0.25 |
| the pass run by nothing, and its recorded reason | `mobilenet-ssd-trained` — 300 px at 0.25 is the pair the 2026-08-26 checkpoint sweep scored through and no launcher runs; it is here so the published tables can be lined up against configurations that are real |
<!-- /TABLE-PROVENANCE -->

**The frames are reproducible from a clone, and this is checked rather than asserted.** The
scoring host's 284 JPEGs and the 284 a clone decodes from the commit
`detector/labels/CROSSDAY.md` names have the same sha256-of-sha256s digest:

```
95f18062c9f0aa2f51072d03b8ab202bc4ad09c658340fc688d66acab625e9bc
```

That matters because `CROSSDAY.md` measured the same model at 56% and 60% false-positive
rate on the same frames extracted twice at different JPEG qualities — the extraction is
part of the measurement, so a digest is worth more than a frame count. The incumbent's rows
were then re-scored on a laptop, off a clone, against those locally decoded frames, and
came back identical.

**Two scorers agree cell for cell.**
`evidence/2026-08-26-detector-input-size/reproduce.py` implements the same rule
independently. Run against the frames a clone decodes it prints, on all three splits at
both squares:

```
split     size       peer recall      false alarms   person  +shaped  any-shaped
test       300       32/47 = 68%      76/136 = 56%       22        8          16
test       224       25/47 = 53%      45/136 = 33%       22        7           8
whole      300       41/60 = 68%     108/221 = 49%       56       32          40
whole      224       30/60 = 50%      57/221 = 26%       44       25          27
```

Every one of those is what `score_crossday.py` writes for `mobilenet-ssd-trained` and
`go2-peer-supervised`. Two implementations of one rule agreeing on twenty-four cells is the
check that neither is quietly measuring something else.

⚠️ **The incumbent's PROTOTXT is not byte-identical to the one the input-size page hashed,
and its weights are.** That page records the PINTO mirror's `d6ff8f177ceb…`; the copy on
the scoring host is `e781559c4f5b…`. Both declare `num_classes: 21` and
`confidence_threshold: 0.25`, and both produce the same rows, so the difference is not
behavioural — but it is a difference, and a page quoting one hash while measuring the other
would be this issue in miniature. The **weights** hash matches exactly.

**The confidence asymmetry, checked on real weights rather than argued.** A profile whose
floor is ABOVE the prototxt layer's is allowed here, on the argument that Python discards
the extra rows itself and the run really does measure what it asked for. Measured: the
`go2-run-smoke` profile (0.45) against the shipped 0.25 prototxt gives **8/60, 5/221,
hold 20, +shaped 20**, and against a prototxt patched to a 0.45 layer floor gives the same
four numbers. A profile whose floor is BELOW the layer's is refused, and the refusal is a
message and exit 2 rather than a traceback:

```
REFUSED
profile 'go2-peer-supervised' asks for confidence 0.25 but this prototxt's DetectionOutput
layer already discards everything below 0.45 inside forward(). The boxes between the two do
not exist by the time Python sees the output, so the run would measure 0.45 and label it
0.25. Lower the prototxt floor with person_detector.prototxt_with_floor, or raise the
profile.
```

**The scripts that ran are the scripts in this commit, byte for byte.** Both were synced to
the scoring host and their sha256 compared before the run started, rather than assumed:

```
764a8c44cacd811e…  detector/score_crossday.py
a542020f6de00234…  robot-stack/unitree/go2/visual_nav/inference_profile.py
```

That is worth doing rather than waving at, and this page is the reason it is: three earlier
attempts at this sweep were discarded because the script changed underneath them — once for
the output format, once when `--classes` turned out to belong in the configuration, and once
for a `try`/`except` boundary that could not have changed a number and was still not worth
publishing an unverifiable claim about.

⚠️ **`cv2.dnn.Net` keeps state across a change of input blob size, and this scorer loads a
fresh one per square because of it.** Measured on the shipped weights and this corpus:
scoring 300 px on a Net that has already run 224 px gives 42/60 peer recall and hold 41,
where a Net that has only ever seen 300 gives 41/60 and hold 40. Same weights, same frames,
same threshold — one frame of difference that depends on the order the sizes were swept in.
`evidence/2026-08-26-detector-input-size/reproduce.py` loops sizes on one Net; its numbers
match this page because it sweeps 300 first, but the method is order-dependent and should
not be copied.

⚠️ **The eval is cross-day, and the check is unchanged from #129.** Training is the Aug-24
corpus (`--images $HOME/go2-peer-dataset-20260824`, `--negatives-glob "$D/neg_*.jpg"`);
evaluation is seven clips shot 2026-08-20, split by clip rather than at random, which
matters at 7 fps where neighbouring frames are near-duplicates. No Aug-20 pixel enters
training and `run_wave6.sh` names no other image source. The one step that is inference
rather than inspection is unchanged too: `--composite 0.3` pastes the peer onto a peer-free
frame, and that flag does not exist in `detector/finetune_ssd.py` on `main`, so which
directory it draws backgrounds from cannot be read from this repository.

## What this does not settle

* **Three of the four launchers are not in this repository**, so nothing here can make them
  ask for their configuration the way `run-peer-supervised.sh` now does. They are declared
  by hand and bound to `dashboard/run-profile.example.json`, which is the one copy of
  `run-smoke.sh`'s invocation a clone holds. **A launcher nobody can read is a launcher
  nothing can reconcile**, and that is the next change, not this one.
* **Reconciling the robot to one configuration is a robot decision, not a tooling one.**
  Nothing here changes what any launcher computes. `visual_nav.py`'s parser default is left
  at 300/0.4 deliberately: changing it changes what a bare run computes. The registry makes
  the disagreement visible and testable; it does not resolve it.
* **No candidate has been run on a robot**, and the weights are still not fetchable from a
  clone — #129's third objection, untouched. Twenty-two checkpoints clear every gate at one
  of the three deployed configurations; **none clears at two**, so "which one would you
  ship" is still a question about which launcher, and that is a decision this measurement
  informs rather than makes.
* **This is not a retraining recommendation.** Nothing here was trained.
* **The launch-script change has not run on a robot.** It is argued the same argv by two
  tests: one compares the derived flags against the literals the file carried at `4d79b45`,
  the other puts them through the real `visual_nav.build_parser()` and gets 224 px,
  confidence 0.25 and twenty classes. Neither is a run. **A tree deployed before this
  change does not carry `inference_profile.py`, and the script will stop with an error
  rather than launch**; redeploy with `deploy/push-to-robot.sh` before the next peer run.

## Filed separately, not fixed here

**`eval_detector.py --mask-overlay` masks the wrong regions below 1920x1080.** Its
`RADAR_REGION = (0.834, 0.0, 1.0, 0.287)` is a *fraction*, and the panel it is meant to
cover is a *fixed pixel size*: `overlay.draw_plan_view` draws 300x300 at
`(width - 316, 16)`, and `visual_nav.py` calls it with that default. The two line up only
at the resolution the fractions were fitted at. Derived from those two facts alone:

| footage | mask ∩ panel | panel left visible |
| --- | ---: | ---: |
| 1920x1080 | 88,200 of 90,000 px² | **2%** |
| 1280x720 | 37,457 of 90,000 px² | **58.4%** |

[`evidence/2026-08-27-lite3-pov-clip-audit/`](../2026-08-27-lite3-pov-clip-audit/README.md)
reports the same 58.4% from its own measurement of the pixels; this derivation was done
independently from the two constants and agrees to 27 px². So any figure quoted as
"masked" on 720p footage was measured with most of the radar inset still in frame. That
needs its own change and a decision about already-published numbers; it is flagged here
rather than silently corrected inside this one.

## Continues in

* [#129](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/129) — the defect
  this closes, and the correction to its own premise.
* [`evidence/2026-08-27-89-runs-survived-14-can-be-dated/`](../2026-08-27-89-runs-survived-14-can-be-dated/README.md)
  — what the recorded runs actually computed, and how it had to be established.
* [`evidence/2026-08-26-detector-input-size/`](../2026-08-26-detector-input-size/README.md)
  — the measurement that started this.
* [`evidence/2026-08-26-checkpoint-sweep/`](../2026-08-26-checkpoint-sweep/README.md) — the
  sweep whose rankings this supersedes.
