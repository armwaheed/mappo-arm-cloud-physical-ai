<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Coding agent guidelines for robotics development

How we drive coding agents on this demo. Tool-agnostic — Codex, Qwen Code, Claude Code,
Cursor, whatever you run. Nothing here is a slash command or a vendor feature.

**The one idea:** *the agent session is disposable, the GitHub issue is the memory.*
Every session ends by running out of context, by you closing the laptop, or by you
switching tools. Anything the next session needs must be in the issue before that
happens — not in your scrollback, not in the model's head, not in Slack.

---

## The loop

```
issue  ──▶  one-line prompt pointing at it  ──▶  agent works  ──▶  evidence (run, log,
  ▲                                                                 video, test count)
  │                                                                        │
  └──── continuation comment: results, what broke, what's next ◀───────────┘
```

1. **Write the issue first.** Requirements, blockers, measurements, options. This is the prompt.
2. **Start the session with a URL**, not a paragraph. `Please continue from GitHub issue: <url>`
3. **Interrupt early and often.** A wrong run is cheaper to stop than to review.
4. **End with a continuation comment**, always, even mid-thought. Then close or open the next issue.

A fresh agent, a different vendor's agent, or a colleague on the other side of the world
picks up from step 2 with no handover call.

Live exemplars — read one before you write your first issue. In this repo,
**[issue #3](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/3)**,
**[issue #4](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/4)** and
**[issue #5](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/5)** are written
in this shape, and
**[issue #4](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/4)** is the one
to copy. The upstream Go2 stack repository
(located in `PROVENANCE.md`) carries a matching pair — an issue written to be executed, and
the continuation comment that handed off cleanly to the next one. Ask @armwaheed for access.

---

## Four prompt moves

| move | when | say it like this |
| --- | --- | --- |
| **Point at the issue** | starting or resuming any real work | `Please continue from GitHub issue: <url>` — plus the state of the room (below) |
| **Challenge me** | you have already formed a technical opinion | *"…the Go2 should lie down, recenter the arm, lock it, then resume. **Challenge me**"* |
| **Ranked options** | you have *no* opinion and the decision is yours to own | `Give me a selection list of choices, cheapest first, with a recommendation` |
| **Adversarial pass** | every single time code changes | the [standing suffix](#the-standing-suffix) below |

**On "challenge me".** It is not politeness, and it is not "review my thinking". It is
attached to a *specific, falsifiable proposal* so the agent has something to attack:
*"I suspect this PR should just be closed, won't fix. Challenge me."* Then actually take
the loss — *"Agree with your challenge, just make a README with no code"* is a normal
outcome. Two useful variants:

- **Non-blocking:** `Challenge me. In the meantime keep working.` You get the argument without stalling the run.
- **Adversarial framing:** state your position as strongly as you can, then `please challenge me, as maybe I am missing something`.

**On ranked options.** Ask for them when the tradeoff is expensive to reverse — a schema,
a frame convention, a branch strategy. Insist on *cheapest first* and on a recommendation;
then push back inside the list rather than accepting it whole.

---

## The standing suffix

Paste this at the end of any prompt that will touch code. Verbatim — it is the single most
used sentence in this workflow:

```
Once you are done, do a pass on the code for software engineering best practices,
mistakes, sloppy or brute force algorithms, inconsistent style, incorrect comments, etc.
```

Repo-specific additions for this project:

```
Both integration/ and robot-stack/unitree/go2/visual_nav/ carry a ruff.toml —
`ruff check .` must be clean in each.
Run the tests: cd integration && for t in test_*.py; do python3 $t; done
Anything that moves a leg is governed by robot-stack/SAFETY.md.
The product is "Arm Device Connect" / "DC". The naming rule in AGENTS.md is absolute —
read it before writing prose, and before pasting anything in from another repository.
```

---

## Writing an issue an agent can execute

Not a ticket. A brief. Six parts, in this order:

1. **Lineage.** *"Continuation of #9, which is done ([PR #10](…), merged)."* Where the state came from.
2. **Target scenario, concretely.** *"Goal = a chair. Obstacle = a blue recycling bin."* Props, distances, success condition.
3. **⛔ Read this first — the known blocker.** The thing that makes this harder than it looks, quoted from the code or docs. *"This is new capability, not a new test."*
4. **What was already measured.** A table. Detector hits, scores, latencies, temperatures. Measurements you take *before* opening the issue save the agent an hour of rediscovery and stop it inventing a premise.
5. **Options, cheapest first.** With the cost of each spelled out, and the "swap the prop instead" option included.
6. **Open questions**, phrased so the closing comment can answer them one by one.

Write it so a competent stranger with no context could execute it. That stranger is your
agent — and it is also Sagar, Jackie, David, Odin, and Deep Robotics.

## Writing a continuation comment

The other half, and the half people skip. Four parts:

1. **Outcome table.** Numbers, not adjectives. Distance, cycle count, errors, temperatures, reason mix.
2. **Answers to the open questions in the body** — including *"the premise was wrong: measured median 285 ms, not the ~160 ms the docstring claimed."*
3. **🔴 What the run found.** The defect the work exposed, with the live observation that exposed it.
4. **Continues in #N.** Link forward, then close.

Trigger phrases that should always produce one: *"we are done for today"*, *"let's stop
here and ship it"*, *"good is good enough"*, *"we need a fresh session"*.

---

## Robotics-specific: the room is not in the repo

Your agent cannot see the lab. Four things belong in the prompt, every time:

- **The state of the room.** *"I positioned a chair as the goal, about 4.5 m ahead. There is a blue recycling bin about 3 m ahead as an obstacle."* When you move something, say so: *"I turned the bin so its narrow end is facing you. The lane is no longer too narrow."*
- **The authority boundary, explicitly.** *"Please ask me before you do any walking. You can move the back-mounted arm and use all of the sensors without my permission."* Do this in the first prompt, not after something moves.
- **Evidence artefacts, requested up front.** *"Record the video stream from the robot and show me absolute filepaths of the mp4s after, so I can select a hero video."* Terminal-rendered paths you cannot click are useless — ask for absolute paths, or for output written to a file.
- **The synchronisation point.** *"Prompt me when you're ready to run the test, so I can record it."* / *"Robot repositioned, I'm standing 2.5 m in front, go."*

And two habits that separate a demo from a claim:

- **Every measurement gets a paired control.** Same ticks, obstacle removed. Without it, *"the policy steered 36° off the goal bearing"* is not evidence of anything — this checkpoint carries a 6–16° heading bias with an empty scene.
- **Prefer the limit case first.** Run the extreme setting before you build the careful version; it falsifies faster and costs less.

## ⛔ Never in a prompt, an issue, or a commit

Robot SSH passwords, WiFi PSKs, tokens. We have had to scrub these out of a GitHub issue
after the fact, and "as best as you can" is not a security control — the value is in the
event log for good. Put them in a local untracked file and reference it:

```
You can access the robot at ssh unitree@192.168.123.18 — credentials in
~/.robot-creds (not in this repo, not in the issue).
```

Same rule for short-lived `?token=` asset URLs: merge the PR first, then link the
permanent raw URL from the default branch.

---

## Tool-agnostic notes

| | |
| --- | --- |
| **Standing rules** | `AGENTS.md` at the repo root. Codex and Qwen Code read it; Claude Code reads `AGENTS.md` or `CLAUDE.md`. If your tool reads neither, paste its contents into your first prompt. |
| **Context loss** | Every agent hits a context limit and summarises. Assume the summary drops detail. The issue is the durable copy — write to it *before* you hit the wall, not after. |
| **Model choice** | Nothing here depends on it. The four moves work on Codex, Qwen and Claude because they are instructions to a reader, not tool features. |
| **Parallelism** | Ask for independent tracks to run concurrently, and for a laid-out plan rather than a forced single choice. |
| **Issues over chat** | If it was decided in Slack or on a call, it did not happen. Restate it in the issue, or the next session will contradict it. |

## What not to copy

- **Don't open a PR without evidence in it.** A run log, a test count, a video, or a measurement table. "It should work" is not a status.
- **Don't let the agent grade its own homework silently.** The adversarial pass is a separate instruction for a reason — ask for it explicitly, and read the findings before saying "fix them all".
- **Don't accept a green result you did not verify runs.** A check that never executes, a latch proved by "the joints stopped moving" on an unpowered arm — both happened here. Ask what would make the test fail, and then make it fail.
- **Don't write the issue after the work.** Then it is a report, and nobody can start from it.
