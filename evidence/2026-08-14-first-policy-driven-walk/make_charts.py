#!/usr/bin/env python3
"""Figures for the gait-floor finding. Palette: dataviz reference slots 1 and 2,
validated all-pairs light (CVD dE 24.7, normal 33.6, contrast pass)."""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
S1 = "#2a78d6"   # slot 1 — 0.21 m/s
S2 = "#eb6834"   # slot 2 — 0.35 m/s
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})


def legs(path, tmax=24.0):
    """Per-half-second knee swing, forward speed and body height, from the encoders."""
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    rows = [x for x in rows if "odom" in x and x["t"] <= tmax]
    buckets = {}
    for x in rows:
        buckets.setdefault(round(x["t"] * 2) / 2.0, []).append(x)
    out, prev = [], None
    for k in sorted(buckets):
        xs = buckets[k]
        o0, o1 = xs[0]["odom"], xs[-1]["odom"]
        moved = math.hypot(o1["x"] - o0["x"], o1["y"] - o0["y"])
        step = moved if prev is not None else 0.0
        sway = sum(math.degrees(max(y["q"][n] for y in xs) - min(y["q"][n] for y in xs))
                   for n in ("FR_calf", "FL_calf", "RR_calf", "RL_calf")) / 4
        out.append({"t": k, "swing": sway, "speed": step / 0.5,
                    "height": xs[-1]["sport"]["body_height"]})
        prev = k
    return out


def since_standing(series, settle_s=0.5):
    """The WALKING window only, re-zeroed on the stand.

    Two things have to be cut or the chart lies. The 70-plus-degree stand-up would
    dominate both panels and squeeze the walking phase flat, so time is re-zeroed on
    the stand and its tail is dropped. And the run ends by LYING DOWN, which is another
    large knee swing — left in, the stalled run appears to start walking again at the
    end. Everything from the start of the park is dropped.
    """
    up = next((r["t"] for r in series if r["height"] > 0.30), series[0]["t"]) + settle_s
    walk = [dict(r, t=r["t"] - up) for r in series if r["t"] >= up]
    park = next((i for i, r in enumerate(walk) if r["t"] > 2 and r["height"] < 0.20),
                len(walk))
    return walk[:park]


B3 = since_standing(legs("stalled-run-leg-encoders.jsonl"))
B6 = since_standing(legs("hero-run-leg-encoders.jsonl"))

# ── Figure 1 — the mechanism ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4), sharex=True,
                         gridspec_kw={"hspace": 0.28, "wspace": 0.18})
cols = ((B3, "0.21 m/s commanded  —  stalled", S1),
        (B6, "0.35 m/s commanded  —  arrived", S2))

for col, (data, title, colour) in enumerate(cols):
    t = [r["t"] for r in data]
    for row, (key, ymax) in enumerate((("swing", 32), ("speed", 0.30))):
        ax = axes[row][col]
        ax.plot(t, [r[key] for r in data], color=colour, lw=2.0,
                solid_capstyle="round")
        ax.fill_between(t, 0, [r[key] for r in data], color=colour, alpha=0.10, lw=0)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if col == 1:
            ax.tick_params(labelleft=False)
    axes[0][col].set_title(title, color=INK, fontsize=10, fontweight="bold", pad=8)

axes[0][0].set_ylabel("knee swing (deg)\nmean of four legs", color=INK2)
axes[1][0].set_ylabel("forward speed (m/s)\nfrom odometry", color=INK2)
for ax in axes[1]:
    ax.set_xlabel("seconds since the robot finished standing", color=INK2)
    ax.xaxis.set_major_locator(MultipleLocator(2))

# Label what actually differs. BOTH runs settle for ~3 s after standing before the
# gait starts, so that flat opening is not the finding and must not be labelled as it.
# The finding is what happens AFTER: one sustains, the other decays to nothing.
axes[1][0].annotate(
    "a few shuffles, then\ndecays to a standstill\nwhile still commanded",
                    xy=(7.2, 0.01), xytext=(5.6, 0.15), color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axes[1][1].annotate("sustained 0.20–0.26 m/s", xy=(8.0, 0.235), xytext=(4.4, 0.075),
                    color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
for ax in axes[:, 0]:
    ax.axvspan(0, 3.3, color=MUTED, alpha=0.07, lw=0, zorder=0)
for ax in axes[:, 1]:
    ax.axvspan(0, 3.3, color=MUTED, alpha=0.07, lw=0, zorder=0)
axes[0][0].text(1.65, 29, "settling\n(both runs)", ha="center", va="top",
                fontsize=7.5, color=MUTED)

fig.suptitle("Below ~0.35 m/s the Go2 does not produce a gait — it stands still",
             x=0.5, y=0.985, fontsize=12, fontweight="bold", color=INK)
fig.text(0.5, 0.925, "Same policy, same robot, same lane. Only the commanded speed "
                     "differs. Joint encoders, independent of the state estimator.",
         ha="center", fontsize=9, color=INK2)
fig.subplots_adjust(top=0.845, left=0.115, right=0.985, bottom=0.11)
fig.savefig("gait-floor.png", dpi=170)
print("wrote gait-floor.png")

# ── Figure 2 — distance travelled before stopping ───────────────────────────
runs = [("MAPPO, run 1", 0.34, S1), ("MAPPO, run 2", 0.38, S1),
        ("MAPPO, run 3", 0.43, S1), ("planner, run 1", 0.50, S1),
        ("planner, run 2", 0.57, S1),
        ("planner", 2.58, S2), ("MAPPO (hero run)", 2.78, S2)]

fig2, ax = plt.subplots(figsize=(9.6, 3.9))
ys = list(range(len(runs)))[::-1]
for y, (label, dist, colour) in zip(ys, runs):
    ax.barh(y, dist, height=0.62, color=colour, zorder=3)
    ax.text(dist + 0.05, y, f"{dist:.2f} m", va="center", ha="left",
            fontsize=9, color=INK, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in runs], fontsize=9)
ax.set_xlim(0, 3.25)
ax.set_xlabel("distance travelled before the run ended (m)", color=INK2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

ax.axvline(2.6, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(2.6, len(runs) - 0.25,
        "  arrival circle\n  (stops 0.8 m short\n  of the chair)",
        color=INK2, fontsize=8, va="top", linespacing=1.35)

ax.text(3.18, 5.0, "commanded 0.21 m/s", ha="right", fontsize=9,
        color=S1, fontweight="bold")
ax.text(3.18, 1.45, "commanded 0.35 m/s", ha="right", fontsize=9,
        color=S2, fontweight="bold")

fig2.suptitle("Five stalls at 0.21 m/s across two controllers; both runs at 0.35 m/s "
              "reached the goal", x=0.5, y=0.97, fontsize=12, fontweight="bold",
              color=INK, ha="center")
fig2.subplots_adjust(top=0.83, left=0.155, right=0.99, bottom=0.145)
fig2.savefig("stall-distance.png", dpi=170)
print("wrote stall-distance.png")
