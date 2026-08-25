# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check a track's apparent growth against the robot's own odometry, and drop the liars.

A class-agnostic detector — every VOC class counts as an obstacle, because the stock
21-class MobileNet-SSD read that way finds the peer robot where a day of fine-tuning
could not — buys recall by giving up the one thing :func:`person_detector.estimate_range`
needs: **which object it is, and therefore how big it is**. Range comes from a size
prior. Get the prior wrong by a factor and the range is wrong by the same factor, and
nothing in the single-frame pipeline can tell.

That error is not symmetric, and only one half of it costs anything. Write
``k = R_reported / R_true = size_prior / true_size``:

  * ``k > 1`` — the prior is too BIG for the thing (a 1.70 m person prior on a 0.40 m
    peer robot: ``k = 4.25``). The obstacle is reported FARTHER than it is. Dangerous,
    but not a false alarm; nothing below addresses it.
  * ``k < 1`` — the prior is too SMALL (a person prior on a door frame, a cabinet run,
    a wall). The obstacle is reported NEARER than it is, lands inside the planner's
    2.5 s horizon, and the robot stops for something that was never in the way.

``k < 1`` is what this module removes, and the observable is ego-motion. An object that
is really out there has a range that must shrink at the rate the robot's own odometry
demands. A box drawn around the far wall does not.

## The measurement

Range from a size prior is exactly proportional to true range — ``R_reported(t) =
k · R_true(t)`` with ``k`` constant for a rigid object under a fixed prior — so the
*logarithmic* rate is prior-free::

    d(ln R_reported)/dt  =  d(ln R_true)/dt  =  -v_closing / R_true  =  -k · v_closing / R_reported

The gate predicts the second-to-last form using the position the track is *claimed* to
occupy, held fixed in odom, and the robot's own pose sequence. No camera model, no
velocity input, no size prior: two range readings and two robot poses. Dividing observed
by predicted estimates ``k`` — reported as :attr:`Verdict.scale`, with the caveats
there — but the DECISION is taken on the difference, which is zero exactly when the two
series agree.

## Why this is a filter and NOT an avoidance signal

Settled by measurement, and stated here so it is not re-derived. For a CROSSING peer,
``dR/dt`` is zero at closest approach, so growth is zero — below any noise floor — and
at the frame edge the box jitter is 6.86%/frame against a 2.50% signal (SNR 0.36).
Expansion cannot tell you to dodge. It can only tell you that something you were about
to dodge is not where you think. Use it to drop tracks, never to keep them.

## Everything below is one-sided, and that is the safety property

A track is dropped only when its range shrinks **more slowly** than ego-motion demands.
Shrinking FASTER — the obstacle is nearer than reported, or is coming at you — raises no
verdict at all. The gate can therefore only ever discard a threat it has over-estimated.

## Measured constants

All from 2,800 frames of the robot's own camera on the peer robot, 2026-08-24, run
through the robot's own weights and the repo's own :func:`estimate_range`.

  * :data:`LOG_RANGE_SIGMA` — 3.07%, the per-sample scatter of ``ln R`` over 1,133
    consecutive detection pairs, **with the ranging source held**. The median step is
    0.94% and the underlying box height moves 0.68%; the sigma is larger than either
    because the distribution has a tail (p95 = 10.6%).
  * The tail that is NOT in that number, and the reason :meth:`ExpansionConsistency.observe`
    restarts its window on a source change: when ``estimate_range`` switches between the
    height prior, the width prior and the frame-fill constant, the median step is
    **103%** — a factor of 2.8, from one frame to the next, on a parked robot. That
    happened on 208 of 1,471 frame pairs (14.1%). Fitting a growth rate across one of
    those measures the estimator changing its mind, not the world moving.
  * :data:`REJECT_SIGMAS` = 4.0 from the measured tail, not from a Gaussian. On the
    parked null the one-sided exceedance of ``+4 sigma`` is 1 window in 595 at 8
    samples and 0 of 290 at 20 samples. ``+3 sigma`` is 0.29-0.50%, which is three
    times what a Gaussian would give — the noise is heavier-tailed than normal, so a
    sigma count read off a table would have been wrong.

## What is NOT established

**The gate has never run against a moving robot.** Every clip in that corpus is a
parked robot: phase-correlating consecutive frames gives a net camera displacement of
at most 14.7 px at 480-wide over an entire clip, sub-pixel per frame. So the corpus
measures the NOISE — which is what sets the threshold — and can say nothing at all
about the SIGNAL. The evaluation this needs is a walking sequence past a structure the
detector fires on. Until that exists this is an instrument with a calibrated noise
floor and no demonstrated catch.

**A mis-scaled ghost and a retreating real obstacle are the same measurement.** Both
make the range shrink more slowly than ego-motion demands, and one monocular range
series cannot separate them. Three things bound the harm rather than remove it: the drop
additionally requires the observed rate to put contact beyond :data:`TAU_HORIZON_S`, so
the gap is barely closing either way; the track is RETAINED in the tracker, so a target
that turns round and starts closing properly releases the rejection on that positive
evidence — note it takes a window's worth of it, not one frame, because the latch is
deliberately hard to leave; and :data:`REJECT_SIGMAS` is set where the measured null
never fired.
What would actually separate them is bearing parallax, which is immune to radial target
motion — and degenerate straight ahead, which is where obstacles matter. That is a
different piece of work and it is not this one.

Pure stdlib. ``python3 test_expansion.py``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from tracker import UNMEASURED_SOURCES

#: Per-sample 1-sigma scatter of ``ln R``, with the ranging source held. MEASURED:
#: 3.07% over 1,133 consecutive detection pairs from 100 source-homogeneous runs on
#: the 2026-08-24 peer corpus, ranged by the repo's own ``estimate_range``. The
#: robot is parked throughout, so this is noise and nothing else.
LOG_RANGE_SIGMA = 0.031

#: How far above the ego-motion prediction an observed rate must sit before the track
#: is dropped. 4.0 rather than the customary 3.0 because the null is NOT Gaussian: the
#: measured one-sided exceedance at +3 sigma is 0.29-0.50% against the 0.13% a normal
#: distribution predicts, and at +4 sigma it is 0.17% at 8 samples and 0 of 290 at 20.
REJECT_SIGMAS = 4.0

#: Time-to-contact beyond which an obstacle cannot constrain this cycle's plan. The
#: planner rolls :data:`avoidance.PlannerConfig.horizon_s` = 2.5 s, and the tracker
#: only believes the detector to ``max_range_m`` = 6.0 m, which at the 0.35 m/s gait
#: floor is 17 s away. 8.0 sits between the two: comfortably past anything the planner
#: can reach, comfortably inside anything the tracker will hold.
#:
#: This is the SAFETY half of the drop condition. Being inconsistent with ego-motion is
#: not on its own a reason to discard an obstacle — an obstacle can be inconsistent and
#: still be closing. The observed rate must ALSO say contact is this far off.
TAU_HORIZON_S = 8.0

#: Samples kept per track. 30 at the perception thread's ~7 Hz is ~4.3 s, and the
#: whiteness of the noise was only checked out to 3 s (measured window-slope scatter
#: within 0.81-1.15x of the independent-sample prediction over 0.7-3.0 s), so a longer
#: window would be extrapolating the noise model rather than using it.
WINDOW_SAMPLES = 30

#: Below this many samples the fit is not attempted. NOT a sensitivity knob — the two
#: power tests do that job, and they do it from the actual sample times rather than from
#: a count, so this is only a floor under the arithmetic.
#:
#: It is deliberately set BELOW where the gate can actually decide anything. At the
#: perception thread's ~7 Hz, 8 samples give a slope sigma of 0.033/s, and
#: ``REJECT_SIGMAS`` times that is more than the ``1/TAU_HORIZON_S`` the drop has to
#: clear — so an 8-sample window always comes back UNRESOLVED, with a reason that says
#: which precision it was short of. Encoding the real floor here instead would state it
#: twice, in units (a count) that stop being true the moment the frame rate moves.
MIN_SAMPLES = 8

#: How far the PREDICTION may itself depart from the straight line it is fitted with,
#: in multiples of :data:`LOG_RANGE_SIGMA`, before the window is trimmed.
#:
#: ⚠️ ONE OF THE TWO GUARDS THAT KEEP THE VERDICT HONEST — this one makes the
#: comparison VALID, and :meth:`ExpansionConsistency._latch` makes it MONOTONE. Neither
#: is sufficient alone, and it took both to get the reversal sweep to zero.
#: The whole test is one straight-line fit against another, which is only meaningful
#: while the predicted series IS a straight line. ``ln|anchor - robot|`` is not: it
#: falls to the closest approach and RISES, and it is steeply curved on either side of
#: that turn. Fit across it and the prediction flattens, the drop condition stops being
#: met, and a ghost correctly rejected earlier in the walk comes back CONSISTENT later.
#: More evidence, weaker conclusion.
#:
#: Two GEOMETRIC guards were tried before this and both left reversals behind, which is
#: why it is stated as a residual instead. Refusing a predicted range of zero or less can
#: never fire, since ``hypot`` is non-negative — a robot 1.3 m short of the anchor and
#: one 1.3 m past it are the same number. Requiring the newest sample to be the closest
#: approach, plus a floor under the predicted range, cut a 124-case reversal sweep to 86
#: rather than to 0. Measuring the departure directly catches every version at once:
#: walking past, arriving on top of it, and plain steep curvature near the end.
#:
#: And it is still NOT ENOUGH BY ITSELF, which is why the latch exists. Trimming a
#: strongly curved series short enough makes it look straight — measured, an 8-sample
#: tail of a prediction spanning a factor of 2 in range has a residual of 0.015, well
#: inside this limit — so the guard eventually passes a window it should have refused,
#: just a shorter one. A shorter window is a wider sigma, and a wider sigma is what
#: reverses the verdict.
#:
#: 3.0 from the measured spread. A robot at 0.35 m/s closing on a 2.0 m claim over a
#: 30-sample window bends its prediction by 2.0 sigma and a 3.0 m claim by 0.55; the
#: pathological cases sit at 10-34.
PREDICTION_CURVATURE_SIGMAS = 3.0

#: A gap longer than this ends the window. The perception thread runs at ~7 Hz; a
#: second of silence means the track coasted, and the samples either side of the gap
#: were not taken of the same continuous approach.
MAX_GAP_S = 1.0

#: Verdict states. ``UNRESOLVED`` is not a soft ``CONSISTENT``: it means the evidence
#: to decide does not exist yet, and a caller must treat it exactly as it treats a
#: track with no gate at all.
CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Verdict:
    """What the gate can say about one track, and why.

    ``scale`` estimates ``R_reported / R_true`` — the observed rate over the predicted
    one. ⚠️ **It is a diagnostic and NOT a correction factor.** The identity it comes
    from, ``d(ln R_reported)/dt = d(ln R_true)/dt``, is instantaneous, and both rates
    here are straight-line fits to curves; the ratio is exact only in the limit of small
    travel, and its error grows with how much of the claimed range the window closed.
    What is pinned by test, and what a caller may rely on, is the DIRECTION: the error
    always runs AWAY from 1.0, and it is exactly 1.0 when the prior is right whatever
    the curvature. So it is a bound and not a value — a rejected track is *at least*
    ``1/scale`` times further off than it says.

    No decision reads it. The verdict is decided on the rate DIFFERENCE, which is zero
    exactly when the two series agree and therefore carries no such bias.
    """

    state: str
    reason: str
    samples: int
    span_s: float
    observed_rate: float | None = None       # d(ln R)/dt, 1/s. Negative = closing.
    predicted_rate: float | None = None      # the same, if the track were where it says
    sigma_rate: float | None = None          # 1-sigma on the observed rate
    scale: float | None = None               # R_reported / R_true

    @property
    def rejected(self) -> bool:
        return self.state == INCONSISTENT


@dataclass(frozen=True)
class _Sample:
    """One range reading, with the two odom points needed to predict its successor."""

    time_s: float
    range_m: float
    log_range: float
    source: str
    odom_xy: tuple                  # where the track is CLAIMED to be, from this reading
    robot_xy: tuple                 # where the robot was when it took the reading


def _slope(times: list, values: list) -> tuple:
    """Least-squares ``d(values)/d(times)`` and the fit's ``Sxx``.

    ``Sxx`` is returned rather than a standard error because the caller needs it twice:
    the slope's variance is ``sigma^2 / Sxx``, and a zero ``Sxx`` is the degenerate case
    where every sample shares a timestamp. Computed from the ACTUAL sample times, not
    from a count times a nominal period — the frame interval on this camera ranges over
    50-142 ms, and a uniform-spacing formula would misstate the confidence by the ratio.
    """
    n = len(times)
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    sxx = sum((t - mean_t) ** 2 for t in times)
    if sxx <= 0.0:
        return 0.0, 0.0
    sxy = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    return sxy / sxx, sxx


def _straightness(times: list, values: list) -> float:
    """RMS departure of ``values`` from its own least-squares line.

    How far the series is from being the straight line the comparison assumes it is.
    See :data:`PREDICTION_CURVATURE_SIGMAS` — this is applied to the PREDICTION, whose
    curvature is geometry rather than noise and is therefore knowable in advance.
    """
    n = len(times)
    slope, sxx = _slope(times, values)
    if sxx <= 0.0:
        return math.inf
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    intercept = mean_v - slope * mean_t
    return math.sqrt(sum((v - (slope * t + intercept)) ** 2
                         for t, v in zip(times, values)) / n)


class ExpansionConsistency:
    """Per-track ego-motion consistency of the reported range.

    Args:
        log_range_sigma: per-sample scatter of ``ln R``. See :data:`LOG_RANGE_SIGMA`;
            override only against a re-measured corpus, never to make the gate fire.
        reject_sigmas: see :data:`REJECT_SIGMAS`.
        tau_horizon_s: see :data:`TAU_HORIZON_S`.
        window_samples: see :data:`WINDOW_SAMPLES`.
        min_samples: see :data:`MIN_SAMPLES`.
        curvature_sigmas: see :data:`PREDICTION_CURVATURE_SIGMAS`.

    Feed it with :meth:`observe` once per track per accepted measurement and read
    :meth:`verdict`. It holds no reference to the tracker and does not delete anything:
    a rejected track keeps living, keeps being measured, and can be restored. Dropping
    it from the obstacle set is the caller's decision — see ``visual_nav._obstacles``.
    """

    def __init__(self, log_range_sigma: float = LOG_RANGE_SIGMA,
                 reject_sigmas: float = REJECT_SIGMAS,
                 tau_horizon_s: float = TAU_HORIZON_S,
                 window_samples: int = WINDOW_SAMPLES,
                 min_samples: int = MIN_SAMPLES,
                 max_gap_s: float = MAX_GAP_S,
                 curvature_sigmas: float = PREDICTION_CURVATURE_SIGMAS) -> None:
        if min_samples < 3:
            raise ValueError(f"min_samples must be at least 3, got {min_samples}")
        if window_samples < min_samples:
            raise ValueError(f"window_samples {window_samples} is below min_samples "
                             f"{min_samples}: the gate could never resolve anything")
        if curvature_sigmas <= 0.0:
            raise ValueError("curvature_sigmas must be positive, got "
                             f"{curvature_sigmas}")
        self._sigma = float(log_range_sigma)
        self._reject_sigmas = float(reject_sigmas)
        self._tau_horizon_s = float(tau_horizon_s)
        self._window = int(window_samples)
        self._min_samples = int(min_samples)
        self._max_gap_s = float(max_gap_s)
        self._curvature_sigmas = float(curvature_sigmas)
        self._history: dict = {}
        self._verdicts: dict = {}
        self._latched: set = set()

    # ── Feeding ─────────────────────────────────────────────────────────────
    def observe(self, track_id: int, *, time_s: float, range_m: float, source: str,
                odom_xy: tuple, robot_xy: tuple) -> Verdict:
        """Record one measurement of ``track_id`` and return the resulting verdict.

        ``odom_xy`` is where this reading places the track — ``Observation.x, .y``,
        already computed by the tracker, so no geometry is repeated here. ``robot_xy``
        is the pose that reading was taken from. The two together are the whole input:
        holding ``odom_xy`` fixed and walking ``robot_xy`` gives the range series a
        real object at that claimed position would have produced.

        THREE THINGS RESTART THE WINDOW rather than being averaged into it, because
        each makes consecutive samples incomparable:

          * a change of ranging ``source``. Measured on the 2026-08-24 corpus: a
            source switch moves the reported range by a median of 103% between
            consecutive frames, on a PARKED robot, and it happened on 14.1% of frame
            pairs. That is the estimator changing prior, not the world moving, and a
            fit spanning one measures the switch.
          * a source that reports a constant (:data:`tracker.UNMEASURED_SOURCES`).
            ``frame-fill`` returns a fixed 0.8 m and ``width-capped`` returns the fit
            range; neither moves when the robot does. Their observed rate is exactly
            zero however hard the robot is closing, so every one of them would look
            like a total ghost and be dropped — the failure this clause exists to
            prevent, and the one that would have hit hardest, since a frame-fill box
            is by definition an obstacle filling the frame.
          * a gap over ``max_gap_s``, or time running backwards.
        """
        history = self._history.get(track_id)
        if history is None:
            history = deque(maxlen=self._window)
            self._history[track_id] = history
        if history:
            last = history[-1]
            if (source != last.source
                    or time_s <= last.time_s
                    or time_s - last.time_s > self._max_gap_s):
                self._discard(track_id, history)
        if source in UNMEASURED_SOURCES:
            self._discard(track_id, history)
            verdict = Verdict(UNRESOLVED, f"ranging source {source!r} reports a "
                                          f"constant, not a measurement", 0, 0.0)
            self._verdicts[track_id] = verdict
            return verdict
        if not (range_m > 0.0 and math.isfinite(range_m)):
            self._discard(track_id, history)
            verdict = Verdict(UNRESOLVED, f"range {range_m} is not usable", 0, 0.0)
            self._verdicts[track_id] = verdict
            return verdict

        history.append(_Sample(time_s=float(time_s), range_m=float(range_m),
                               log_range=math.log(range_m), source=source,
                               odom_xy=(float(odom_xy[0]), float(odom_xy[1])),
                               robot_xy=(float(robot_xy[0]), float(robot_xy[1]))))
        verdict, inconsistent_rate = self._decide(list(history))
        verdict = self._latch(track_id, verdict, inconsistent_rate)
        self._verdicts[track_id] = verdict
        return verdict

    def _latch(self, track_id: int, verdict: Verdict, inconsistent_rate: bool
               ) -> Verdict:
        """Hold a rejection until the evidence for it is contradicted or discarded.

        ⚠️ WITHOUT THIS THE VERDICT REVERSES ON THE NOISE TERM ALONE. Both halves of the
        drop condition carry ``REJECT_SIGMAS * sigma``, and sigma depends on how many
        samples the window currently holds — so a window that loses ONE sample as the
        robot closes on the claimed position widens the margin and un-drops the track.
        Traced exactly: a ghost at 10x its reported range held for 60 consecutive
        frames, then the window went 12 samples to 11, 4 sigma went 0.0725 to 0.0827,
        and the contact-horizon comparison crossed. The signal had not moved at all.

        Three ways out of a rejection, and "the fit got noisier" is not one:

          * the range starts falling as fast as odometry demands — POSITIVE evidence
            that it is a real obstacle, and the only one that can arrive while the gate
            is still resolving;
          * the window is discarded (source switch, gap, backwards time), which throws
            away the measurements the rejection was made from;
          * the tracker deletes the track, which reaches :meth:`retain`.

        The contact-horizon clause is deliberately NOT one of them. It is an entry
        condition — a rejection needs the range to be both inconsistent AND slow — and
        letting it also be an exit condition is what put the reversal there.
        """
        if verdict.state == INCONSISTENT:
            self._latched.add(track_id)
            return verdict
        if verdict.state == CONSISTENT and not inconsistent_rate:
            self._latched.discard(track_id)
            return verdict
        if track_id not in self._latched:
            return verdict
        return Verdict(INCONSISTENT,
                       f"still rejected on earlier evidence; this window says "
                       f"{verdict.reason}",
                       verdict.samples, verdict.span_s, verdict.observed_rate,
                       verdict.predicted_rate, verdict.sigma_rate, verdict.scale)

    def _discard(self, track_id: int, history) -> None:
        """Throw away a window, and the rejection that was made from it.

        The two go together: a latched rejection is an opinion about a particular run
        of measurements, and keeping it after those measurements are gone would carry
        a verdict across a ranging-source change — the 103% step this window reset
        exists to avoid in the first place.
        """
        history.clear()
        self._latched.discard(track_id)

    def forget(self, track_id: int) -> None:
        """Discard everything held for a track the tracker has deleted."""
        self._history.pop(track_id, None)
        self._verdicts.pop(track_id, None)
        self._latched.discard(track_id)

    def retain(self, track_ids) -> None:
        """Forget every track not in ``track_ids``.

        The tracker prunes by rebuilding its list, so there is no per-track deletion
        hook to hang :meth:`forget` on. Sweeping against the surviving ids instead
        means a track deleted by any route — misses, coast timeout, or a future one —
        cannot leave a window behind for a later track to inherit through a reused id.
        """
        live = set(track_ids)
        for track_id in [t for t in self._history if t not in live]:
            self.forget(track_id)

    # ── Reading ─────────────────────────────────────────────────────────────
    def verdict(self, track_id: int) -> Verdict:
        """The latest verdict for a track. Unknown tracks are ``UNRESOLVED``."""
        return self._verdicts.get(
            track_id, Verdict(UNRESOLVED, "no measurements", 0, 0.0))

    def rejects(self, track_ids) -> set:
        """Which of ``track_ids`` the gate currently rules inconsistent."""
        return {t for t in track_ids if self.verdict(t).rejected}

    # ── The test ────────────────────────────────────────────────────────────
    def _conditioned(self, window: list) -> tuple:
        """The longest tail of ``window`` whose PREDICTION is close to a straight line.

        Returns ``(samples, predicted_log_ranges)`` — the prediction is computed here
        to decide the trim and would otherwise be computed twice.

        See :data:`PREDICTION_CURVATURE_SIGMAS`. Trimming from the front rather than
        abstaining outright is what keeps the gate live through a close approach: the
        oldest samples are the ones whose anchor the robot has passed or nearly reached,
        and dropping them re-anchors the prediction on where the track is claimed to be
        NOW. Where trimming cannot recover — the robot went through the claimed position
        and kept going — the window falls under ``min_samples`` and the gate abstains.
        Abstaining is a weaker statement than the rejection it replaces; reversing to
        CONSISTENT would be a contradiction, and that is the line that matters.
        """
        limit = self._curvature_sigmas * self._sigma
        trimmed = list(window)
        while len(trimmed) >= self._min_samples:
            anchor_x, anchor_y = trimmed[0].odom_xy
            ranges = [math.hypot(anchor_x - s.robot_xy[0], anchor_y - s.robot_xy[1])
                      for s in trimmed]
            if min(ranges) > 0.0:
                logs = [math.log(r) for r in ranges]
                if _straightness([s.time_s for s in trimmed], logs) <= limit:
                    return trimmed, logs
            trimmed.pop(0)
        return trimmed, []

    def _decide(self, full_window: list) -> tuple:
        """Compare the observed log-range rate with the one ego-motion demands.

        Returns ``(verdict, inconsistent_rate)``. The flag is the ego-motion half of
        the drop condition on its own, and :meth:`_latch` needs it: a track kept only
        because the contact horizon saved it has not been shown to be a real obstacle,
        and must not release a rejection.

        The prediction anchors on the window's FIRST sample: the odom point that
        reading claims the track occupies, held fixed while the robot walks. Anchoring
        on the claimed position rather than on a corrected one is the point — the test
        is whether the claim survives the robot's own motion, so the claim has to be
        what is under test.
        """
        window, predicted_logs = self._conditioned(full_window)
        n = len(window)
        span = window[-1].time_s - window[0].time_s if n else 0.0
        if n < self._min_samples or not predicted_logs:
            return Verdict(UNRESOLVED,
                           f"{n} usable samples, need {self._min_samples}"
                           + (" (the robot has closed on the claimed position, so "
                              "the prediction is no longer a straight line)"
                              if n < len(full_window) or not predicted_logs else ""),
                           n, span), False

        times = [s.time_s for s in window]
        observed_rate, sxx = _slope(times, [s.log_range for s in window])
        if sxx <= 0.0:
            return Verdict(UNRESOLVED, "every sample shares a timestamp", n, span), False
        sigma_rate = self._sigma / math.sqrt(sxx)
        predicted_rate, _ = _slope(times, predicted_logs)

        margin = self._reject_sigmas * sigma_rate
        contact_rate = -1.0 / self._tau_horizon_s

        # POWER, and it is two conditions because the drop is two conditions. A track
        # is dropped only when its observed rate clears BOTH the prediction and the
        # horizon by ``margin``, and the largest rate a static object can show is zero
        # — so the test can fire at all only if zero clears both. Without this a
        # far-off track, where the prediction is nearly flat and every rate is above
        # the horizon, is decided by whichever way the noise fell.
        if predicted_rate + margin >= 0.0:
            return Verdict(UNRESOLVED,
                           f"ego-motion predicts {predicted_rate:+.4f}/s, too little "
                           f"to resolve against {margin:.4f}/s of noise",
                           n, span, observed_rate, predicted_rate, sigma_rate), False
        if contact_rate + margin >= 0.0:
            return Verdict(UNRESOLVED,
                           f"{margin:.4f}/s of noise cannot resolve the "
                           f"{self._tau_horizon_s:.0f}s contact horizon",
                           n, span, observed_rate, predicted_rate, sigma_rate), False

        scale = observed_rate / predicted_rate
        inconsistent = observed_rate - predicted_rate > margin
        # The safety half: inconsistent is not enough. An obstacle can grow more slowly
        # than ego-motion demands and still be closing on the robot, and that one is
        # kept whatever the arithmetic says about its size prior.
        #
        # ``margin`` on this comparison too, and it was NOT there first. A bare
        # threshold on the observed rate has no confidence attached, so a track sitting
        # near the horizon is dropped whenever the noise leans that way: measured at
        # 12 samples, 12 of 120 runs at the measured noise floor were discarded for a
        # target genuinely closing inside the horizon. The whole drop condition has to
        # be significance-controlled, not just the half that looks statistical.
        beyond_horizon = observed_rate - margin > contact_rate
        if inconsistent and beyond_horizon:
            return Verdict(
                INCONSISTENT,
                f"range falls at {observed_rate:+.4f}/s where odometry demands "
                f"{predicted_rate:+.4f}/s ({(observed_rate - predicted_rate) / sigma_rate:.1f} "
                f"sigma); at least {1.0 / scale:.1f}x further off than reported",
                n, span, observed_rate, predicted_rate, sigma_rate, scale), True
        reason = ("closing as odometry demands" if not inconsistent else
                  f"slower than odometry demands, but still closing inside "
                  f"{self._tau_horizon_s:.0f}s")
        return Verdict(CONSISTENT, reason, n, span,
                       observed_rate, predicted_rate, sigma_rate, scale), inconsistent
