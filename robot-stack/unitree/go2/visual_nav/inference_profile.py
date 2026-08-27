#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The one place the detector's preprocessing is written down, and the guard that binds it.

    python3 inference_profile.py                       # every profile, and its source
    python3 inference_profile.py --argv go2-peer-supervised   # what the launcher passes

WHAT IS ACTUALLY WRONG, WHICH IS NOT WHAT ISSUE #129 FIRST SAID.

#129 opened as "the scorers use 300 and the robot uses 224". That is not it. Enumerated
against every launcher:

    deploy/run-peer-supervised.sh          --input-size 224   --confidence 0.25
    run-smoke.sh / run-berth.sh / run-chair.sh    (no flag) -> 300   --confidence 0.45
    visual_nav.py's own parser defaults           (default)    300              0.4

**The robot runs at two different squares and three different score floors depending on
which script starts it, and nothing reconciles them.** The 89-run corpus in
``evidence/2026-08-27-89-runs-survived-14-can-be-dated/`` settles which is which: those
runs ran at 300 px, because the scripts that launched them pass no ``--input-size`` at all.

So the scorers' 300 was right for most runs and wrong for the peer runs -- and the
checkpoint sweep managed to be wrong for all of them at once, because it took the SQUARE
from the trainer (300) and the FLOOR from the peer launcher (0.25). No launcher has ever
run that pair. A ranking of 94 checkpoints was scored through a configuration that exists
nowhere.

That is why there is no ``PRODUCTION`` object in this module. There is no single production
preprocessing to point at; pretending otherwise is how one gets picked by accident. What
there is instead is :data:`DEPLOYMENTS` -- every configuration a launcher actually produces,
each naming the file that produces it -- and a scorer must say which one it is scoring for.

THE HONEST SOURCE IS THE ARGV A LAUNCHER PASSES, so this module defines those argvs and the
launcher asks. ``deploy/run-peer-supervised.sh`` no longer contains ``--input-size``,
``--confidence`` or ``--classes``; it runs ``inference_profile.py --argv`` and passes what
it is handed. The alternatives were each already tried here:

* *A parser default* is a claim about production, not production -- and here it is a claim
  that is true of three launchers and false of the fourth.
  ``deep_robotics/lite3/commissioning/camera_calibration.py`` reads that default and calls
  it "production", which is right for a smoke run and wrong for a peer run.
* *The launch scripts* are the truth, and three of the four are not in this repository at
  all, so nothing here can read them. They are declared below, by hand, from the copy of
  ``run-smoke.sh``'s invocation that ``dashboard/run-profile.example.json`` carries -- and
  ``test_inference_profile`` checks that file against the declaration, so the one copy this
  repository does hold cannot drift silently.
* *A scorer constant* is a copy of a copy.

CONFIDENCE IS DIFFERENT FROM THE REST, AND THE DIFFERENCE IS LOAD-BEARING.

``--confidence`` is applied in Python, after ``forward()``. The floor that decides what
EXISTS is ``detection_output_param { confidence_threshold }`` inside the prototxt, applied
by the ``DetectionOutput`` layer before ``forward()`` returns -- see
:func:`person_detector.prototxt_with_floor`. A caller asking for LESS than the layer's floor
measures the layer's floor and is told nothing; a caller asking for MORE genuinely gets what
it asked for, because Python discards the extra rows. :func:`assert_prototxt_floor` refuses
the first and records the second.

⛔ **A sub-threshold score is not a near miss.** Patch the layer to 0.01 and the 224 px path
reports a ``person`` in 535 of 535 frames of one walk, including the ~300 that show only
cardboard (``evidence/2026-08-27-89-runs-survived-14-can-be-dated``). Below ~0.25 the score
is noise, so "it almost fired" is not a reading anything supports.

WHAT IS DECLARED HERE AND WHAT IS VERIFIED ELSEWHERE.

``scale``, ``mean`` and ``swap_rb`` are baked into the published weights and cannot be
re-derived from a ``.caffemodel``, so :func:`assert_matches_person_detector` checks them
against ``person_detector``'s own constants, and ``test_inference_profile`` checks the
declared deployments against the two launcher artefacts this repository does hold.
Declared-and-checked, not declared-and-hoped.

Pure stdlib and Python 3.8 syntax on purpose: this is imported by the robot's Jetson
(Ubuntu 20.04 / JetPack 5), by the ``detector/`` scorers on the training host, and by tests
on a laptop with no ``opencv-python``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DEPLOYMENTS",
    "PROFILES",
    "InferenceProfile",
    "PreprocessingMismatch",
    "add_arguments",
    "assert_matches_person_detector",
    "assert_prototxt_floor",
    "prototxt_floor",
    "resolve",
    "stamp",
]


class PreprocessingMismatch(RuntimeError):
    """Raised when a measurement would be taken through preprocessing no launcher runs.

    A hard error rather than a warning, because the failure mode it exists to stop is a
    *plausible number*. A warning scrolls off; a table does not.
    """


#: VOC-21 in the order the published prototxt emits class ids. Declared here rather than
#: imported from ``person_detector`` because that module imports ``cv2`` at module scope
#: and this one must import on a machine that has none.
#: :func:`assert_matches_person_detector` checks the two agree.
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

#: The 20 real VOC labels, i.e. :data:`VOC_CLASSES` without ``background``. This is what
#: ``run-peer-supervised.sh`` passes to ``--classes``, and it matters for the *denominator*
#: a peer run holds on: with every class a mover, no label is filtered out and the aspect
#: gate alone decides hold-versus-route.
VOC_OBJECT_CLASSES = VOC_CLASSES[1:]

#: The ``DetectionOutput`` score floor, as it appears in the prototxt. Same pattern as
#: ``person_detector._CONFIDENCE_FLOOR_RE``; :func:`assert_matches_person_detector` checks
#: the two find the same thing in the same text.
_FLOOR_RE = re.compile(r"confidence_threshold:\s*([0-9]*\.?[0-9]+)")


@dataclass(frozen=True)
class InferenceProfile:
    """One complete preprocessing path: everything between a JPEG and ``forward()``.

    Frozen and compared by value, so "is this what a launcher runs?" is an equality
    test and not a judgement call.
    """

    name: str
    role: str
    input_size: int
    confidence: float
    scale: float
    mean: float
    swap_rb: bool
    classes: tuple[str, ...]
    #: The file that decides this configuration. For a deployment that is a launcher; for
    #: the reference profile it is the weights. Named rather than described, because
    #: "production" turned out to name four different things.
    source: str
    why: str

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")

    @property
    def blob_size(self) -> tuple[int, int]:
        """The ``(width, height)`` argument to ``cv2.dnn.blobFromImage``."""
        return (self.input_size, self.input_size)

    @property
    def is_deployed(self) -> bool:
        """Whether some launcher actually runs this configuration.

        Compared BY VALUE, not by name: a profile called something else that squashes to
        the same square, normalises the same way and floors at the same score is the same
        computation, and one carrying a deployment's name while computing something else
        is not that deployment.
        """
        return any(self.key() == d.key() for d in DEPLOYMENTS)

    @property
    def deployments(self) -> tuple[str, ...]:
        """Every launcher that runs this exact configuration. Usually one; can be more."""
        return tuple(d.source for d in DEPLOYMENTS if d.key() == self.key())

    def key(self) -> tuple:
        """Everything that changes what the ROBOT does with a frame. Names and prose out.

        ``classes`` is in here, and it is not decoration. ``PersonDetector.detect_tiered``
        drops any detection whose label is not in the list before anything downstream sees
        it, so the class list decides *which boxes exist* as far as the planner and the
        hold path are concerned. It differs between deployments exactly as the square and
        the floor do: ``run-peer-supervised.sh`` passes all twenty VOC labels, and the
        other launchers pass none, taking ``visual_nav.py``'s default of ``("person",)``.
        A scorer that ignored it would count a `chair` box as an obstacle on a run that
        throws `chair` away.
        """
        return (self.input_size, self.confidence, self.scale, self.mean, self.swap_rb,
                self.classes)

    def differences_from(self, other: InferenceProfile) -> dict[str, tuple]:
        """``{field: (mine, theirs)}`` for every field that changes the computation."""
        fields = ("input_size", "confidence", "scale", "mean", "swap_rb", "classes")
        return {f: (getattr(self, f), getattr(other, f))
                for f in fields if getattr(self, f) != getattr(other, f)}

    def as_dict(self) -> dict:
        """JSON-shaped, for stamping into a results file or a telemetry header. A number
        nobody can attribute to a preprocessing path is the whole defect this exists for.
        """
        return {"profile": self.name, "role": self.role, "source": self.source,
                "input_size": self.input_size, "confidence": self.confidence,
                "scale": self.scale, "mean": self.mean, "swap_rb": self.swap_rb,
                "classes": list(self.classes),
                "deployed": self.is_deployed, "deployments": list(self.deployments)}

    def argv(self) -> tuple[str, ...]:
        """The flags a launcher passes to reproduce this profile."""
        return ("--input-size", str(self.input_size),
                "--confidence", repr(self.confidence),
                "--classes", *self.classes)


#: ⚠️ CHANGING ``input_size`` OR ``confidence`` HERE CHANGES WHAT THE ROBOT DOES. This is
#: not a scorer knob: ``deploy/run-peer-supervised.sh`` takes its flags from this object, so
#: an edit here is a robot behaviour change and needs a live run, not a re-score.
#:
#: 224 rather than the trained 300 is deliberate and is measured on both sides. On the
#: 2026-08-25 clip it detected the peer on 12 of 12 frames against 2 of 12 at 300, and it is
#: ~1.7x faster on the Jetson. Against that: on the 2026-08-20 cross-day day it is 18 points
#: WORSE on peer recall, and on walk 3 of 2026-08-27 the shipped weights fire 0 times in 535
#: frames at 224 against 2 at 300, with a best score of 0.4274 against 0.6155. The axis is
#: non-monotonic, which is a marginal-detection smell, and no result on it generalises.
GO2_PEER_SUPERVISED = InferenceProfile(
    name="go2-peer-supervised",
    role="deployment",
    input_size=224,
    confidence=0.25,
    scale=1.0 / 127.5,
    mean=127.5,
    swap_rb=False,
    classes=VOC_OBJECT_CLASSES,
    source="deploy/run-peer-supervised.sh",
    why="the peer-avoidance runs, and the only launcher in this repository. It takes its "
        "--input-size/--confidence/--classes from this object, so the two cannot drift.",
)

#: ⚠️ **DECLARED BY HAND, BECAUSE THE LAUNCHER IS NOT IN THIS REPOSITORY.**
#: ``run-smoke.sh``, ``run-berth.sh`` and ``run-chair.sh`` live on the robot at
#: ``/home/unitree/``. None passes ``--input-size``, so all three take
#: ``visual_nav.py``'s 300; the 0.45 is the ``--confidence`` that
#: ``dashboard/run-profile.example.json`` carries, and that file states in its own header
#: that it was copied from ``run-smoke.sh``. That JSON is the ONE copy of this invocation
#: the repository holds, and ``test_inference_profile`` checks it against this declaration.
#: The other two launchers cannot be checked from a clone at all, which is a finding rather
#: than an oversight: a launcher nobody can read is a launcher nothing can reconcile.
#:
#: These are the runs the corpus is made of: 89 of them, and
#: ``evidence/2026-08-27-89-runs-survived-14-can-be-dated`` establishes they ran at 300 px.
GO2_RUN_SMOKE = InferenceProfile(
    name="go2-run-smoke",
    role="deployment",
    input_size=300,
    confidence=0.45,
    scale=1.0 / 127.5,
    mean=127.5,
    swap_rb=False,
    classes=("person",),
    source="/home/unitree/run-smoke.sh (via dashboard/run-profile.example.json)",
    why="the smoke, berth and chair runs -- i.e. every logged run in the 89-run corpus. "
        "300 px because no --input-size is passed, 0.45 because it is, and ('person',) "
        "because no --classes is passed either -- so on these runs a box the network "
        "labels anything else does not reach the planner at all.",
)

#: What a bare ``visual_nav.py`` computes. It is a deployment because it is reachable
#: without editing anything: any launcher that omits both flags runs THIS, and three do.
#: It differs from :data:`GO2_RUN_SMOKE` only in the floor, and only because those scripts
#: spell ``--confidence`` and nothing forces them to spell the same one.
GO2_NAVIGATOR_DEFAULT = InferenceProfile(
    name="go2-navigator-default",
    role="deployment",
    input_size=300,
    confidence=0.4,
    scale=1.0 / 127.5,
    mean=127.5,
    swap_rb=False,
    classes=("person",),
    source="robot-stack/unitree/go2/visual_nav/visual_nav.py (parser defaults)",
    why="what runs when a launcher passes neither flag. Checked against the parser itself "
        "in test_inference_profile, rather than restated here and left to rot.",
)

#: ⛔ **NOT A DEPLOYMENT, AND THIS IS THE ONE THE CHECKPOINT SWEEP USED.** 300 is the square
#: the published MobileNet-SSD weights were fitted at, and 0.25 is the floor the published
#: prototxt bakes into ``DetectionOutput``. Both are real numbers about the weights. Their
#: PAIR is not a configuration any launcher runs: the launchers at 300 px floor at 0.45 or
#: 0.4, and the launcher that floors at 0.25 runs 224 px.
#:
#: It is correct for a TRAINER -- ``detector/finetune_ssd.py`` and
#: ``detector/train_new_class.py`` are optimising the weights this square produced, and 224
#: would be a different experiment. It is a named profile rather than a bare ``300`` so
#: that a *scorer* reaching for it has to say the word out loud and give a reason.
MOBILENET_SSD_TRAINED = InferenceProfile(
    name="mobilenet-ssd-trained",
    role="reference",
    input_size=300,
    confidence=0.25,
    scale=1.0 / 127.5,
    mean=127.5,
    swap_rb=False,
    classes=VOC_OBJECT_CLASSES,
    source="the published MobileNet-SSD weights and prototxt",
    why="the square the weights were fitted at, and the prototxt's own DetectionOutput "
        "floor. Correct for a trainer; run by no launcher; scored by the 2026-08-26 sweep.",
)

#: Every configuration a launcher actually produces. ORDER IS NOT PRECEDENCE -- there is no
#: precedence, which is the point: these are four different detectors on one robot.
DEPLOYMENTS: tuple = (GO2_PEER_SUPERVISED, GO2_RUN_SMOKE, GO2_NAVIGATOR_DEFAULT)

PROFILES: dict[str, InferenceProfile] = {
    p.name: p for p in (*DEPLOYMENTS, MOBILENET_SSD_TRAINED)}


def prototxt_floor(text: str) -> float:
    """The ``DetectionOutput`` score floor baked into ``text``.

    Raises unless exactly one is present, for the reason
    :func:`person_detector.prototxt_with_floor` gives: with two, one is silently
    authoritative and no caller can tell which.
    """
    found = _FLOOR_RE.findall(text)
    if len(found) != 1:
        raise PreprocessingMismatch(
            f"expected exactly one confidence_threshold in the prototxt, found "
            f"{len(found)}: {found}. With none there is no floor to check against; with "
            f"two, one of them is silently authoritative.")
    return float(found[0])


def assert_prototxt_floor(prototxt: Path, profile: InferenceProfile) -> float:
    """Refuse when ``profile.confidence`` asks for boxes the network has already deleted.

    THE ASYMMETRY THAT MAKES CONFIDENCE UNLIKE EVERY OTHER FIELD. ``DetectionOutput``
    drops rows below its own ``confidence_threshold`` inside ``forward()``, so:

    * **Below the layer's floor is a lie.** A profile at 0.10 against a 0.25 prototxt
      measures 0.25 and would print it under the profile's name. Refused.
    * **Above the layer's floor is honest.** Python discards the extra rows itself, so a
      profile at 0.45 against a 0.25 prototxt really does measure 0.45 — which is exactly
      what ``run-smoke.sh`` does, and refusing it would forbid scoring the deployment the
      whole 89-run corpus was recorded through. Allowed, and the layer's floor is returned
      so the caller can record both numbers rather than only the one it chose.

    Returns the floor it found.
    """
    floor = prototxt_floor(Path(prototxt).read_text())
    if profile.confidence < floor:
        raise PreprocessingMismatch(
            f"profile {profile.name!r} asks for confidence {profile.confidence} but this "
            f"prototxt's DetectionOutput layer already discards everything below {floor} "
            f"inside forward(). The boxes between the two do not exist by the time Python "
            f"sees the output, so the run would measure {floor} and label it "
            f"{profile.confidence}. Lower the prototxt floor with "
            f"person_detector.prototxt_with_floor, or raise the profile.")
    return floor


def assert_matches_person_detector(module: object) -> dict[str, object]:
    """Check this module's declarations against the production detector's own constants.

    ``scale``, ``mean`` and the class list cannot be re-derived from a weights file, so
    they are declared in two places by necessity. This is the check that stops the second
    place from being a *copy* -- it is what ``INPUT_SIZE = 300`` in five scorers never
    had. Run by ``test_inference_profile.py``.

    ``swap_rb`` is NOT checked here, because there is nothing to compare it against:
    ``person_detector`` gets BGR by not passing ``swapRB`` at all, so the value lives in
    OpenCV's default rather than in a constant. ``test_inference_profile`` asserts on the
    call site's source instead, and fails if it starts passing the flag.

    Returns what it compared, so the assertion's inputs are recorded and not just its
    verdict.
    """
    checked = {}
    for attribute, mine in (("_SSD_SCALE", GO2_PEER_SUPERVISED.scale),
                            ("_SSD_MEAN", GO2_PEER_SUPERVISED.mean)):
        if not hasattr(module, attribute):
            raise PreprocessingMismatch(
                f"person_detector no longer defines {attribute}, so this assertion can no "
                f"longer tell what production preprocesses with. Fix the assertion rather "
                f"than deleting it.")
        theirs = getattr(module, attribute)
        if theirs != mine:
            raise PreprocessingMismatch(
                f"{attribute} is {theirs} in person_detector and {mine} in "
                f"inference_profile. One of them is what the robot computes and the other "
                f"is what every published table describes.")
        checked[attribute] = theirs
    if tuple(getattr(module, "VOC_CLASSES", ())) != VOC_CLASSES:
        raise PreprocessingMismatch(
            "person_detector.VOC_CLASSES and inference_profile.VOC_CLASSES disagree; a "
            "class id would mean two different labels depending on which was imported.")
    checked["VOC_CLASSES"] = len(VOC_CLASSES)
    sample = "detection_output_param { confidence_threshold: 0.25 }"
    theirs = module._CONFIDENCE_FLOOR_RE.findall(sample)
    if not theirs or float(theirs[0][1]) != prototxt_floor(sample):
        raise PreprocessingMismatch(
            "person_detector._CONFIDENCE_FLOOR_RE and inference_profile._FLOOR_RE read "
            "different floors out of the same prototxt text.")
    checked["floor_regex"] = prototxt_floor(sample)
    return checked


def add_arguments(parser: argparse.ArgumentParser, *,
                  multiple: bool = False) -> argparse.ArgumentParser:
    """Give a scorer the ONLY preprocessing flags it is allowed to have.

    ``--preprocessing`` is **required and has no default**, and that is the whole design.
    A default would have to be one of four configurations this robot runs, and picking one
    silently is precisely the accident being fixed: the checkpoint sweep did not choose 300
    px at 0.25, it *inherited* the square from a scorer constant and the floor from a
    launcher and never noticed it had invented a fifth. Naming the deployment costs one
    flag and makes the choice visible in the shell history, the log and the output file.

    ``--input-size`` is still here, because a size sweep is a real experiment and forcing
    it into a registry entry would be theatre. It is not a way back in: :func:`resolve`
    refuses any configuration no launcher runs unless ``--allow-preprocessing-mismatch``
    supplies a reason, and that reason is written into the results.
    """
    group = parser.add_argument_group(
        "preprocessing",
        "Taken from robot-stack/unitree/go2/visual_nav/inference_profile.py, which is also "
        "where deploy/run-peer-supervised.sh gets the robot's own flags. Run that module "
        "with no arguments to see every profile and the launcher that defines it.")
    group.add_argument("--preprocessing", required=True, choices=sorted(PROFILES),
                       metavar="PROFILE",
                       action="append" if multiple else "store",
                       help="which deployment this measurement is for. REQUIRED and with "
                            "no default: this robot runs "
                            f"{len(DEPLOYMENTS)} different configurations depending on "
                            "which script starts it, so there is no such thing as 'the' "
                            "production preprocessing to fall back on."
                            + (" Repeatable: profiles that share an input size are scored "
                               "from ONE forward pass, so a difference between two of them "
                               "cannot be an inference difference." if multiple else "")
                            + " Choices: " + ", ".join(sorted(PROFILES)))
    group.add_argument("--input-size", type=int, default=None, metavar="N",
                       help="override the profile's square. Needs "
                            "--allow-preprocessing-mismatch; there is no way to set this "
                            "quietly.")
    group.add_argument("--allow-preprocessing-mismatch", default=None, metavar="REASON",
                       help="state, in words, why this measurement is worth taking through "
                            "a configuration no launcher runs. The reason is written into "
                            "the results beside the numbers it qualifies.")
    return parser


def resolve(args: argparse.Namespace) -> tuple[InferenceProfile, str | None]:
    """The guard. ``(profile, reason)``, or :class:`PreprocessingMismatch`.

    A configuration some launcher runs is allowed and recorded. Anything else costs the
    caller a sentence that ends up in the output file.
    """
    profile = PROFILES[args.preprocessing]
    size = getattr(args, "input_size", None)
    if size is not None and size != profile.input_size:
        profile = InferenceProfile(
            name=f"{profile.name}+{size}px", role="ad-hoc", input_size=size,
            confidence=profile.confidence, scale=profile.scale, mean=profile.mean,
            swap_rb=profile.swap_rb, classes=profile.classes,
            source="--input-size on the command line",
            why="--input-size override; no launcher passes this")
    reason = getattr(args, "allow_preprocessing_mismatch", None)
    if profile.is_deployed:
        return profile, reason
    if not reason:
        deployed = "; ".join(f"{d.name} = {d.input_size} px at {d.confidence} "
                             f"({d.source})" for d in DEPLOYMENTS)
        raise PreprocessingMismatch(
            f"refusing to score at {profile.name!r} — {profile.input_size} px at "
            f"confidence {profile.confidence}. NO LAUNCHER RUNS THAT. What is run is: "
            f"{deployed}. A number measured outside that set describes no run, which is "
            f"how a 94-checkpoint sweep came to rank candidates through a square taken "
            f"from the trainer and a floor taken from a launcher that uses the other "
            f"square (issue #129). If the measurement is still worth taking — reproducing "
            f"a published table usually is — say why: "
            f"--allow-preprocessing-mismatch 'reason'.")
    return profile, reason


def matching_profile(input_size: int, confidence: float, classes: Sequence[str], *,
                     scale: float = 1.0 / 127.5, mean: float = 127.5,
                     swap_rb: bool = False) -> InferenceProfile | None:
    """The declared profile a live configuration corresponds to, or ``None``.

    Used by the telemetry header, so a run's own record says which of this robot's several
    detectors produced it. ``None`` is a real and reportable answer: an operator can type
    ``--input-size 256`` and get a configuration nothing here has ever measured, and the
    header should say so rather than round it to the nearest name.
    """
    key = (int(input_size), float(confidence), float(scale), float(mean), bool(swap_rb),
           tuple(classes))
    for profile in PROFILES.values():
        if profile.key() == key:
            return profile
    return None


def resolve_many(args: argparse.Namespace) -> tuple[list, str | None]:
    """:func:`resolve` for a scorer that takes several ``--preprocessing`` flags.

    One reason covers the invocation, not each profile: a run that scores a deployment and
    a non-deployment side by side is doing so for one purpose, and repeating the sentence
    per profile would make it decoration. Every profile is still stamped individually with
    whether a launcher runs it.
    """
    names = list(args.preprocessing)
    reason = getattr(args, "allow_preprocessing_mismatch", None)
    profiles = []
    for name in names:
        one = argparse.Namespace(preprocessing=name,
                                 input_size=getattr(args, "input_size", None),
                                 allow_preprocessing_mismatch=reason)
        profiles.append(resolve(one)[0])
    if len({p.name for p in profiles}) != len(profiles):
        raise PreprocessingMismatch(f"--preprocessing repeated: {names}")
    return profiles, reason


def stamp(profile: InferenceProfile, reason: str | None) -> dict:
    """What every results file must carry, so a number can be attributed to a path."""
    def plain(value):
        """Tuples become lists, so the object in memory equals the object in the file.

        ``json.dumps`` would do this on the way out anyway; doing it here means a test or
        a caller comparing the stamp sees what a reader of the JSON will see, rather than
        something that differs from it only after serialisation.
        """
        return list(value) if isinstance(value, tuple) else value

    out = profile.as_dict()
    out["mismatch_reason"] = reason
    if not profile.is_deployed:
        out["differences_from_deployments"] = {
            d.name: {field: {"this_run": plain(mine), "deployed": plain(theirs)}
                     for field, (mine, theirs) in profile.differences_from(d).items()}
            for d in DEPLOYMENTS}
    return out


def _render(profiles: Iterable[InferenceProfile]) -> str:
    lines = []
    for profile in profiles:
        mark = "  <- DEPLOYED" if profile.is_deployed else "  <- run by nothing"
        lines.append(f"{profile.name}{mark}")
        lines.append(f"    input_size {profile.input_size}   confidence "
                     f"{profile.confidence}   scale 1/{1.0 / profile.scale:.1f}   "
                     f"mean {profile.mean}   swap_rb {profile.swap_rb}")
        lines.append(f"    source: {profile.source}")
        lines.append(f"    {profile.why}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--argv", metavar="PROFILE", nargs="?",
                        const=GO2_PEER_SUPERVISED.name,
                        help="print the launch flags for PROFILE, one per line, for a "
                             "shell script to read. Default the production profile.")
    args = parser.parse_args(argv)
    if args.argv is None:
        print(_render(PROFILES[name] for name in sorted(PROFILES)))
        return 0
    if args.argv not in PROFILES:
        parser.error(f"no such profile {args.argv!r}; have {sorted(PROFILES)}")
    print("\n".join(PROFILES[args.argv].argv()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
