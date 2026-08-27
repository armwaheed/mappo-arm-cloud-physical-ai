#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Every number in this directory's README, regenerated from the committed sweep files.

    python3 report.py --markdown          # the tables, with their markers
    python3 report.py --write-readme      # splice them into README.md
    python3 report.py --check-readme      # fail if the page has drifted from them

No weights, no frames, no network and no GPU. The three files were written by
``detector/score_crossday.py`` on the training host (``sweep_on_spark.sh`` is the exact
invocation) and each carries, per profile, the configuration it was taken through, whether
a launcher runs that configuration, the prototxt's own layer floor, the manifest digest,
the pixel digest, and a sha256 per checkpoint.

This script re-derives the tables and REFUSES if any of that disagrees with what the page
claims — which is the check the 2026-08-26 sweep did not have, and is why it published a
configuration no launcher runs as though it were the robot's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Sequence

HERE = pathlib.Path(__file__).resolve().parent

#: The three passes. Every one carries all four profiles; ``candidates`` also carries the
#: coverage record.
PASSES = ("incumbent", "base", "candidates")

#: The profiles the tables lead with, in the order they are shown: the three configurations
#: a launcher produces, then the one the checkpoint sweep used and nothing runs.
ORDER = ("go2-peer-supervised", "go2-run-smoke", "go2-navigator-default",
         "mobilenet-ssd-trained")

#: ``whole`` is all 284 frames — 60 with the peer, 221 without, 3 ambiguous. ``test`` is the
#: manifest's clip-wise held-out split and is what the checkpoint sweep reported; both are
#: in the files.
SPLIT = "whole"

COUNT_HEADER = ("", "peer recall", "false alarms", "hold", "+shaped", "`person` label")


def load(directory: pathlib.Path) -> dict:
    """The three passes, each checked against the profiles it claims to carry."""
    out = {}
    for name in PASSES:
        path = directory / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing. Regenerate the sweep with sweep_on_spark.sh on a host "
                f"that holds the checkpoints; see this directory's README.")
        data = json.loads(path.read_text())
        carried = {s["profile"] for s in data["preprocessing"]}
        if carried != set(ORDER):
            raise SystemExit(f"{path.name} carries {sorted(carried)}, expected "
                             f"{sorted(ORDER)}. Refusing to build a table on it.")
        for stamp in data["preprocessing"]:
            if stamp["deployed"] and not stamp["deployments"]:
                raise SystemExit(f"{path.name}: {stamp['profile']} claims to be deployed "
                                 f"and names no launcher.")
            if not stamp["deployed"] and not stamp["mismatch_reason"]:
                raise SystemExit(f"{path.name}: {stamp['profile']} is run by no launcher "
                                 f"and carries no recorded reason; it should not exist.")
        out[name] = data
    _check_class_filter_took_effect(out)
    return out


def _check_class_filter_took_effect(passes: dict) -> None:
    """For a ``person``-only profile, ``hold`` MUST equal ``person_shaped``.

    ``hold`` counts frames where a detection the profile's ``--classes`` lets through is at
    or above the aspect gate. When the only allowed label is ``person`` that is, by
    construction, exactly the person-labelled person-shaped set. If the two ever differ,
    the class filter did not run and every `fire`/`hold` number in the file is the
    class-agnostic one — which is the bug this whole page is about, quietly back.

    Cheap, total, and it fires on data rather than on code: it checks all three passes,
    every model, every split.
    """
    for pass_name, data in passes.items():
        person_only = {s["profile"] for s in data["preprocessing"]
                       if s.get("classes") == ["person"]}
        if not person_only:
            raise SystemExit(
                f"{pass_name}.json carries no person-only profile, so this check cannot "
                f"fire. Two of the three deployments pass --classes person; if none is "
                f"here, the file was not produced by the documented sweep.")
        for model in data["models"]:
            for profile in person_only:
                for split, counts in model["results"][profile].items():
                    if counts["hold"] != counts["person_shaped"]:
                        raise SystemExit(
                            f"{pass_name}.json: {model['name']} at {profile} ({split}) has "
                            f"hold={counts['hold']} and person_shaped="
                            f"{counts['person_shaped']}. For a person-only class list "
                            f"those are the same set, so the class filter did not run.")


def rows(data: dict, profile: str, split: str = SPLIT) -> dict:
    """``{model name: counts}`` for one pass at one profile."""
    return {m["name"]: m["results"][profile][split] for m in data["models"]}


def only(data: dict, profile: str, split: str = SPLIT) -> dict:
    return next(iter(rows(data, profile, split).values()))


def stamps(data: dict) -> dict:
    return {s["profile"]: s for s in data["preprocessing"]}


def rate(counts: dict) -> tuple:
    return (counts["recall_n"] / counts["recall_d"], counts["fp_n"] / counts["fp_d"])


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.0f}%"


def beats(candidate: dict, incumbent: dict) -> bool:
    """Higher peer recall AND fewer false alarms. The sweep's own two-axis rule."""
    mine, theirs = rate(candidate), rate(incumbent)
    return mine[0] > theirs[0] and mine[1] < theirs[1]


def keeps(candidate: dict, incumbent: dict) -> bool:
    """Loses none of the people the incumbent holds for, on BOTH denominators."""
    return (candidate["hold"] >= incumbent["hold"]
            and candidate["person_shaped"] >= incumbent["person_shaped"])


def clears_all(candidate: dict, incumbent: dict) -> bool:
    return beats(candidate, incumbent) and keeps(candidate, incumbent)


def label(stamp: dict) -> str:
    """How a profile is named in a table: its whole configuration, then what runs it.

    The class list is in the name because it is in the configuration: it decides which
    boxes reach the planner at all, and it differs between launchers exactly as the square
    and the floor do.
    """
    where = ", ".join(f"`{d}`" for d in stamp["deployments"]) or "**run by nothing**"
    classes = stamp.get("classes", [])
    which = "`person` only" if classes == ["person"] else f"{len(classes)} VOC labels"
    return (f"`{stamp['profile']}` — {stamp['input_size']} px, floor "
            f"{stamp['confidence']}, {which}<br>{where}")


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in body]
    return "\n".join(lines)


def _counts_row(name: str, counts: dict, bold: bool = False) -> list:
    mark = (lambda s: f"**{s}**") if bold else (lambda s: s)
    return [name,
            mark(pct(counts["recall_n"], counts["recall_d"])),
            mark(pct(counts["fp_n"], counts["fp_d"])),
            mark(str(counts["hold"])), mark(str(counts["person_shaped"])),
            str(counts["person"])]


def sections(data: dict, split: str = SPLIT) -> dict:
    """``{marker name: markdown block}``, in the order the README carries them."""
    marks = stamps(data["incumbent"])
    blocks = {}

    # ---- the incumbent, once per configuration -------------------------------------
    blocks["INCUMBENT"] = _table(
        ("one detector, four configurations", *COUNT_HEADER[1:]),
        [_counts_row(label(marks[name]), only(data["incumbent"], name, split),
                     bold=name == "mobilenet-ssd-trained") for name in ORDER])

    # ---- add_class.py's claim, checked ----------------------------------------------
    same = all(only(data["base"], n, split) == only(data["incumbent"], n, split)
               for n in ORDER)
    blocks["BASE"] = (
        "The 22-class starting point every candidate was grown from — the incumbent's own "
        "weights through `detector/add_class.py` — reproduces **every one of those rows "
        "exactly**, at all four configurations. That is `add_class.py`'s claim checked "
        "rather than quoted, and it is what lets a candidate row be read against an "
        "incumbent row at all." if same else
        "⚠️ **The grown 22-class base no longer scores identically to the incumbent.** "
        "`detector/add_class.py` claims growing the class count preserves the network "
        "exactly; these files disagree. Read no candidate row until that is explained.")

    # ---- the same rows on the split the sweep reported -------------------------------
    blocks["SPLIT"] = _table(
        ("incumbent, the manifest's `test` split", *COUNT_HEADER[1:]),
        [_counts_row(label(marks[name]), only(data["incumbent"], name, "test"),
                     bold=name == "mobilenet-ssd-trained") for name in ORDER])

    # ---- how many candidates clear the bar, per configuration ------------------------
    at = {name: rows(data["candidates"], name, split) for name in ORDER}
    incumbent = {name: only(data["incumbent"], name, split) for name in ORDER}
    winners = {name: [m for m, c in at[name].items() if beats(c, incumbent[name])]
               for name in ORDER}
    clearers = {name: [m for m, c in at[name].items() if clears_all(c, incumbent[name])]
                for name in ORDER}
    total = len(at[ORDER[0]])
    blocks["HEADLINE"] = _table(
        ["configuration", "candidates beating both peer axes", "...and keeping the people",
         "the bar: incumbent recall", "hold", "candidates holding for nobody"],
        [[label(marks[name]),
          f"{len(winners[name])} of {total}",
          f"**{len(clearers[name])}**",
          pct(incumbent[name]["recall_n"], incumbent[name]["recall_d"]),
          str(incumbent[name]["hold"]),
          str(sum(1 for c in at[name].values() if c["hold"] == 0))]
         for name in ORDER])

    # ---- who clears everything, per configuration ------------------------------------
    out = []
    for name in ORDER:
        deployed = bool(marks[name]["deployments"])
        best = sorted(clearers[name],
                      key=lambda m, n=name: (-rate(at[n][m])[0], rate(at[n][m])[1]))
        out.append(f"**{label(marks[name])}**\n")
        if not best:
            out.append(f"No checkpoint of the {total} clears every gate here.\n")
            continue
        body = [_counts_row(f"`{m}`", at[name][m]) for m in best[:8]]
        body.append(_counts_row("**incumbent — the bar**", incumbent[name], bold=True))
        out.append(_table((f"clears all four{'' if deployed else ' (run by nothing)'}",
                           *COUNT_HEADER[1:]), body))
        out.append(f"\n...and {len(best) - 8} more.\n" if len(best) > 8 else "")
    blocks["CLEARERS"] = "\n".join(out).strip()

    # ---- do the winners at one configuration win at another? --------------------------
    body = [[label(marks[a])] + [
        "—" if a == b else str(len(set(clearers[a]) & set(clearers[b])))
        for b in ORDER] for a in ORDER]
    everywhere = set.intersection(*(set(clearers[n]) for n in ORDER))
    deployed_only = set.intersection(*(set(clearers[n]) for n in ORDER[:3]))
    blocks["OVERLAP"] = "\n".join([
        _table(["checkpoints clearing all four gates at BOTH", *(f"`{n}`" for n in ORDER)],
               body),
        "",
        f"Clearing every gate at **all three deployed** configurations: "
        f"**{len(deployed_only)}** of {total}"
        + (f" — {sorted(deployed_only)}" if deployed_only else "."),
        "",
        f"Clearing every gate at **all four**, the sweep's own included: "
        f"**{len(everywhere)}**" + (f" — {sorted(everywhere)}" if everywhere else "."),
    ])

    # ---- the sweep's own picks, at every configuration --------------------------------
    named = ["s_pseudo02_aug/epoch020", "k_full_pseudo03/epoch022",
             "k_full_pseudo03/epoch020", "k_full_pseudo03/epoch017",
             "k_full_pseudo03/epoch010", "p_bb02_d01_aug/epoch020",
             "l_full_bb02/epoch040", "f_full_distil01/epoch020"]
    body = []
    for model in named:
        row = [f"`{model}`"]
        for name in ORDER:
            counts = at[name].get(model)
            if counts is None:
                row.append("NOT SCORED")
                continue
            verdict = ("**clears all four**" if clears_all(counts, incumbent[name])
                       else "beats the peer axes, loses people"
                       if beats(counts, incumbent[name]) else "no")
            row.append(f"{pct(counts['recall_n'], counts['recall_d'])}<br>"
                       f"hold {counts['hold']} of {incumbent[name]['hold']}<br>{verdict}")
        body.append(row)
    blocks["NAMED"] = _table(["the sweep's picks", *(f"at `{n}`" for n in ORDER)], body)

    # ---- coverage ---------------------------------------------------------------------
    coverage = data["candidates"].get("coverage", {})
    missed = coverage.get("not_scored", [])
    blocks["COVERAGE"] = _table(
        ["", ""],
        [["candidate checkpoints scored", f"**{coverage.get('scored', total)}**"],
         ["`.caffemodel` files matching the inventory glob",
          str(coverage.get("available", "—"))],
         ["not scored", f"**{len(missed)}**" + (f" — {missed}" if missed else "")],
         ["configurations each was scored at", f"**{len(ORDER)}**"],
         ["forward passes per checkpoint",
          str(len({marks[n]["input_size"] for n in ORDER}))]])

    # ---- provenance --------------------------------------------------------------------
    reference = data["candidates"]
    marks_c = stamps(reference)
    blocks["PROVENANCE"] = _table(
        ["what", "value"],
        [["frames scored", str(reference["manifest"]["frames"])],
         ["manifest sha256", f"`{reference['manifest']['sha256'][:16]}…`"],
         ["pixel digest", f"`{reference['frames']['digest'][:16]}…`"],
         ["candidate prototxt sha256", f"`{reference['prototxt']['sha256'][:16]}…`"],
         ["incumbent weights sha256",
          f"`{data['incumbent']['models'][0]['sha256'][:16]}…`"],
         ["hold gate", f"{reference['rule']['source']} = "
                       f"{reference['rule']['aspect_min']}"],
         ["prototxt DetectionOutput floor",
          str(marks_c["go2-peer-supervised"]["prototxt_layer_floor"])],
         ["the pass run by nothing, and its recorded reason",
          f"`mobilenet-ssd-trained` — "
          f"{marks_c['mobilenet-ssd-trained']['mismatch_reason']}"]])

    order = ("HEADLINE", "INCUMBENT", "BASE", "SPLIT", "CLEARERS", "OVERLAP", "NAMED",
             "COVERAGE", "PROVENANCE")
    if set(blocks) ^ set(order):
        raise SystemExit(f"sections() and its README order disagree about "
                         f"{set(blocks) ^ set(order)}")
    return {name: blocks[name] for name in order}


def rendered(blocks: dict[str, str]) -> str:
    return "\n\n".join(f"<!-- TABLE-{name} -->\n{block}\n<!-- /TABLE-{name} -->"
                       for name, block in blocks.items())


def splice(readme: str, blocks: dict[str, str]) -> str:
    """Put each block between its markers, leaving every word around them alone."""
    out = readme
    for name, block in blocks.items():
        pattern = re.compile(rf"<!-- TABLE-{name} -->.*?<!-- /TABLE-{name} -->", re.DOTALL)
        if not pattern.search(out):
            raise SystemExit(
                f"README.md has no <!-- TABLE-{name} --> … <!-- /TABLE-{name} --> pair. "
                f"Every generated table must sit between its markers, or --check-readme "
                f"cannot tell a stale number from a prose change.")
        replacement = f"<!-- TABLE-{name} -->\n{block}\n<!-- /TABLE-{name} -->"
        out = pattern.sub(lambda _m, r=replacement: r, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", type=pathlib.Path, default=HERE / "sweep",
                        help="directory of the three JSON files (default: %(default)s)")
    parser.add_argument("--markdown", action="store_true",
                        help="emit the README's tables verbatim, with their markers")
    parser.add_argument("--write-readme", action="store_true",
                        help="splice them into README.md between those markers")
    parser.add_argument("--check-readme", action="store_true",
                        help="fail if README.md's tables are not what this run produces. "
                             "This is what makes the page a measurement rather than a "
                             "transcription of one.")
    parser.add_argument("--split", default=SPLIT, choices=("whole", "test", "select"))
    args = parser.parse_args(argv)
    blocks = sections(load(args.sweep), args.split)
    readme = HERE / "README.md"
    if args.write_readme:
        readme.write_text(splice(readme.read_text(), blocks))
        print(f"wrote {readme}")
        return 0
    if args.check_readme:
        current = readme.read_text()
        if splice(current, blocks) != current:
            print("README.md disagrees with the committed sweep files. Regenerate it:\n"
                  "    python3 report.py --write-readme")
            return 1
        print(f"README.md agrees with {args.sweep}: {len(blocks)} generated tables")
        return 0
    text = rendered(blocks)
    print(text if args.markdown else text.replace("|", " ").replace("**", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
