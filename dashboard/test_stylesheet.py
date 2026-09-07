#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Two bug classes in the page's own assets, neither visible by looking at the page.

`.hidden { display: none }` and `.safety { display: flex }` carry the SAME CSS specificity,
so whichever is declared later wins. Giving a component a `display` therefore silently
disables hiding on it — everywhere, not only at the control that looked broken. The safety
banner's X did nothing, and the banner also stopped hiding for robots with motion disabled,
which is the half nobody reported because nobody was looking for it.

The `hidden` attribute fails the same way from the other side: `[hidden] { display: none }`
is a user-agent rule and any author rule beats it, so `fleet-more` never hid either.

Neither is visible by looking at the page — the element is present, the class really is
applied, and the script really did run. It is visible by reading the stylesheet.

Pure stdlib. ``python3 test_stylesheet.py``.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = os.path.join(HERE, "static", "dashboard.css")
JS = os.path.join(HERE, "static", "dashboard.js")
HTML = os.path.join(HERE, "templates", "index.html")

sys.path.insert(0, HERE)

# Imported after the sys.path insert above, which is what makes it importable.
import run_control

#: ``$("some-id")`` in the script, and ``id="some-id"`` in the markup.
_LOOKUP = re.compile(r'\$\("([a-z0-9-]+)"\)')
_MARKUP_ID = re.compile(r'\bid="([a-z0-9-]+)"')

_HIDES = re.compile(
    r'classList\.(?:toggle|add|remove)\(\s*"hidden"'      # .classList.toggle("hidden", ...)
    r'|\.hidden\s*=\s*(?:true|false|[A-Za-z_$])'          # element.hidden = ...
)


def _rules(css: str):
    """(selector, body) per top-level rule, with comments stripped first.

    Stripping is not optional: the explanatory comment above the hiding block quotes CSS,
    braces and all, and a parser that does not remove comments reads those braces as rules
    and then cannot find the real ones. The first version of this test did exactly that and
    reported the hiding block missing while it sat six lines below.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def test_hidden_beats_every_component_rule_that_sets_display():
    """Both hiding mechanisms must be declared last AND marked important.

    Made to fail by dropping the `!important`, or by moving the block up the file above the
    component rules that set `display`.
    """
    with open(CSS) as handle:
        rules = _rules(handle.read())

    found = {}
    for order, (selector, body) in enumerate(rules):
        if selector in ("[hidden]", ".hidden"):
            assert re.search(r"display\s*:\s*none\s*!important", body), (
                f"`{selector}` does not use !important, so any component rule declaring a "
                f"`display` after it wins and the element stays visible")
            found[selector] = order

    for selector in ("[hidden]", ".hidden"):
        assert selector in found, f"`{selector}` is not defined; script-driven hiding is broken"

    # Nothing may re-declare display on either selector afterwards.
    last = max(found.values())
    for order, (selector, body) in enumerate(rules):
        if order > last and re.search(r"display\s*:", body) and selector in ("[hidden]", ".hidden"):
            raise AssertionError(f"`{selector}` is re-declared after the hiding block")


def test_the_script_still_hides_things_this_way():
    """Keeps the test above from passing vacuously if the hiding mechanism ever changes."""
    with open(JS) as handle:
        js = handle.read()
    assert _HIDES.search(js), (
        "nothing in the script hides an element by class or attribute any more — this test "
        "guards a pattern that no longer exists and should be deleted, not left green")


def test_every_element_the_script_reaches_for_exists_in_the_page():
    """``$("run-arm")`` on an id the markup does not have returns ``null``, and the failure
    lands on the NEXT line as "cannot read properties of null" — in a browser console
    nobody has open, halfway through ``init()``, which stops wiring every listener after it.

    So a typo in one id silently disables the rest of the page, and the visible symptom is
    a control elsewhere that does nothing. Made to fail by renaming any id in the markup
    without renaming it in the script.
    """
    with open(JS) as handle:
        wanted = set(_LOOKUP.findall(handle.read()))
    with open(HTML) as handle:
        present = set(_MARKUP_ID.findall(handle.read()))
    missing = sorted(wanted - present)
    assert not missing, (
        f"the script reaches for {missing}, which the markup does not define. Each one is a "
        f"null dereference that stops the script where it happens.")


def test_the_id_check_is_reading_a_page_that_has_ids():
    """Keeps the test above from passing because a regex stopped matching anything."""
    with open(HTML) as handle:
        present = set(_MARKUP_ID.findall(handle.read()))
    with open(JS) as handle:
        wanted = set(_LOOKUP.findall(handle.read()))
    assert len(present) > 20 and len(wanted) > 20, (len(present), len(wanted))


def _function_body(js: str, name: str) -> str:
    """The full body of ``function name(...) { ... }``, brace-matched.

    A regex cannot know where a JS function ends — nested braces mean the first ``}`` is
    rarely the last one. This walks the source counting braces from the opening one instead,
    which is exactly as much of a parser as a file that is otherwise all regexes needs.
    """
    marker = f"function {name}("
    start = js.index(marker)
    open_brace = js.index("{", start)
    depth = 0
    for i in range(open_brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[open_brace:i + 1]
    raise AssertionError(f"unbalanced braces reading function {name}")


def test_a_gait_floor_caveat_is_not_shown_on_a_transport_that_has_no_gait_floor():
    """``applyCaveats`` and ``groupHeader`` used to warn "no measured lateral/yaw/forward
    gait floor" on every Lite3 key and every Lite3 fleet row, regardless of transport — even
    though a gait floor is not a thing the sign-only axis transport has at all
    (``drive_bridge._precheck`` skips the identical check for the identical reason), and even
    once a primitive there carries a real ``measured_m_s`` entry. Measured 2026-09-07: robot
    1's lateral primitives went from absent to evidenced at 0.208/0.209 m/s, and a strafe key
    that could now execute at a known speed still said it "may produce no movement at all".

    ``renderCapabilities`` already suppresses the equivalent summary note and the
    ``force sub-floor`` checkbox for exactly this reason (its own ``signOnly``); this pins
    that the two other places carrying the same caveat text got the same guard.
    """
    with open(JS) as handle:
        js = handle.read()
    for name in ("applyCaveats", "groupHeader"):
        body = _function_body(js, name)
        assert "preserves_magnitude === false" in body, (
            f"{name} shows the 'no measured ... gait floor' caveat with no guard for a "
            f"transport that has no gait floor to be unmeasured on")


def test_a_direction_that_fires_unmeasured_gets_its_own_caveat_before_the_generic_one():
    """The sign-only axis transport's OWN "unmeasured" is
    ``motion_directions[fn].measured === false`` — an evidenced primitive that fires with no
    declared speed — and it is the more precise answer wherever both it and the generic
    per-axis gait-floor caveat could apply to the same key. It has to be checked first, or
    the generic caveat wins and prints a sentence about a gait floor that transport does not
    have, instead of the true one about a primitive nobody has timed.
    """
    with open(JS) as handle:
        js = handle.read()
    body = _function_body(js, "applyCaveats")
    assert "measured === false" in body, \
        "applyCaveats no longer distinguishes a fired-but-unmeasured direction"
    measured_check = body.index("measured === false")
    generic_check = body.index("unmeasured.has(axisFor[fn])")
    assert measured_check < generic_check, (
        "the per-direction unmeasured-speed caveat must be checked before the generic "
        "gait-floor caveat, or the generic one always wins")


def test_every_place_that_declares_the_heading_servo_default_agrees():
    """THREE files declare it and they silently disagreed for a whole session.

    `mappo_drive.py`'s argparse default was changed to `goal` in #212. `HEADING_SERVOS`
    in `run_control.py` is the dropdown ORDER. And `index.html` ships a placeholder
    `<option>` that is selected before any JavaScript runs -- which `dashboard.js` then
    preserves, because preserving an operator's choice across a re-render is correct.
    So the placeholder IS the default, and it outvoted the other two.

    An operator who never touches the control gets whatever this test protects.
    """
    with open(HTML) as handle:
        html = handle.read()
    marker = html[html.index('id="run-servo"'):]
    placeholder = marker[marker.index('value="') + 7:]
    placeholder = placeholder[:placeholder.index('"')]

    assert placeholder == run_control.HEADING_SERVOS[0], (
        f"index.html ships {placeholder!r} but HEADING_SERVOS leads with "
        f"{run_control.HEADING_SERVOS[0]!r}")
    # The third declaration -- mappo_drive.py's argparse `default=GOAL` -- lives in a
    # tree this test cannot import without dragging in the vision stack. It is pinned
    # from its own side by test_mappo_drive.py's
    # `test_a_drive_command_that_names_no_servo_gets_the_goal_law_and_never_travel`.
    # Both ends assert the same literal, so moving either one alone fails a test.
    assert placeholder == "goal", (
        "the default must be the law that faces the goal; 'travel' is issue #16 and "
        "'off' was measured walking 1.23x the direct line")


def test_every_place_that_declares_the_policy_mode_default_agrees():
    """The heading-servo bug, applied to the control that decides whether the veto runs.

    `run-mode`'s options are written out in `index.html` rather than filled from
    `POLICY_MODES`, so the two are declarations that can disagree -- and the FIRST option
    ships selected, which makes page order the default an operator gets by not touching
    the control. That is how `off` outvoted the heading-servo default for a whole
    session, and this is the same shape on a control whose two settings differ by whether
    anything stops for a person.

    Pinned rather than de-duplicated because the page's option LABELS carry text no
    constant should ("raw -- NO veto, cleared lane only"), so the list stays hand-written
    and this test keeps it honest.
    """
    with open(HTML) as handle:
        html = handle.read()
    marker = html[html.index('id="run-mode"'):]
    options = re.findall(r'<option value="([a-z]+)"', marker[:marker.index("</select>")])

    assert options == list(run_control.POLICY_MODES), (
        f"index.html offers {options} but POLICY_MODES is "
        f"{list(run_control.POLICY_MODES)} -- the first entry of each is the default an "
        f"operator gets without choosing, so these two disagreeing is a silent "
        f"behaviour change")
    assert options[0] == "raw", (
        "raw is the default since 2026-09-07: measured 7 arrived/2 collided against "
        "supervised's 6/0/4 on the repo's 10 scenarios, chosen because the veto was "
        "holding for people walking PAST the lane. If this is being changed back, change "
        "mappo_drive.py's argparse default and robot_driver.start_run together")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"stylesheet: {len(tests)}/{len(tests)} passed")
