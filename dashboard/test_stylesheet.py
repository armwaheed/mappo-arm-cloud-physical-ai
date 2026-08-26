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

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = os.path.join(HERE, "static", "dashboard.css")
JS = os.path.join(HERE, "static", "dashboard.js")
HTML = os.path.join(HERE, "templates", "index.html")

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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"stylesheet: {len(tests)}/{len(tests)} passed")
