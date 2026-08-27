#!/usr/bin/env python3
"""Re-derive the Hugging Face username list in docs/hf-collaborators.md.

Queries the public quicksearch endpoint. No token needed, no writes.
A display-name match is a lead, not an identity proof — see the doc.
"""
import json
import urllib.parse
import urllib.request

# Each colleague, and the search terms to try for them. GitHub handles are included
# where we have them, because a handle that IS the Hugging Face username would be the
# strongest evidence available. As of 2026-08-26 none of them was.
CANDIDATES = [
    ("Belinda Wang", ["Belinda Wang"]),
    ("Fei Xiang", ["Fei Xiang"]),
    ("Na Li", ["Na Li"]),
    ("Alexander Tsyplikhin", ["Alexander Tsyplikhin", "tsyplikhin"]),
    ("Darrell Malone", ["Darrell Malone", "dmalone-arm"]),
    ("Dominica Amanfo", ["Dominica Amanfo", "DominicaAmanfo"]),
    ("Shaneil Parsad", ["Shaneil Parsad", "shaneil"]),
    ("Jackie Lee", ["Jackie Lee", "JKLEE1015", "Yiwei"]),
    ("Masoud Koleini", ["Masoud Koleini", "koleini"]),
    ("Odin Shen", ["Odin Shen", "odincodeshen"]),
    ("Sagar Surendran", ["Sagar Surendran", "spsagar13"]),
    ("Timo Tang", ["Timo Tang", "tngchien"]),
]


def search(term):
    """Return up to five (username, fullname) pairs matching `term`."""
    url = f"https://huggingface.co/api/quicksearch?q={urllib.parse.quote(term)}&type=user"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:  # a lookup failure is a result, not a crash
        return [("<lookup failed>", repr(exc))]
    return [(u.get("user"), u.get("fullname")) for u in payload.get("users", [])[:5]]


def main():
    for person, terms in CANDIDATES:
        print(f"=== {person}")
        hits = 0
        for term in terms:
            for user, fullname in search(term):
                hits += 1
                print(f"    [{term:<20}] {user:<26} fullname={fullname!r}")
        if not hits:
            print("    (no account found under any term tried)")


if __name__ == "__main__":
    main()
