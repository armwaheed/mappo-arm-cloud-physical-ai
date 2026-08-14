#!/bin/bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Remove what deploy/install.sh created on this machine — and nothing else.
#
# It removes only paths the manifest records as CREATED BY AN INSTALL RUN. An environment
# or an SDK clone that was already there when install ran belongs to whoever made it and
# is left alone; the manifest records which is which at install time, because afterwards
# there is no way to tell.
#
# It never touches the repository, the recordings, the telemetry, or anything under it.
# Uninstalling is not the same as cleaning up after a demo, and a script that deleted the
# evidence would be the more expensive mistake.
#
# Usage:
#   ./uninstall.sh              # say what would be removed, remove nothing
#   ./uninstall.sh --yes        # actually remove it
#   MANIFEST=~/other ./uninstall.sh --yes

set -euo pipefail

MANIFEST="${MANIFEST:-$HOME/.mappo-go2-deploy.manifest}"
CONFIRMED=0

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y) CONFIRMED=1 ;;
        -h|--help) sed -n '5,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$MANIFEST" ] || {
    echo "no manifest at $MANIFEST — nothing was installed from here, or it was already"
    echo "removed. Nothing to do."
    exit 0
}

# Last value wins, so re-running install updates a key rather than appending a second
# truth. `awk` rather than a bash loop because a manifest is a file the user can edit and
# a malformed line should be ignored, not executed.
value() { awk -v k="$1" '$1 == k { v = $2 } END { print v }' "$MANIFEST"; }

REPO="$(value repo)"
ENV_DIR="$(value env_dir)"
SDK_REPO="$(value sdk_repo)"
ROBOTKIT_DIR="$(value robotkit_dir)"

# A path is only ever removed if it is absolute, is not a prefix of the repository, and
# is not one of the obvious catastrophes. `rm -rf "$X"` with an empty X is the classic
# and it is one typo away in a script like this.
REMOVE=()
consider() {
    local label="$1" path="$2" created="$3"
    [ -n "$path" ] || return 0
    if [ "$created" != "1" ]; then
        echo "  keep    $label $path"
        echo "          (it existed before install ran)"
        return 0
    fi
    case "$path" in
        /|"$HOME"|"$HOME"/|.|..|"") echo "  REFUSE  $label $path"; return 0 ;;
        /*) ;;
        *) echo "  REFUSE  $label $path (not an absolute path)"; return 0 ;;
    esac
    if [ -n "$REPO" ] && [ "${REPO#"$path"}" != "$REPO" ]; then
        echo "  REFUSE  $label $path (the repository is inside it)"
        return 0
    fi
    if [ ! -e "$path" ]; then
        echo "  gone    $label $path"
        return 0
    fi
    echo "  REMOVE  $label $path"
    REMOVE+=("$path")
}

echo "manifest $MANIFEST"
echo "installed $(value installed_at)"
echo ""
consider "venv       " "$ENV_DIR" "$(value created_env)"
consider "sdk clone  " "$SDK_REPO" "$(value created_sdk_clone)"
consider "shared core" "$ROBOTKIT_DIR" "$(value created_robotkit)"
echo ""
echo "  keep    repository $REPO"
echo "          (source, telemetry, recordings and evidence are never removed)"
echo ""

if [ "${#REMOVE[@]}" = 0 ]; then
    echo "Nothing to remove."
else
    if [ "$CONFIRMED" != 1 ]; then
        echo "Dry run. Re-run with --yes to remove the ${#REMOVE[@]} path(s) above."
        exit 0
    fi
    for path in "${REMOVE[@]}"; do
        # `${path:?}` makes an empty variable an error rather than an argument-less rm.
        rm -rf "${path:?}"
        echo "  removed $path"
    done
fi

if [ "$CONFIRMED" = 1 ]; then
    rm -f "$MANIFEST"
    echo "  removed $MANIFEST"
    echo ""
    echo "Done. The repository is untouched; deploy/install.sh will rebuild from it."
fi
