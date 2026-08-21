#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bring the demo up on a host with no robots attached: a simulated fleet on a real Device
# Connect mesh, a replayed camera, and a self-hosted checkpoint server.
#
# EVERY ROBOT HERE IS SIMULATED and says so — on its own fleet row, in its device identity,
# and in get_capabilities. A demo fleet that looks identical to a real one is a hazard, not a
# better demo: someone eventually presses a key believing a robot is on the other end.
#
#   ./run_demo.sh start | stop | status
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN="${MAPPO_DEMO_RUN:-$HOME/.mappo-demo}"
VENV="${MAPPO_DEMO_VENV:-$RUN/venv}"
PY="$VENV/bin/python"

DASH_PORT="${MAPPO_DASH_PORT:-8090}"
MODEL_PORT="${MAPPO_MODEL_PORT:-9000}"
# The second source the demo contrasts against. Same server program, different label and
# different contents — see the note by SOURCES below for why it is not really S3.
S3_PORT="${MAPPO_S3_PORT:-9001}"
BIND="${MAPPO_BIND:-0.0.0.0}"
FRAMES="${MAPPO_FRAMES:-$RUN/frames}"
PKG="${MAPPO_PKG:-$RUN/policy}"
STORE="${MAPPO_MODEL_STORE:-$RUN/served-models}"
S3_STORE="${MAPPO_S3_STORE:-$RUN/served-models-s3}"
SOURCES="$RUN/model-sources.json"
LOGS="$RUN/logs"

# The fleet: two Go2s and two Lite3s. One Lite3 is left WITHOUT --allow-motion on purpose, so
# the demo always shows a robot whose motion keys are correctly refused — the gate is easier
# to believe when you can see it holding.
FLEET=(
  "go2   demo-go2-01   --allow-motion"
  "go2   demo-go2-02   --allow-motion"
  "lite3 demo-lite3-01 --allow-motion"
  "lite3 demo-lite3-02"
)

# THE TWO SOURCES THE DEMO OFFERS, advertised by every robot so the dashboard can name them
# rather than asking anyone to remember a URL.
#
# ⚠️ NEITHER IS WHAT ITS NAME SAYS, and both say so. The first is this VM in eastus, not a
# CPU server in Tokyo. The second is the SAME program with a different label and different
# contents — it is not AWS, there is no bucket, and nothing here speaks the S3 API. They
# exist so the demo can show the CHOICE between two places, which is the actual subject;
# `simulated: true` travels with each one into the dashboard, which prints it.
#
# ⚠️ The addresses are what the ROBOT must reach, not what a browser must. Here they are the
# same host, so loopback would work — but a real deployment has the robot elsewhere, and a
# loopback address that happens to work on a demo box is a trap for whoever copies this file.
# The newest .npz in a directory, or empty. Used to prefill the dashboard's Source field so
# a demo opens with a loadable address in it rather than an empty box. Computed from what
# the store actually holds — a hard-coded filename here is a demo that breaks silently the
# first time somebody swaps the checkpoints out.
newest_model() {
  ls -t "$1"/*.npz 2>/dev/null | head -1 | xargs -r basename
}

write_sources() {
  local host; host="$(hostname -I 2>/dev/null | awk "{print \$1}")"
  local arm_default s3_default
  arm_default="$(newest_model "$STORE")"
  s3_default="$(newest_model "$S3_STORE")"
  cat > "$SOURCES" <<JSON
{
  "sources": [
    {
      "label": "Arm AGI CPU server",
      "location": "Tokyo, Japan",
      "kind": "server",
      "index_url": "http://${host}:${MODEL_PORT}/index.json",
      "default_model": "http://${host}:${MODEL_PORT}/${arm_default}",
      "simulated": true
    },
    {
      "label": "AWS S3",
      "location": "cn-north-1, Beijing",
      "kind": "s3",
      "index_url": "http://${host}:${S3_PORT}/index.json",
      "default_model": "http://${host}:${S3_PORT}/${s3_default}",
      "simulated": true
    }
  ]
}
JSON
}

start() {
  mkdir -p "$LOGS" "$S3_STORE"
  [ -x "$PY" ] || { echo "no venv at $VENV — run install_demo.sh first" >&2; exit 1; }
  [ -d "$FRAMES" ] || echo "WARNING: no replay frames at $FRAMES; cameras will be synthetic" >&2
  stop >/dev/null 2>&1 || true

  write_sources
  echo "==> Arm AGI checkpoint server on :$MODEL_PORT"
  nohup "$PY" "$HERE/model_server.py" --dir "$STORE" --port "$MODEL_PORT" --host "$BIND" \
      --label "Arm AGI CPU server" --location "Tokyo, Japan" \
      > "$LOGS/model-server.log" 2>&1 &

  echo "==> stand-in for the China S3 bucket on :$S3_PORT"
  nohup "$PY" "$HERE/model_server.py" --dir "$S3_STORE" --port "$S3_PORT" --host "$BIND" \
      --label "AWS S3 (stand-in — not the S3 API)" --location "cn-north-1, Beijing" \
      > "$LOGS/model-server-s3.log" 2>&1 &

  for spec in "${FLEET[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    local platform="$1" id="$2"; shift 2
    echo "==> $id ($platform, simulated)"
    DEVICE_CONNECT_ALLOW_INSECURE=true nohup "$PY" "$ROOT/dashboard/robot_driver.py" \
        --platform "$platform" --simulate --package "$PKG" --device-id "$id" \
        --camera-replay-dir "$FRAMES" --model-sources "$SOURCES" "$@" \
        > "$LOGS/$id.log" 2>&1 &
    sleep 1
  done

  # After the drivers, so the dashboard's first discovery already finds them and the page is
  # populated the moment anybody opens it.
  sleep 6
  echo "==> dashboard on :$DASH_PORT"
  DEVICE_CONNECT_ALLOW_INSECURE=true nohup "$PY" "$ROOT/dashboard/server.py" \
      --port "$DASH_PORT" --host "$BIND" > "$LOGS/dashboard.log" 2>&1 &
  sleep 8
  status
}

stop() {
  pkill -f "$ROOT/dashboard/robot_driver.py" 2>/dev/null || true
  pkill -f "$ROOT/dashboard/server.py" 2>/dev/null || true
  pkill -f "$HERE/model_server.py" 2>/dev/null || true
  sleep 1
  echo "stopped"
}

status() {
  echo
  # pgrep -f matches this script's own command line too, so count by the interpreter path.
  printf "drivers:   %s running\n" "$(pgrep -fc "$PY $ROOT/dashboard/robot_driver.py" || echo 0)"
  printf "dashboard: %s\n" "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$DASH_PORT/" || echo down)"
  printf "models:    %s (Arm)  %s (S3 stand-in)\n" \
    "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MODEL_PORT/index.json" || echo down)" \
    "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$S3_PORT/index.json" || echo down)"
  echo
  echo "  dashboard     http://$(hostname -I 2>/dev/null | awk '{print $1}'):$DASH_PORT/"
  echo "  Arm server    http://$(hostname -I 2>/dev/null | awk '{print $1}'):$MODEL_PORT/index.json"
  echo "  S3 stand-in   http://$(hostname -I 2>/dev/null | awk '{print $1}'):$S3_PORT/index.json"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
