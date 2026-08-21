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
BIND="${MAPPO_BIND:-0.0.0.0}"
FRAMES="${MAPPO_FRAMES:-$RUN/frames}"
PKG="${MAPPO_PKG:-$RUN/policy}"
STORE="${MAPPO_MODEL_STORE:-$RUN/served-models}"
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

start() {
  mkdir -p "$LOGS"
  [ -x "$PY" ] || { echo "no venv at $VENV — run install_demo.sh first" >&2; exit 1; }
  [ -d "$FRAMES" ] || echo "WARNING: no replay frames at $FRAMES; cameras will be synthetic" >&2
  stop >/dev/null 2>&1 || true

  echo "==> checkpoint server on :$MODEL_PORT"
  nohup "$PY" "$HERE/model_server.py" --dir "$STORE" --port "$MODEL_PORT" --host "$BIND" \
      --label "Arm Neoverse CPU server" --location "Tokyo, Japan" \
      > "$LOGS/model-server.log" 2>&1 &

  for spec in "${FLEET[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    local platform="$1" id="$2"; shift 2
    echo "==> $id ($platform, simulated)"
    DEVICE_CONNECT_ALLOW_INSECURE=true nohup "$PY" "$ROOT/dashboard/robot_driver.py" \
        --platform "$platform" --simulate --package "$PKG" --device-id "$id" \
        --camera-replay-dir "$FRAMES" "$@" \
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
  printf "models:    %s\n" "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MODEL_PORT/index.json" || echo down)"
  echo
  echo "  dashboard     http://$(hostname -I 2>/dev/null | awk '{print $1}'):$DASH_PORT/"
  echo "  model server  http://$(hostname -I 2>/dev/null | awk '{print $1}'):$MODEL_PORT/index.json"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
