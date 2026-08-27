#!/usr/bin/env bash
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# One command that starts the checkpoint server, one driver and the dashboard, and prints
# the URL. Ctrl-C stops all three.
#
#     ./start-dashboard.sh                          # bench double, no robot
#     ./start-dashboard.sh --robot 192.168.123.18   # simulated pose, REAL camera
#
# WHY THIS FILE EXISTS. The documented way in was three terminals, a version-specific
# interpreter, a pip line and a flag nobody would guess, and each of those has its own way
# of failing late:
#
#   * `python3` on a Mac with Homebrew Python installed is still /usr/bin/python3 — 3.9.6
#     here — and `device-connect-edge` needs >= 3.11. The traceback names a module, not a
#     version, so the first guess is a missing package.
#   * `pip install device-connect-edge device-connect-agent-tools eclipse-zenoh` — the line
#     that was in circulation — installs none of what the three commands after it actually
#     import. `eclipse-zenoh` is already a dependency of `device-connect-edge`; `aiohttp`
#     and `numpy` are not dependencies of anything here and are what `server.py` and
#     `robot_driver.py` die on. MEASURED: in a clean 3.11 venv that line leaves all three
#     of the next commands failing at ModuleNotFoundError.
#   * A previous demo leaves three processes on 8080/8800 and the next one fails to bind
#     twenty-five minutes later, on a line that has nothing to do with the reason. Both
#     halves are handled: this refuses to start on a busy port and NAMES the process, and
#     it takes all three down together on the way out.
#
# WHAT IT WILL NOT DO. It never enables motion. `--allow-motion` is passed through only
# when a person types it on this command line; there is no environment variable, no config
# file and no default that can turn it on, and `test_start_dashboard.py` pins that.
# `robot-stack/SAFETY.md` governs anything that moves a leg.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ---- defaults ---------------------------------------------------------------------------
PLATFORM=""                       # decided below: sim with no robot, go2 with one
PACKAGE="$HERE/../policy"
MODELS_DIR=""                     # defaults to $PACKAGE/models
DEVICE_ID=""
PORT=8080
MODEL_PORT=8800
BIND=127.0.0.1
CAMERA_URL=""
ROBOT=""
PYTHON="${MAPPO_PYTHON:-}"
ALLOW_MOTION=0                    # set ONLY by the literal --allow-motion argument
EXTRA=()
LOG_DIR=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
start-dashboard.sh — the checkpoint server, a driver and the dashboard, together.

  --robot HOST         a Go2 running go2_frame_server.py: real camera, simulated pose.
                       Shorthand for --platform go2 --simulate --camera-url http://HOST:8801/
  --camera-url URL     an endpoint returning one JPEG per GET (the long form of --robot)
  --platform NAME      go2 | lite3 | sim          (default: sim, or go2 with --robot)
  --simulate           drive the bench double while presenting --platform's rules
  --package DIR        the policy package                 (default: ../policy)
  --models-dir DIR     what the checkpoint server serves  (default: <package>/models)
  --device-id ID       device id on the mesh              (default: mappo-<platform>)
  --port N             dashboard port                     (default: 8080)
  --model-port N       checkpoint server port             (default: 8800)
  --host ADDR          what BOTH servers bind             (default: 127.0.0.1)
  --python PATH        interpreter to use (or set MAPPO_PYTHON)
  --log-dir DIR        keep the three logs here instead of a temporary directory
  --allow-motion       ⚠  enable the motion RPCs. Read ../robot-stack/SAFETY.md first.
  --dry-run            print the three commands this would run, one per line, and stop
  --                   everything after this goes to robot_driver.py unchanged
  -h, --help

Ctrl-C stops all three. Nothing is left listening on 8080 or 8800.
USAGE
}

die() { printf '\n\033[1;31mstart-dashboard: %s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; echo >&2; exit 2; }
say() { printf '\033[1;36m·\033[0m %s\n' "$1"; }

# ---- arguments --------------------------------------------------------------------------
# `--robot --port 8090` must not take "--port" as the hostname and fail later at a camera
# URL of http://--port:8801/. Checked HERE rather than after the loop: after it, the
# swallowed option's own argument is the next token and the loop dies on that instead, so
# a post-loop guard is one that almost never fires.
# NOT a command substitution returning the value — `die` inside one exits the subshell and
# the script carries on with an empty variable.
need() {  # need <flag> <value>
  case "${2:-}" in
    "" ) die "$1 needs a value and is the last argument." ;;
    -* ) die "$1 was given '$2', which is another option." \
             "Every option here except --allow-motion, --dry-run and --help takes a value." ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --robot)        need --robot "${2:-}"; ROBOT="$2"; shift 2 ;;
    --camera-url)   need --camera-url "${2:-}"; CAMERA_URL="$2"; shift 2 ;;
    --platform)     need --platform "${2:-}"; PLATFORM="$2"; shift 2 ;;
    --simulate)     SIMULATE=1; shift ;;
    --package)      need --package "${2:-}"; PACKAGE="$2"; shift 2 ;;
    --models-dir)   need --models-dir "${2:-}"; MODELS_DIR="$2"; shift 2 ;;
    --device-id)    need --device-id "${2:-}"; DEVICE_ID="$2"; shift 2 ;;
    --port)         need --port "${2:-}"; PORT="$2"; shift 2 ;;
    --model-port)   need --model-port "${2:-}"; MODEL_PORT="$2"; shift 2 ;;
    --host)         need --host "${2:-}"; BIND="$2"; shift 2 ;;
    --python)       need --python "${2:-}"; PYTHON="$2"; shift 2 ;;
    --log-dir)      need --log-dir "${2:-}"; LOG_DIR="$2"; shift 2 ;;
    --allow-motion) ALLOW_MOTION=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --)             shift; EXTRA=("$@"); break ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unrecognised argument: $1" "Run --help for the list." ;;
  esac
done

if [ -n "$ROBOT" ] && [ -n "$CAMERA_URL" ]; then
  die "--robot and --camera-url both given, and they set the same thing." \
      "--robot HOST is shorthand for --camera-url http://HOST:8801/. Pick one."
fi
if [ -n "$ROBOT" ]; then
  CAMERA_URL="http://${ROBOT}:8801/"
  [ -n "$PLATFORM" ] || PLATFORM=go2
  SIMULATE=1
fi
[ -n "$PLATFORM" ] || PLATFORM=sim
SIMULATE="${SIMULATE:-0}"
[ -n "$MODELS_DIR" ] || MODELS_DIR="$PACKAGE/models"

case "$PLATFORM" in go2|lite3|sim) : ;; *) die "--platform must be go2, lite3 or sim (got '$PLATFORM')" ;; esac

# ---- 1. the interpreter -----------------------------------------------------------------
# This is the first thing anyone hits and it must not be a traceback. `python3` on macOS is
# the Command Line Tools 3.9.6 whatever else is installed, and Device Connect needs >= 3.11.
version_of() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
is_ok()      { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; }
is_ready()   { is_ok "$1" && "$1" -c 'import device_connect_edge' >/dev/null 2>&1; }

# TWO passes, and the order matters. A machine can easily hold python3.11, 3.12 and 3.13
# with Device Connect installed into exactly one of them; picking the NEWEST >= 3.11 would
# then be correct on version and wrong in practice, and the operator's first experience
# would be a pip line for an interpreter they have never used. So: prefer one that can
# already import device_connect_edge; fall back to any >= 3.11 and ask for the install.
pick_python() {
  local cand
  for cand in python3.14 python3.13 python3.12 python3.11 python3; do
    command -v "$cand" >/dev/null 2>&1 && is_ready "$cand" && { echo "$cand"; return 0; }
  done
  for cand in python3.14 python3.13 python3.12 python3.11 python3; do
    command -v "$cand" >/dev/null 2>&1 && is_ok "$cand" && { echo "$cand"; return 0; }
  done
  return 1
}

if [ -n "$PYTHON" ]; then
  command -v "$PYTHON" >/dev/null 2>&1 || die "--python '$PYTHON' is not an executable on PATH."
  if ! is_ok "$PYTHON"; then
    found="$(version_of "$PYTHON")"
    suggest="$(pick_python || true)"
    if [ -n "$suggest" ]; then
      die "'$PYTHON' is Python ${found:-unknown}; device-connect-edge needs >= 3.11." \
          "This machine has one that works. Run instead:" \
          "" "    $0 --python $suggest" ""
    fi
    die "'$PYTHON' is Python ${found:-unknown}; device-connect-edge needs >= 3.11." \
        "No Python >= 3.11 was found on PATH. Install one, e.g.:" \
        "" "    brew install python@3.11        # macOS" \
        "    sudo apt install python3.11     # Debian/Ubuntu" "" \
        "This cannot be worked around with a virtualenv: a venv is built FROM an" \
        "interpreter and cannot supply a version the machine does not have."
  fi
else
  PYTHON="$(pick_python || true)"
  if [ -z "$PYTHON" ]; then
    have="$(command -v python3 >/dev/null 2>&1 && version_of python3 || echo "none")"
    die "no Python >= 3.11 on PATH — 'python3' is ${have}, and device-connect-edge needs 3.11." \
        "Install one and re-run, or point at one you already have:" \
        "" "    brew install python@3.11        # macOS" \
        "    sudo apt install python3.11     # Debian/Ubuntu" \
        "    $0 --python /path/to/python3.11" "" \
        "A virtualenv will not help: it is built FROM an interpreter and cannot supply a" \
        "version the machine does not have. This is also why the driver does not run on" \
        "the Go2 — see README.md, 'The robot cannot host the driver'."
  fi
fi
PYV="$(version_of "$PYTHON")"

# ---- 2. the packages --------------------------------------------------------------------
# Named by IMPORT, checked with the interpreter that will run them, and reported as one
# pip line rather than as three tracebacks twenty seconds apart. aiohttp and numpy are on
# this list because nothing installs them for you: device-connect-agent-tools depends only
# on device-connect-edge, which depends on eclipse-zenoh, nats-py, nkeys, pydantic, pyyaml.
missing_pip=""
missing_mod=""
check() {  # check <import name> <pip name>
  "$PYTHON" -c "import $1" >/dev/null 2>&1 && return 0
  missing_mod="$missing_mod $1"; missing_pip="$missing_pip $2"
}
check device_connect_edge        device-connect-edge
check device_connect_agent_tools device-connect-agent-tools
check aiohttp                    aiohttp
check numpy                      numpy
if [ -n "$missing_pip" ]; then
  die "$PYTHON ($PYV) cannot import:$missing_mod" \
      "Install them into that interpreter — this exact line, not a shorter one:" \
      "" "    $PYTHON -m pip install$missing_pip" "" \
      "If your index is an internal mirror that 404s on these, add:" \
      "    --index-url https://pypi.org/simple"
fi
# Pillow is a warning, not a refusal: only the SYNTHETIC sim camera needs it, and a real
# camera or --camera-url does not. Without it the sim viewport is a stated "unavailable"
# rather than a black rectangle, which is a working dashboard with one panel missing.
PILLOW_NOTE=""
if [ "$PLATFORM" = sim ] && [ -z "$CAMERA_URL" ] && ! "$PYTHON" -c "import PIL" >/dev/null 2>&1; then
  PILLOW_NOTE="Pillow is not installed, so the sim camera will report itself unavailable. $PYTHON -m pip install Pillow"
fi

# ---- 3. the paths -----------------------------------------------------------------------
[ -d "$PACKAGE" ]    || die "--package '$PACKAGE' is not a directory." "It is the directory holding config.json and models/."
[ -d "$MODELS_DIR" ] || die "--models-dir '$MODELS_DIR' is not a directory." \
    "It is what the checkpoint server serves, and it defaults to <package>/models." \
    "Pass --models-dir at a directory of .npz checkpoints."
[ -f "$PACKAGE/config.json" ] || die "'$PACKAGE' has no config.json, so it is not a policy package."

# ---- 4. the ports -----------------------------------------------------------------------
# Refuse EARLY and NAME the process. `Address already in use` arrives from inside a Python
# server after the other two have started, and the usual reading of it is "something is
# wrong with the port", not "my last demo is still running". This has happened.
# DETECTED BY BINDING, not by lsof: lsof is absent on plenty of Linux images, and a
# detector that silently answers "free" on those machines is worse than none — the failure
# it exists to prevent would come back only on the machines that cannot see it. lsof is
# used only to NAME the holder, which is a nicety, and its absence costs the pid and
# nothing else.
port_free() { "$PYTHON" -c 'import socket, sys
s = socket.socket()
try:
    s.bind((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    s.close()' "$BIND" "$1"; }
port_holder() {
  if command -v lsof >/dev/null 2>&1; then
    # -Fpc asks for the pid and command fields, and lsof emits the fd field anyway, so
    # the `f` lines are dropped explicitly — one of them read "pid 93649 (Python) f32".
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fpc 2>/dev/null | grep -E '^[pc]' | tr '\n' ' ' \
      | sed 's/p\([0-9]*\) *c\([^ ]*\)/pid \1 (\2)/g' | sed 's/ *$//'
  fi
}
for pair in "$PORT:the dashboard" "$MODEL_PORT:the checkpoint server"; do
  p="${pair%%:*}"; what="${pair#*:}"
  if ! port_free "$p"; then
    holder="$(port_holder "$p")"
    kill_line="    (run lsof -nP -iTCP:$p -sTCP:LISTEN to find it)"
    if [ -n "$holder" ]; then
      first="${holder#*pid }"          # "pid 93649 (Python) pid 93650 (Python)" -> the first
      kill_line="    kill ${first%% *}"
    fi
    die "port $p ($what) is already in use${holder:+ by $holder}" \
        "That is almost always a previous run of this script or of the three commands it" \
        "replaces. Stop it, or choose another port:" \
        "" "$kill_line" \
        "    $0 --port $((PORT + 10)) --model-port $((MODEL_PORT + 10))" ""
  fi
done

# ---- 5. the camera, if one was asked for ------------------------------------------------
# A warning, not a refusal: the dashboard is useful without a camera, and the driver reports
# the same failure in capabilities.camera.error. But naming the robot-side command here
# saves the round trip through a black viewport.
CAMERA_NOTE=""
if [ -n "$CAMERA_URL" ]; then
  if command -v curl >/dev/null 2>&1 && ! curl -sf -m 4 -o /dev/null "$CAMERA_URL"; then
    CAMERA_NOTE="$CAMERA_URL is not answering. On the Go2 that server is started by hand:
      ssh unitree@${ROBOT:-<robot>}
      source /home/unitree/mappo-main/robot-stack/unitree/go2/install/setup_env.sh
      export PYTHONPATH=/home/unitree/deps:/home/unitree/unitree_sdk2_python
      setsid nohup python3 /home/unitree/go2_frame_server.py \\
             > /home/unitree/frame_server.log 2>&1 < /dev/null &
    Without the source line it SEGFAULTS at the first VideoClient RPC (measured, rc 139),
    so a connection refused here is the expected shape of a missing LD_LIBRARY_PATH.
    Everything else on the dashboard works; only the viewport is affected."
  fi
fi

# ---- 6. run -----------------------------------------------------------------------------
if [ -n "$LOG_DIR" ]; then mkdir -p "$LOG_DIR"; else LOG_DIR="$(mktemp -d -t mappo-dashboard)"; fi
PIDS=(); NAMES=(); LOGS=()

# ONE cleanup path, on every exit: Ctrl-C, a child dying, and a normal return all arrive
# here. Three processes started together must stop together, or the next run of this script
# refuses on a port and the operator's next move is to hunt PIDs.
CLEANED=0
cleanup() {
  [ "$CLEANED" -eq 1 ] && return 0
  CLEANED=1
  trap - EXIT INT TERM
  # bash 3.2 with `set -u` errors on "${arr[@]}" for an EMPTY array, and this trap is armed
  # before the first child starts — so a refusal between the two would die in the cleanup.
  [ "${#PIDS[@]}" -eq 0 ] && return 0
  local i pid
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    kill -TERM "$pid" 2>/dev/null && say "stopping ${NAMES[$i]} (pid $pid)"
  done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    local alive=0
    for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 0.5
  done
  # SIGTERM first, always, and only then SIGKILL: robot_driver.py's motion worker installs
  # a damp on SIGTERM, and SportClient.Move has no dead-man timeout, so a bare kill of a
  # walking robot leaves the last velocity latched on the bus.
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null; done
  wait 2>/dev/null
}
# ⚠️ A bash signal trap RETURNS to where it was interrupted; it does not exit. With one
# `trap cleanup EXIT INT TERM`, Ctrl-C stopped all three correctly and then RESUMED the
# watchdog loop below, which found a dead child and reported "checkpoint server exited"
# with twenty lines of its log — an operator's clean exit reading as a crash. Measured, and
# the reason the signal path is a separate handler that exits.
on_signal() { echo; cleanup; exit 130; }
trap cleanup EXIT
trap on_signal INT TERM

start() {  # start <name> <logfile> <argv...>
  local name="$1" log="$2"; shift 2
  "$@" > "$log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name"); LOGS+=("$log")
}

echo
say "python      $PYTHON  ($PYV)"
say "package     $PACKAGE"
say "logs        $LOG_DIR"
case "$BIND" in
  127.0.0.1|localhost|::1) : ;;
  *) printf '\033[1;33m!\033[0m %s\n' \
       "bound to $BIND, not loopback. THIS DASHBOARD HAS NO LOGIN: anyone who can reach
    port $PORT can drive any robot on the mesh that was started with motion enabled." ;;
esac
[ -n "$PILLOW_NOTE" ] && printf '\033[1;33m!\033[0m %s\n' "$PILLOW_NOTE"
[ -n "$CAMERA_NOTE" ] && printf '\033[1;33m!\033[0m %s\n' "$CAMERA_NOTE"

model_args=(--models-dir "$MODELS_DIR" --host "$BIND" --port "$MODEL_PORT"
            --emit-sources "$LOG_DIR/sources.json" --label "local checkpoint server")

driver_args=(--platform "$PLATFORM" --package "$PACKAGE" --model-sources "$LOG_DIR/sources.json")
[ "$SIMULATE" -eq 1 ] && driver_args+=(--simulate)
[ -n "$CAMERA_URL" ] && driver_args+=(--camera-url "$CAMERA_URL")
[ -n "$DEVICE_ID" ]  && driver_args+=(--device-id "$DEVICE_ID")
# The ONLY place --allow-motion is added, and it is gated on the literal argument.
[ "$ALLOW_MOTION" -eq 1 ] && driver_args+=(--allow-motion)
[ "${#EXTRA[@]}" -gt 0 ] && driver_args+=("${EXTRA[@]}")


server_args=(--host "$BIND" --port "$PORT")

# --dry-run prints the arrays that start() is about to be handed, from the same variables,
# so it cannot describe a launch different from the one it would perform. What it does NOT
# cover is start() itself, and that is stated rather than implied.
if [ "$DRY_RUN" -eq 1 ]; then
  printf 'model_server %s %s\n' "$PYTHON" "${model_args[*]}"
  printf 'robot_driver %s %s\n' "$PYTHON" "${driver_args[*]}"
  printf 'server %s %s\n' "$PYTHON" "${server_args[*]}"
  exit 0
fi

start "checkpoint server" "$LOG_DIR/model_server.log" \
      "$PYTHON" "$HERE/model_server.py" "${model_args[@]}"

# The checkpoint server writes sources.json at startup; the driver reads it once, at
# construction. Starting them in the same instant gives the driver an empty Source picker
# about one run in three.
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$LOG_DIR/sources.json" ] && break; sleep 0.3; done

start "driver" "$LOG_DIR/driver.log" "$PYTHON" "$HERE/robot_driver.py" "${driver_args[@]}"
start "dashboard" "$LOG_DIR/server.log" "$PYTHON" "$HERE/server.py" "${server_args[@]}"

# Wait for the page rather than for a fixed sleep, so the URL is printed when it works.
ready=0
for _ in $(seq 1 40); do
  if command -v curl >/dev/null 2>&1; then
    curl -sf -m 2 -o /dev/null "http://${BIND}:${PORT}/" && { ready=1; break; }
  else
    ready=1; break
  fi
  for i in "${!PIDS[@]}"; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || {
      echo; printf '\033[1;31mstart-dashboard: %s exited during startup. The last of %s:\033[0m\n' "${NAMES[$i]}" "${LOGS[$i]}"
      tail -20 "${LOGS[$i]}"
      exit 1
    }
  done
  sleep 0.5
done

echo
if [ "$ready" -eq 1 ]; then
  printf '\033[1;32m  http://%s:%s\033[0m\n' "$BIND" "$PORT"
else
  printf '\033[1;33m  http://%s:%s  (not answering yet — see %s/server.log)\033[0m\n' "$BIND" "$PORT" "$LOG_DIR"
fi
echo
say "fleet       $PLATFORM$( [ "$SIMULATE" -eq 1 ] && echo ' (simulated pose)')$( [ -n "$CAMERA_URL" ] && echo ", live camera from $CAMERA_URL")"
if [ "$ALLOW_MOTION" -eq 1 ]; then
  printf '\033[1;31m  MOTION IS ENABLED — robot-stack/SAFETY.md applies: clear area, operator on the abort.\033[0m\n'
else
  say "motion      DISABLED (status and checkpoints only). Add --allow-motion to change that."
fi
say "Ctrl-C stops all three."
echo

# Hold, and take everything down if any one of them dies — a dashboard with no driver and a
# driver with no dashboard are both worse than a stopped demo, because both look up.
while true; do
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo; printf '\033[1;31mstart-dashboard: %s exited. Stopping the other two. The last of %s:\033[0m\n' "${NAMES[$i]}" "${LOGS[$i]}"
      tail -20 "${LOGS[$i]}"
      exit 1
    fi
  done
  sleep 1
done
