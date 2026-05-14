#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/agilex/rhos_cobot}"
PIPER_DIR="${PIPER_DIR:-/home/agilex/cobot_magic/Piper_ros_private-ros-noetic}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/headless_init}"
CAMERA_LAUNCH="${CAMERA_LAUNCH:-${REPO_DIR}/scripts/launch/multi_camera_rgb.launch}"
SUDO_PASSWORD="${SUDO_PASSWORD:-agx}"

DRY_RUN=0
START_CAMERA=1
CONFIGURE_CAN=1
START_PIPER=0
ENABLE_DEPTH=0

usage() {
  cat <<'USAGE'
Usage: scripts/init_headless.sh [options]

Headless startup for data collection. Starts RGB-only cameras without opening
viewer windows, optionally configures Piper CAN, and can optionally start the
Piper record-mode driver.

Options:
  --dry-run        Print commands without running hardware actions.
  --no-camera      Skip RGB camera roslaunch.
  --no-can         Skip Piper CAN configuration.
  --with-depth     Enable depth topics while keeping point cloud disabled.
  --start-piper    Also start roslaunch piper start_ms_piper.launch mode:=0.
  -h, --help       Show this help.

Environment:
  REPO_DIR         Default: /home/agilex/rhos_cobot
  PIPER_DIR        Default: /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
  LOG_DIR          Default: $REPO_DIR/logs/headless_init
  CAMERA_LAUNCH    Default: $REPO_DIR/scripts/launch/multi_camera_rgb.launch
  SUDO_PASSWORD    Default: agx
USAGE
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --no-camera)
      START_CAMERA=0
      ;;
    --no-can)
      CONFIGURE_CAN=0
      ;;
    --with-depth)
      ENABLE_DEPTH=1
      ;;
    --start-piper)
      START_PIPER=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[headless] ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

run() {
  echo "[headless] $*"
  if ((DRY_RUN)); then
    echo "[headless] DRY-RUN: skipped"
    return 0
  fi
  "$@"
}

start_background() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  echo "[headless] starting ${name}; log=${log_file}"
  if ((DRY_RUN)); then
    echo "[headless] DRY-RUN: setsid nohup $* > ${log_file} 2>&1 &"
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  setsid nohup "$@" >"${log_file}" 2>&1 </dev/null &
  echo "$!" >"${pid_file}"
  echo "[headless] ${name} pid=$(cat "${pid_file}")"
}

mkdir -p "${LOG_DIR}"

if ((START_CAMERA)); then
  camera_args=()
  if ((ENABLE_DEPTH)); then
    camera_args+=(enable_depth:=true enable_point_cloud:=false depth_align:=false)
  fi
  start_background camera_rgb roslaunch "${CAMERA_LAUNCH}" "${camera_args[@]}"
else
  echo "[headless] skipping camera launch"
fi

if ((CONFIGURE_CAN)); then
  if ((DRY_RUN)); then
    echo "[headless] DRY-RUN: cd ${PIPER_DIR} && echo ****** | sudo -S bash ./can_config.sh"
  else
    echo "[headless] configuring Piper CAN in ${PIPER_DIR}"
    (
      cd "${PIPER_DIR}"
      echo "${SUDO_PASSWORD}" | sudo -S bash ./can_config.sh
    )
  fi
else
  echo "[headless] skipping CAN configuration"
fi

if ((START_PIPER)); then
  start_background piper_record roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false
else
  echo "[headless] piper record launch not started; run init_record separately if needed"
fi

echo "[headless] startup commands issued."
echo "[headless] logs: ${LOG_DIR}"
