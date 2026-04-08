#!/usr/bin/env bash
# Check whether the planner server and the Pi0 policy server are reachable.
#
# Usage:
#   bash scripts/check_servers.sh
#   bash scripts/check_servers.sh local
#   VLLM_HOST=192.168.3.123 PI0_HOST=192.168.3.101 bash scripts/check_servers.sh remote

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_server_env.sh"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/check_servers.sh [local|remote]

Checks:
  1. vLLM planner HTTP endpoint /v1/models
  2. Pi0 policy websocket handshake + reset

Environment overrides:
  VLLM_HOST         Planner host to probe
                    default: 127.0.0.1 for local; value from config for remote
  VLLM_PORT         Planner port
                    default: value from config/servers.toml
  VLLM_MODEL        Expected served model name
                    default: config vllm.<mode>.served_model_name
  PI0_HOST          Pi0 websocket host to probe
                    default: 127.0.0.1 for local; value from config for remote
  PI0_PORT          Pi0 websocket port
                    default: value from config/servers.toml
  PYTHON_CMD        Python interpreter with websockets dependency
                    default: examples/piper_real/.venv/bin/python, fallback: python3
  OPENPI_ROOT       OpenPI repo root for openpi_client
                    default: config pi0.local.openpi_root
  OPENPI_CLIENT_SRC Explicit openpi_client source path
                    default: \$OPENPI_ROOT/packages/openpi-client/src
  TIMEOUT_SEC       Network timeout per check
                    default: 5

Examples:
  bash scripts/check_servers.sh
  bash scripts/check_servers.sh local
  VLLM_HOST=192.168.3.123 PI0_HOST=192.168.3.101 bash scripts/check_servers.sh remote
EOF
}

MODE="${1:-local}"
case "$MODE" in
  local|remote)
    ;;
  -h|--help)
    print_usage
    exit 0
    ;;
  *)
    print_usage >&2
    exit 2
    ;;
esac

server_require_config

TIMEOUT_SEC="${TIMEOUT_SEC:-5}"
VLLM_PORT="${VLLM_PORT:-$(server_cfg vllm.port)}"
PI0_PORT="${PI0_PORT:-$(server_cfg pi0.port)}"

if [[ "$MODE" == "local" ]]; then
  VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
  PI0_HOST="${PI0_HOST:-127.0.0.1}"
  VLLM_MODEL="${VLLM_MODEL:-$(server_cfg vllm.local.served_model_name)}"
else
  VLLM_HOST="${VLLM_HOST:-$(server_cfg vllm.remote.host)}"
  PI0_HOST="${PI0_HOST:-$(server_cfg pi0.remote.host)}"
  VLLM_MODEL="${VLLM_MODEL:-$(server_cfg vllm.remote.served_model_name)}"
fi

PYTHON_CMD="${PYTHON_CMD:-$(server_default_python_cmd)}"
OPENPI_ROOT="${OPENPI_ROOT:-$(server_default_openpi_root)}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-}"

PYTHON_CMD="$(server_resolve_python_cmd "$PYTHON_CMD")"
OPENPI_CLIENT_SRC="$(server_resolve_openpi_client_src "$OPENPI_ROOT" "$OPENPI_CLIENT_SRC")"
server_export_openpi_pythonpath "$OPENPI_CLIENT_SRC"
cd "$SERVER_REPO_ROOT"

VLLM_BASE_URL="http://$VLLM_HOST:$VLLM_PORT/v1"

echo "Mode: $MODE"
echo "Planner check: $VLLM_BASE_URL"
echo "Expected planner model: $VLLM_MODEL"
echo "Pi0 check: ws://$PI0_HOST:$PI0_PORT"
echo "Python: $PYTHON_CMD"
echo "openpi_client src: $OPENPI_CLIENT_SRC"
echo

"$PYTHON_CMD" -m examples.piper_real.server_checks \
  --planner-base-url "$VLLM_BASE_URL" \
  --planner-model "$VLLM_MODEL" \
  --pi0-host "$PI0_HOST" \
  --pi0-port "$PI0_PORT" \
  --timeout-sec "$TIMEOUT_SEC"

echo
echo "Both servers look healthy."
