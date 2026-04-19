#!/usr/bin/env bash
# Shared helpers for server-related shell scripts.

if [[ -n "${RHOS_COBOT_SERVER_ENV_LOADED:-}" ]]; then
  return 0
fi
RHOS_COBOT_SERVER_ENV_LOADED=1

SERVER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_REPO_ROOT="$(cd "$SERVER_HELPER_DIR/.." && pwd)"
SERVER_CONFIG="$SERVER_REPO_ROOT/config/servers.toml"
SERVER_TASK_PROMPT_CATALOG="$SERVER_REPO_ROOT/config/task_prompts.json"

server_cfg() {
  python3 "$SERVER_HELPER_DIR/_read_toml.py" "$SERVER_CONFIG" "$1"
}

server_cfg_optional() {
  python3 "$SERVER_HELPER_DIR/_read_toml.py" "$SERVER_CONFIG" "$1" 2>/dev/null || true
}

server_require_config() {
  if [[ -f "$SERVER_CONFIG" ]]; then
    return 0
  fi

  echo "Missing config file: $SERVER_CONFIG" >&2
  echo "Copy config/servers.example.toml to config/servers.toml and fill in your deployment settings first." >&2
  return 1
}

server_default_python_cmd() {
  local candidate="$SERVER_REPO_ROOT/examples/piper_real/.venv/bin/python"
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "python3"
  fi
}

server_resolve_python_cmd() {
  local cmd="$1"

  if [[ "$cmd" == */* ]]; then
    if [[ ! -x "$cmd" ]]; then
      echo "Python interpreter is not executable: $cmd" >&2
      return 1
    fi
    printf '%s\n' "$cmd"
    return 0
  fi

  if command -v "$cmd" >/dev/null 2>&1; then
    printf '%s\n' "$cmd"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi

  echo "Python interpreter '$cmd' is not on PATH." >&2
  return 1
}

server_default_openpi_root() {
  printf '%s/%s\n' "$SERVER_REPO_ROOT" "$(server_cfg pi0.local.openpi_root)"
}

server_resolve_openpi_client_src() {
  local openpi_root="$1"
  local openpi_client_src="${2:-$openpi_root/packages/openpi-client/src}"

  if [[ ! -d "$openpi_client_src" ]]; then
    echo "openpi_client source path does not exist: $openpi_client_src" >&2
    echo "Set OPENPI_ROOT or OPENPI_CLIENT_SRC explicitly." >&2
    return 1
  fi

  printf '%s\n' "$openpi_client_src"
}

server_export_openpi_pythonpath() {
  local openpi_client_src="$1"
  export PYTHONPATH="$openpi_client_src${PYTHONPATH:+:$PYTHONPATH}"
}

server_lookup_task_prompt() {
  local task_name="$1"

  if [[ -z "$task_name" ]]; then
    echo "TASK_NAME must be non-empty when resolving a catalog prompt." >&2
    return 1
  fi

  python3 "$SERVER_HELPER_DIR/_resolve_task_prompt.py" \
    "$SERVER_TASK_PROMPT_CATALOG" \
    "$task_name"
}

server_remote_ssh_target() {
  local service_name="$1"
  local ssh_target

  ssh_target="$(server_cfg_optional "$service_name.remote.ssh_target")"
  if [[ -n "$ssh_target" ]]; then
    printf '%s\n' "$ssh_target"
    return 0
  fi

  server_cfg "$service_name.remote.host"
}
