'''
Usage:
python scripts/qz_pi0_server.py --create --image <image> --model-id <qz-model-id>
python scripts/qz_pi0_server.py --health
python scripts/qz_pi0_server.py --stop

Environment:
QZ_USERNAME / QZ_PASSWORD are required for create/stop/query.
QZ_PI0_IMAGE and QZ_PI0_MODEL_ID are required for create unless passed by CLI.
'''

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any

import requests


def _parse_json_response(resp: requests.Response, *, context: str) -> dict:
    """Parse response JSON with actionable diagnostics for non-JSON payloads."""
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError as exc:
        content_type = resp.headers.get("Content-Type", "")
        preview = (resp.text or "")[:300].replace("\n", " ").strip()
        raise RuntimeError(
            f"{context} returned non-JSON response "
            f"(status={resp.status_code}, content_type={content_type!r}, body_preview={preview!r})"
        ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_STATE_FILE = PROJECT_ROOT / "config" / "pi0_server_state.json"
QWEN_STATE_FILE = PROJECT_ROOT / "config" / "vllm_server_state.json"
ENV_FILE = PROJECT_ROOT / ".env"
SERVERS_FILE = PROJECT_ROOT / "config" / "servers.toml"

SEQ = 0
INFERENCE_ID = ""
API_KEY = ""
STATE_FILE = DEFAULT_STATE_FILE


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file (only for keys not already set)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ENV_FILE)

# Use the explicit QZ proxy by default and ignore shell proxy variables.
_PROXY_URL = os.environ.get("QZ_PROXY_URL", "http://127.0.0.1:8888")
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.proxies = {"http": _PROXY_URL, "https": _PROXY_URL}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid state file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"State file must contain a JSON object: {path}")
    return data


def _state_api_key_fallback() -> str:
    if QWEN_STATE_FILE.exists():
        data = _read_json(QWEN_STATE_FILE)
        api_key = data.get("api_key", "")
        if isinstance(api_key, str):
            return api_key
    return ""


def _load_state(path: Path) -> tuple[int, str, str]:
    if not path.exists():
        return 0, "", os.environ.get("QZ_API_KEY", "") or _state_api_key_fallback()

    data = _read_json(path)
    seq = data.get("seq", 0)
    inference_id = data.get("inference_id", "")
    api_key = data.get("api_key", "") or os.environ.get("QZ_API_KEY", "") or _state_api_key_fallback()

    if not isinstance(seq, int):
        raise ValueError("State key 'seq' must be int")
    if not isinstance(inference_id, str):
        raise ValueError("State key 'inference_id' must be str")
    if not isinstance(api_key, str):
        raise ValueError("State key 'api_key' must be str")
    return seq, inference_id, api_key


def _save_state(path: Path, seq: int, inference_id: str, api_key: str) -> None:
    if not isinstance(seq, int):
        raise ValueError("'seq' must be int")
    if not isinstance(inference_id, str):
        raise ValueError("'inference_id' must be str")
    if not api_key:
        api_key = _state_api_key_fallback()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"seq": seq, "inference_id": inference_id, "api_key": api_key},
            indent=2,
        )
    )
    print(f"Saved seq/inference_id/api_key to {path}.")


def _config_value(path: str, default: str = "") -> str:
    if not SERVERS_FILE.exists():
        return default
    try:
        import tomllib

        data: Any = tomllib.loads(SERVERS_FILE.read_text())
    except Exception:
        return default

    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    if current is None:
        return default
    return str(current)


def _env_or_config(name: str, config_path: str, default: str = "") -> str:
    return os.environ.get(name) or _config_value(config_path, default)


def get(url: str, headers: dict | None = None, **kwargs):
    return _SESSION.get(url, headers=headers, **kwargs)


def post(
    url: str,
    headers: dict | None = None,
    json: dict | None = None,
    **kwargs,
):
    return _SESSION.post(url, headers=headers, json=json, **kwargs)


def _check_response(resp: requests.Response) -> dict | None:
    """Validate OpenAPI response and return data or None."""
    if resp.status_code != 200:
        print(f"HTTP error: {resp.status_code}")
        print(resp.text[:300])
        return None
    json_data = _parse_json_response(resp, context="OpenAPI request")
    if json_data.get("code", -1) != 0:
        print(
            f"API error: code={json_data.get('code')}, msg={json_data.get('msg', 'unknown')}"
        )
        return None
    data = json_data.get("data", {})
    return data if isinstance(data, dict) else {}


def _require(value: str, name: str) -> str:
    if value:
        return value
    raise RuntimeError(f"{name} is required")


def _int_value(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def get_host(seq: int | None = None, *, domain_prefix: str = "pi0") -> str:
    actual_seq = SEQ if seq is None else seq
    return f"{domain_prefix}-{actual_seq}-inf.openapi-qb.sii.edu.cn"


def get_endpoint(seq: int | None = None, *, domain_prefix: str = "pi0") -> str:
    return f"wss://{get_host(seq, domain_prefix=domain_prefix)}"


def _require_seq(seq: int) -> int:
    if seq > 0:
        return seq
    raise RuntimeError(
        f"pi0 seq is missing in {STATE_FILE}. Run scripts/qz_pi0_server.py --create first."
    )


def _build_command(
    *,
    openpi_root: str,
    policy_config: str,
    default_prompt: str,
    record: bool,
) -> str:
    if os.environ.get("QZ_PI0_COMMAND"):
        return os.environ["QZ_PI0_COMMAND"]

    args = [
        "uv",
        "run",
        "--active",
        "scripts/serve_policy.py",
        "--port=${PORT}",
    ]
    if default_prompt:
        args.append(f"--default-prompt={default_prompt}")
    if record:
        args.append("--record")
    args.extend(
        [
            "policy:checkpoint",
            f"--policy.config={policy_config}",
            "--policy.dir=${MODEL_PATH}",
        ]
    )

    passthrough_args = {"--port=${PORT}", "--policy.dir=${MODEL_PATH}"}
    quoted_args = " ".join(arg if arg in passthrough_args else shlex.quote(arg) for arg in args)
    return (
        "export LD_LIBRARY_PATH=/home/Xtrainer/anaconda3/envs/brs/lib:"
        "${LD_LIBRARY_PATH:-} && "
        f"cd {shlex.quote(openpi_root)} && "
        "source .venv/bin/activate && "
        f"{quoted_args}"
    )


def health_check(
    uri: str,
    *,
    api_key: str = "",
    timeout_sec: float = 10.0,
):
    """Check OpenPI websocket handshake and reset on a public QZ endpoint."""
    try:
        import websockets.sync.client

        from openpi_client import msgpack_numpy
    except ImportError as exc:
        raise RuntimeError(
            "health check requires websockets and openpi_client in the active Python environment"
        ) from exc

    header_name = os.environ.get("QZ_PI0_API_KEY_HEADER", "Api-Key")
    headers = {"Authorization": f"{header_name} {api_key}"} if api_key else None
    conn = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=timeout_sec,
        additional_headers=headers,
    )
    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        conn.send(msgpack_numpy.packb({"type": "reset"}))
        response = conn.recv()
        if isinstance(response, str):
            raise RuntimeError(f"unexpected text response from {uri}: {response}")
        msgpack_numpy.unpackb(response)
    finally:
        conn.close()
    print("pi0 metadata:", metadata)
    print("pi0 reset: ok")
    return metadata


def query_token(username: str, password: str) -> str:
    resp = post(
        "https://qz.sii.edu.cn/auth/token",
        json={"password": password, "username": username},
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token request failed with HTTP {resp.status_code}: {resp.text[:300]!r}"
        )
    print("Successfully authenticated.")
    return (
        _parse_json_response(resp, context="Token request")
        .get("data", {})
        .get("access_token", "")
    )


def query(inference_id: str, token: str) -> None:
    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/detail",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_serving_id": inference_id},
    )
    data = _check_response(resp)
    print(data)


def stop(inference_id: str, token: str) -> None:
    _require(inference_id, "inference_id in state file")
    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/stop",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_serving_id": inference_id},
    )
    data = _check_response(resp)
    print(data)


def create(token: str, args: argparse.Namespace) -> None:
    global API_KEY
    global INFERENCE_ID
    global SEQ

    SEQ = int(time.time())
    policy_config = args.policy_config or _env_or_config(
        "QZ_PI0_POLICY_CONFIG", "pi0.remote.policy_config"
    )
    if not policy_config:
        policy_config = _config_value("pi0.policy_config", "")

    port = args.port or _int_value(
        os.environ.get("QZ_PI0_PORT", "") or _config_value("pi0.port", "8001"),
        "port",
    )
    openpi_root = args.openpi_root or os.environ.get(
        "QZ_PI0_OPENPI_ROOT",
        "/inspire/ssd/project/robot-reasoning/xiangyushun-p-xiangyushun/all-in-one-vla-inference/third_party/openpi",
    )
    image = args.image or os.environ.get("QZ_PI0_IMAGE", "")
    model_id = args.model_id or os.environ.get("QZ_PI0_MODEL_ID", "")
    model_version = args.model_version or _int_value(
        os.environ.get("QZ_PI0_MODEL_VERSION", "1"), "QZ_PI0_MODEL_VERSION"
    )
    domain_prefix = args.domain_prefix or os.environ.get("QZ_PI0_DOMAIN_PREFIX", "pi0")
    default_prompt = args.default_prompt or os.environ.get("QZ_PI0_DEFAULT_PROMPT", "")
    record = args.record or os.environ.get("QZ_PI0_RECORD", "0").lower() in {
        "1",
        "true",
        "yes",
    }

    payload = {
        "command": _build_command(
            openpi_root=openpi_root,
            policy_config=_require(
                policy_config,
                "QZ_PI0_POLICY_CONFIG or pi0.remote.policy_config",
            ),
            default_prompt=default_prompt,
            record=record,
        ),
        "custom_domain": f"{domain_prefix}-{SEQ}-inf",
        "image": _require(image, "--image or QZ_PI0_IMAGE"),
        "image_type": os.environ.get("QZ_PI0_IMAGE_TYPE", "SOURCE_PUBLIC"),
        "logic_compute_group_id": os.environ.get(
            "QZ_PI0_LOGIC_COMPUTE_GROUP_ID",
            "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
        ),
        "model_id": _require(model_id, "--model-id or QZ_PI0_MODEL_ID"),
        "model_version": model_version,
        "name": f"{domain_prefix}-{SEQ}",
        "port": port,
        "project_id": os.environ.get(
            "QZ_PI0_PROJECT_ID",
            "project-0ca5ec4c-7cf6-4700-9d2e-cd95c04e6e8e",
        ),
        "replicas": _int_value(
            os.environ.get("QZ_PI0_REPLICAS", "1"),
            "QZ_PI0_REPLICAS",
        ),
        "spec_id": args.spec_id or os.environ.get("QZ_PI0_SPEC_ID", "c18m200g1t"),
        "task_priority": _int_value(
            os.environ.get("QZ_PI0_TASK_PRIORITY", "4"),
            "QZ_PI0_TASK_PRIORITY",
        ),
        "workspace_id": os.environ.get(
            "QZ_PI0_WORKSPACE_ID",
            "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        ),
        "node_num_per_replica": _int_value(
            os.environ.get("QZ_PI0_NODE_NUM_PER_REPLICA", "1"),
            "QZ_PI0_NODE_NUM_PER_REPLICA",
        ),
    }
    if os.environ.get("QZ_PI0_PAYLOAD_OVERRIDES"):
        overrides = json.loads(os.environ["QZ_PI0_PAYLOAD_OVERRIDES"])
        if not isinstance(overrides, dict):
            raise RuntimeError("QZ_PI0_PAYLOAD_OVERRIDES must be a JSON object")
        payload.update(overrides)

    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/create",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    data = _check_response(resp)
    if data:
        inference_id = data.get("inference_serving_id", "")
        print(f"Inference ID: {inference_id}")
        print(f"Pi0 endpoint: {get_endpoint(SEQ, domain_prefix=domain_prefix)}")
        INFERENCE_ID = inference_id
        API_KEY = args.api_key or API_KEY
        _save_state(STATE_FILE, SEQ, inference_id, API_KEY)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QZ OpenPI/pi0 inference serving client")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create new pi0 inference")
    group.add_argument("--stop", action="store_true", help="Stop last pi0 inference")
    group.add_argument("--query", action="store_true", help="Query last pi0 inference")
    group.add_argument("--health", action="store_true", help="Check pi0 websocket health")
    group.add_argument("--endpoint", action="store_true", help="Print current pi0 websocket endpoint")

    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--api-key", default=os.environ.get("QZ_API_KEY", ""))
    parser.add_argument("--image", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-version", type=int, default=0)
    parser.add_argument("--policy-config", default="")
    parser.add_argument("--openpi-root", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--spec-id", default="")
    parser.add_argument("--domain-prefix", default="")
    parser.add_argument("--default-prompt", default="")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--health-uri", default="")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    STATE_FILE = Path(args.state_file).expanduser()
    if not STATE_FILE.is_absolute():
        STATE_FILE = PROJECT_ROOT / STATE_FILE
    SEQ, INFERENCE_ID, API_KEY = _load_state(STATE_FILE)
    if args.api_key:
        API_KEY = args.api_key

    domain_prefix = args.domain_prefix or os.environ.get("QZ_PI0_DOMAIN_PREFIX", "pi0")
    if args.endpoint:
        print(get_endpoint(_require_seq(SEQ), domain_prefix=domain_prefix))
        raise SystemExit(0)

    if args.health:
        uri = args.health_uri or get_endpoint(_require_seq(SEQ), domain_prefix=domain_prefix)
        health_check(uri, api_key=API_KEY, timeout_sec=args.timeout_sec)
        raise SystemExit(0)

    username = os.environ.get("QZ_USERNAME")
    password = os.environ.get("QZ_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            f"QZ_USERNAME and QZ_PASSWORD must be set (e.g. via {ENV_FILE})"
        )
    token = query_token(username, password)

    if args.create:
        create(token, args)
    elif args.stop:
        stop(INFERENCE_ID, token)
    elif args.query:
        query(INFERENCE_ID, token)
