'''
Usage:
python scripts/qz_server.py --create
python scripts/qz_server.py --health
python scripts/qz_server.py --stop

Export Function:
import request_qwen
resp = request_qwen(message)
'''
import argparse
import requests
import os
import json
import time
from pathlib import Path


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

INFERENCE_ID = ""
API_KEY = ""
# Resolve project root from script location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STATE_FILE = PROJECT_ROOT / "config" / "server_state.json"
ENV_FILE = PROJECT_ROOT / ".env"
SEQ = -1


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

# 走 QZ_PROXY_URL 指定的本机 HTTP 代理(默认 127.0.0.1:8888);trust_env=False
# 屏蔽 shell 中的 http_proxy/https_proxy/all_proxy(例如 127.0.0.1:7899),
# 避免被无关代理污染。
_PROXY_URL = os.environ.get("QZ_PROXY_URL", "http://127.0.0.1:8888")
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.proxies = {"http": _PROXY_URL, "https": _PROXY_URL}


def _load_state():
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"State file not found: {STATE_FILE}")

    try:
        data = json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, IOError) as exc:
        raise ValueError(f"Invalid state file: {STATE_FILE}") from exc

    required_keys = ("api_key",)
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"State file missing required keys: {missing}")

    seq = data.get("seq", 0)
    inference_id = data.get("inference_id", "")
    api_key = data["api_key"]

    if not isinstance(seq, int):
        raise ValueError("State key 'seq' must be int")
    if not isinstance(inference_id, str):
        raise ValueError("State key 'inference_id' must be str")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("State key 'api_key' must be non-empty str")

    return seq, inference_id, api_key


def _save_state(seq: int, inference_id: str):
    if not isinstance(seq, int):
        raise ValueError("'seq' must be int")
    if not isinstance(inference_id, str):
        raise ValueError("'inference_id' must be str")

    # Persist seq/inference_id while preserving api_key from current state.
    _, _, current_api_key = _load_state()
    STATE_FILE.write_text(
        json.dumps(
            {"seq": seq, "inference_id": inference_id, "api_key": current_api_key},
            indent=2,
        )
    )
    print("Saved seq/inference_id/api_key to state file.")


SEQ, INFERENCE_ID, API_KEY = _load_state()


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
    """验证API响应，返回data或None"""
    if resp.status_code != 200:
        print(f"HTTP错误: {resp.status_code}")
        return None
    json_data = _parse_json_response(resp, context="OpenAPI request")
    if json_data.get("code", -1) != 0:
        print(
            f"API错误: code={json_data.get('code')}, msg={json_data.get('msg', 'unknown')}"
        )
        return None
    return json_data.get("data", {})


def request(
    url: str,
    model: str,
    messages: list,
    api_key: str,
    **kwargs,
):
    """通用LLM API调用"""
    payload = {"model": model, "messages": messages, **kwargs}
    resp = _SESSION.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM request failed with HTTP {resp.status_code} from {url}: {resp.text[:300]!r}"
        )
    result = _parse_json_response(resp, context=f"LLM request to {url}")
    print(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return result


def request_qwen(messages: list, **kwargs):
    """Qwen3.5简化调用，只需要传messages。"""
    url = f"https://qwen35-9b-{SEQ}-inf.openapi-qb.sii.edu.cn/v1/chat/completions"
    return request(
        url=url,
        model="Qwen/Qwen3.5-9B",
        messages=messages,
        api_key=API_KEY,
        temperature=0.7,
        **kwargs,
    )

def get_endpoint():
    return f"https://qwen35-9b-{SEQ}-inf.openapi-qb.sii.edu.cn/v1/chat/completions"

def health_check(custom_domain: str, api_key: str):
    """检测create后的服务健康状态"""
    url = f"https://{custom_domain}/v1/chat/completions"
    messages = [
        {
            "role": "user",
            "content": "你好，请用一句话介绍你自己是什么模型，证明你已经准备就绪。",
        }
    ]
    resp = request(
        url,
        "Qwen/Qwen3.5-9B",
        messages,
        api_key,
        temperature=0.7,
        # max_tokens=100,
    )
    return resp

def query_token(username: str, password: str):
    resp = post(
        "https://qz.sii.edu.cn/auth/token",
        json={"password": password, "username": username},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed with HTTP {resp.status_code}: {resp.text[:300]!r}")
    print("Successfully Authenticated.")
    return _parse_json_response(resp, context="Token request").get("data", {}).get("access_token", "")


def query(inference_id: str, token: str):
    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/detail",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_serving_id": inference_id},
    )
    data = _check_response(resp)
    print(data)


def stop(inference_id: str, token: str):
    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/stop",
        headers={"Authorization": f"Bearer {token}"},
        json={"inference_serving_id": inference_id},
    )
    data = _check_response(resp)
    print(data)

def create(token: str, **kwargs):
    global SEQ
    global INFERENCE_ID
    SEQ = int(time.time())
    payload = {
        "command": '/inspire/ssd/project/robot-reasoning/xiangyushun-p-xiangyushun/miniforge3/condabin/conda run -n vllm-qwen35 --no-capture-output vllm serve ${MODEL_PATH} \\\n    --port ${PORT} \\\n    --served-model-name "Qwen/Qwen3.5-9B" \\\n    --max-model-len 262144 \\\n    --max-num-seqs 512 \\\n    --reasoning-parser qwen3',
        "custom_domain": f"qwen35-9b-{SEQ}-inf",
        "image": "inspire-studio/yushun-swift:1.8",
        "image_type": "SOURCE_PUBLIC",
        "logic_compute_group_id": "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
        "model_id": "b84bde56-de89-4ad8-b7ff-c855c1a0d055",
        "model_version": 1,
        "name": f"qwen3.5-9b-{SEQ}",
        "port": 8000,
        "project_id": "project-0ca5ec4c-7cf6-4700-9d2e-cd95c04e6e8e",
        "replicas": 1,
        "spec_id": "c18m200g1t",
        "task_priority": 4,
        "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        "node_num_per_replica": 1,
    }
    payload.update(kwargs)
    resp = post(
        "https://qz.sii.edu.cn/openapi/v1/inference_servings/create",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    data = _check_response(resp)
    if data:
        inference_id = data.get("inference_serving_id", "")
        print(f"Inference ID: {inference_id}")
        INFERENCE_ID = inference_id
        _save_state(SEQ, inference_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAPI inference serving client")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create", action="store_true", help="Create new inference")
    group.add_argument("--stop", action="store_true", help="Stop Last inference")
    group.add_argument("--health", action="store_true", help="Check service health")
    args = parser.parse_args()

    username = os.environ.get("QZ_USERNAME")
    password = os.environ.get("QZ_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            f"QZ_USERNAME and QZ_PASSWORD must be set (e.g. via {ENV_FILE})"
        )
    token = query_token(username, password)

    if args.create:
        create(token)
    elif args.stop:
        stop(INFERENCE_ID, token)
    elif args.health:
        health_check(f"qwen35-9b-{SEQ}-inf.openapi-qb.sii.edu.cn", API_KEY)
    else:
        parser.print_help()
