"""Health checks for the planner and Pi0 policy servers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from typing import Any
import urllib.error
import urllib.request

import websockets.sync.client

from openpi_client import msgpack_numpy


DEFAULT_TIMEOUT_SEC = 5.0


class ServerCheckError(RuntimeError):
    """Raised when a required inference service is unavailable or unhealthy."""


@dataclasses.dataclass(frozen=True)
class PlannerCheckResult:
    models: tuple[str, ...]
    expected_model: str
    models_url: str


@dataclasses.dataclass(frozen=True)
class Pi0CheckResult:
    uri: str
    metadata: Any


def planner_models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def check_planner_server(
    base_url: str,
    *,
    expected_model: str = "",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> PlannerCheckResult:
    models_url = planner_models_url(base_url)
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_sec) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise ServerCheckError(f"planner check failed for {models_url}: {exc}") from exc

    models = tuple(
        item.get("id", "")
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    if expected_model and expected_model not in models:
        raise ServerCheckError(
            f"planner check failed: expected model '{expected_model}' not found at {models_url}"
        )

    logging.info("Planner server healthy: %s", ", ".join(models) if models else "<empty>")
    return PlannerCheckResult(models=models, expected_model=expected_model, models_url=models_url)


def check_pi0_server(
    host: str,
    port: int,
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    api_key: str | None = None,
) -> Pi0CheckResult:
    uri = f"ws://{host}:{port}"
    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None

    try:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_sec,
            additional_headers=headers,
        )
    except Exception as exc:
        raise ServerCheckError(f"pi0 check failed for {uri}: {exc}") from exc

    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        conn.send(msgpack_numpy.packb({"type": "reset"}))
        response = conn.recv()
        if isinstance(response, str):
            raise ServerCheckError(f"pi0 check failed for {uri}: unexpected text response: {response}")
        msgpack_numpy.unpackb(response)
    except Exception as exc:
        raise ServerCheckError(f"pi0 check failed for {uri}: {exc}") from exc
    finally:
        conn.close()

    logging.info("Pi0 server healthy: %s", uri)
    return Pi0CheckResult(uri=uri, metadata=metadata)


@dataclasses.dataclass(frozen=True)
class CliArgs:
    planner_base_url: str
    planner_model: str = ""
    pi0_host: str = ""
    pi0_port: int = 0
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    pi0_api_key: str = ""


def run_cli(args: CliArgs) -> None:
    if args.planner_base_url:
        result = check_planner_server(
            args.planner_base_url,
            expected_model=args.planner_model,
            timeout_sec=args.timeout_sec,
        )
        print("planner models:", ", ".join(result.models) if result.models else "<empty>")

    if args.pi0_host and args.pi0_port:
        result = check_pi0_server(
            args.pi0_host,
            args.pi0_port,
            timeout_sec=args.timeout_sec,
            api_key=args.pi0_api_key or None,
        )
        print("pi0 metadata:", result.metadata)
        print("pi0 reset: ok")


def _parse_cli_args() -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-base-url", default="")
    parser.add_argument("--planner-model", default="")
    parser.add_argument("--pi0-host", default="")
    parser.add_argument("--pi0-port", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--pi0-api-key", default="")
    ns = parser.parse_args()
    return CliArgs(
        planner_base_url=ns.planner_base_url,
        planner_model=ns.planner_model,
        pi0_host=ns.pi0_host,
        pi0_port=ns.pi0_port,
        timeout_sec=ns.timeout_sec,
        pi0_api_key=ns.pi0_api_key,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run_cli(_parse_cli_args())


if __name__ == "__main__":
    main()
