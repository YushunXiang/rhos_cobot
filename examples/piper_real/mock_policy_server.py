"""Mock websocket policy server for examples.piper_real.main.

Protocol compatibility targets openpi_client.websocket_client_policy:
1) Send one metadata frame immediately after connection.
2) For each binary request, reply with msgpack-encoded action dict.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 9000
    action_dim: int = 14
    chunk_size: int = 64
    noise_scale: float = 0.0


def _parse_args() -> ServerArgs:
    parser = argparse.ArgumentParser(description="Mock policy websocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    ns = parser.parse_args()
    return ServerArgs(
        host=ns.host,
        port=ns.port,
        action_dim=ns.action_dim,
        chunk_size=ns.chunk_size,
        noise_scale=ns.noise_scale,
    )


def _metadata(args: ServerArgs) -> dict:
    return {
        "name": "mock_policy_server",
        "backend": "mock",
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "reset_pose": [0.0] * args.action_dim,
    }


def _base_action_from_obs(obs: Mapping, action_dim: int) -> np.ndarray:
    state = obs.get("state")
    if isinstance(state, np.ndarray):
        flat = state.astype(np.float32, copy=False).reshape(-1)
        out = np.zeros((action_dim,), dtype=np.float32)
        copy_len = min(action_dim, flat.shape[0])
        out[:copy_len] = flat[:copy_len]
        return out
    return np.zeros((action_dim,), dtype=np.float32)


def _make_action_chunk(base: np.ndarray, chunk_size: int, noise_scale: float) -> np.ndarray:
    actions = np.repeat(base[np.newaxis, :], repeats=chunk_size, axis=0)
    if noise_scale > 0:
        actions = actions + np.random.normal(0.0, noise_scale, size=actions.shape).astype(np.float32)
    return actions.astype(np.float32, copy=False)


def _build_response(obs: Mapping, args: ServerArgs) -> dict:
    base = _base_action_from_obs(obs, args.action_dim)
    return {
        "actions": _make_action_chunk(base, args.chunk_size, args.noise_scale),
        "STOP": False,
        "server_timing": {"mock_infer_ms": 0.1},
    }


def _handle_client(conn: ServerConnection, args: ServerArgs) -> None:
    peer = conn.remote_address
    logging.info("Client connected from %s", peer)
    packer = msgpack_numpy.Packer()
    conn.send(packer.pack(_metadata(args)))

    while True:
        try:
            raw = conn.recv()
        except ConnectionClosed:
            logging.info("Client disconnected: %s", peer)
            return

        if isinstance(raw, str):
            conn.send("Mock server expects binary msgpack payloads.")
            continue

        try:
            obs = msgpack_numpy.unpackb(raw)
            if not isinstance(obs, Mapping):
                raise TypeError(f"Expected mapping observation, got {type(obs)!r}")
            response = _build_response(obs, args)
            conn.send(packer.pack(response))
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to process request")
            conn.send(f"Mock server error: {exc}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting mock policy server on ws://%s:%d", args.host, args.port)
    logging.info(
        "Mock behavior: action_dim=%d chunk_size=%d noise_scale=%s",
        args.action_dim,
        args.chunk_size,
        args.noise_scale,
    )

    with serve(
        lambda conn: _handle_client(conn, args),
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        logging.info("Mock policy server is ready")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Shutting down mock policy server")


if __name__ == "__main__":
    main()
