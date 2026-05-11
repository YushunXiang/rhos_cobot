"""Helpers for OpenPI websocket policies used by robot runtimes."""

from __future__ import annotations

import logging
from typing import Any

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _websocket_client_policy


class RemotePolicyServerResetAdapter:
    """Adds a best-effort server-side reset to OpenPI websocket policies."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def get_server_metadata(self) -> dict:
        return self.client.get_server_metadata()

    def infer(self, observation: dict) -> dict:
        return self.client.infer(observation)

    def reset(self) -> None:
        client_reset = getattr(self.client, "reset", None)
        if callable(client_reset):
            try:
                client_reset()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to reset remote policy client: %s", exc)

        ws = getattr(self.client, "_ws", None)
        if ws is None or not hasattr(ws, "send"):
            return

        try:
            packer = getattr(self.client, "_packer", None)
            if packer is not None and hasattr(packer, "pack"):
                payload = packer.pack({"type": "reset"})
            else:
                payload = msgpack_numpy.packb({"type": "reset"})

            ws.send(payload)
            recv = getattr(ws, "recv", None)
            if not callable(recv):
                return

            response = recv()
            if isinstance(response, str):
                logging.warning(
                    "Failed to send reset to remote policy server: %s",
                    response,
                )
            elif response is not None:
                msgpack_numpy.unpackb(response)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to send reset to remote policy server: %s", exc)


def create_resettable_websocket_policy(
    *,
    host: str,
    port: int | None,
    api_key: str | None = None,
    client_cls: type | None = None,
) -> RemotePolicyServerResetAdapter:
    cls = client_cls or _websocket_client_policy.WebsocketClientPolicy
    kwargs: dict[str, Any] = {"host": host, "port": port}
    if api_key is not None:
        kwargs["api_key"] = api_key
    return RemotePolicyServerResetAdapter(cls(**kwargs))
