import json

import pytest


class _FakeHttpResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args, **kwargs):
        return self._payload


class _FakeWsConnection:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []
        self.closed = False

    def recv(self):
        return self._responses.pop(0)

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


def test_planner_models_url():
    from examples.piper_real.server_checks import planner_models_url

    assert planner_models_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/models"
    assert planner_models_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1/models"


def test_check_planner_server_success(monkeypatch):
    from examples.piper_real import server_checks

    payload = {"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]}
    monkeypatch.setattr(
        server_checks.urllib.request,
        "urlopen",
        lambda url, timeout=0: _FakeHttpResponse(payload),
    )

    result = server_checks.check_planner_server(
        "http://127.0.0.1:8000/v1",
        expected_model="Qwen/Qwen3-VL-8B-Instruct",
    )

    assert result.models == ("Qwen/Qwen3-VL-8B-Instruct",)


def test_check_planner_server_sends_bearer_auth(monkeypatch):
    from examples.piper_real import server_checks

    payload = {"data": [{"id": "Qwen/Qwen3.5-9B"}]}
    captured = {}

    def _urlopen(request, timeout=0):
        captured["auth"] = request.get_header("Authorization")
        return _FakeHttpResponse(payload)

    monkeypatch.setattr(server_checks.urllib.request, "urlopen", _urlopen)

    server_checks.check_planner_server(
        "https://qwen35-9b-1-inf.openapi-qb.sii.edu.cn/v1",
        expected_model="Qwen/Qwen3.5-9B",
        api_key="secret-token",
    )

    assert captured["auth"] == "Bearer secret-token"


def test_check_planner_server_missing_model(monkeypatch):
    from examples.piper_real import server_checks

    payload = {"data": [{"id": "other-model"}]}
    monkeypatch.setattr(
        server_checks.urllib.request,
        "urlopen",
        lambda url, timeout=0: _FakeHttpResponse(payload),
    )

    with pytest.raises(server_checks.ServerCheckError, match="expected model"):
        server_checks.check_planner_server(
            "http://127.0.0.1:8000/v1",
            expected_model="Qwen/Qwen3-VL-8B-Instruct",
        )


def test_check_pi0_server_sends_reset(monkeypatch):
    from examples.piper_real import server_checks
    from openpi_client import msgpack_numpy

    metadata = {"reset_pose": [1, 2, 3]}
    fake_conn = _FakeWsConnection(
        [
            msgpack_numpy.packb(metadata),
            msgpack_numpy.packb({"ok": True}),
        ]
    )
    monkeypatch.setattr(
        server_checks.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: fake_conn,
    )

    result = server_checks.check_pi0_server("127.0.0.1", 8001)

    assert result.metadata == metadata
    assert fake_conn.closed is True
    assert msgpack_numpy.unpackb(fake_conn.sent[0]) == {"type": "reset"}


def test_run_required_server_checks_calls_helpers(monkeypatch):
    from examples.piper_real import main as main_mod

    calls = []
    planner_kwargs = {}

    def _planner(*args, **kwargs):
        calls.append("planner")
        planner_kwargs.update(kwargs)

    def _pi0(*args, **kwargs):
        calls.append("pi0")

    from examples.piper_real import server_checks

    monkeypatch.setattr(server_checks, "check_planner_server", _planner)
    monkeypatch.setattr(server_checks, "check_pi0_server", _pi0)

    args = main_mod.Args()
    args.host = "127.0.0.1"
    args.port = 8001
    args.planner.base_url = "http://127.0.0.1:8000/v1"
    args.planner.model = "Qwen/Qwen3-VL-8B-Instruct"
    args.planner.api_key = "planner-secret"

    assert main_mod._run_required_server_checks(args, needs_pi0=True, needs_planner=True) is True
    assert calls == ["planner", "pi0"]
    assert planner_kwargs["api_key"] == "planner-secret"


def test_run_required_server_checks_returns_false_on_failure(monkeypatch):
    from examples.piper_real import main as main_mod
    from examples.piper_real import server_checks

    def _planner(*args, **kwargs):
        raise server_checks.ServerCheckError("planner down")

    monkeypatch.setattr(server_checks, "check_planner_server", _planner)

    args = main_mod.Args()
    assert main_mod._run_required_server_checks(args, needs_planner=True) is False
