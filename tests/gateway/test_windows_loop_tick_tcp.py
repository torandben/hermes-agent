"""Native Windows regression coverage for the loop-tick witness."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli.gateway import _probe_loop_tick_tcp
from gateway.shutdown_watchdog import (
    get_loop_heartbeat_path,
    loop_heartbeat_forever,
)


def _serve_fixed_tcp_reply(reply: bytes) -> tuple[dict, list[bytes], threading.Thread]:
    """Start a fake loopback witness that ignores its request."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received: list[bytes] = []

    def serve() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                chunks = [conn.recv(128)]
                conn.settimeout(0.05)
                try:
                    while chunk := conn.recv(128):
                        chunks.append(chunk)
                except TimeoutError:
                    pass
                received.append(b"".join(chunks))
                conn.sendall(reply)
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    witness = {
        "transport": "tcp",
        "host": "127.0.0.1",
        "port": listener.getsockname()[1],
        "token": "known-secret-token-that-must-not-be-sent",
    }
    return witness, received, thread


@pytest.mark.windows_only
@pytest.mark.parametrize("reply", [b"1", b"x" * 32], ids=["legacy-one", "arbitrary"])
def test_tcp_probe_rejects_fixed_reply_without_disclosing_token(reply):
    witness, received, thread = _serve_fixed_tcp_reply(reply)

    try:
        assert _probe_loop_tick_tcp(witness, timeout=1.0) is False
    finally:
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert received
    assert received[0] != witness["token"].encode("ascii")


@pytest.mark.windows_only
def test_tcp_probe_send_uses_only_remaining_deadline(monkeypatch):
    """A slow connect must not grant sendall a second full timeout budget."""
    clock = [0.0]

    class SlowSendSocket:
        def __init__(self, timeout):
            self.timeout = timeout

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendall(self, _data):
            clock[0] += self.timeout
            raise TimeoutError

        def close(self):
            pass

    def slow_connect(_address, timeout):
        clock[0] += 0.8
        return SlowSendSocket(timeout)

    monkeypatch.setattr(gateway_cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gateway_cli.socket, "create_connection", slow_connect)
    witness = {
        "transport": "tcp",
        "host": "127.0.0.1",
        "port": 12345,
        "token": "t" * 43,
    }

    assert _probe_loop_tick_tcp(witness, timeout=1.0) is False
    assert clock[0] <= 1.0


@pytest.mark.windows_only
def test_windows_heartbeat_arms_authenticated_tcp_witness(tmp_path, caplog):
    """Windows must publish a live loop-owned witness without AF_UNIX noise."""

    async def scenario() -> dict:
        task = asyncio.create_task(
            loop_heartbeat_forever(interval_s=60.0, home=tmp_path)
        )
        try:
            heartbeat = get_loop_heartbeat_path(tmp_path)
            for _ in range(100):
                if heartbeat.exists():
                    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
                    witness = payload.get("loop_tick_witness")
                    if isinstance(witness, dict):
                        break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("heartbeat never advertised a loop-tick witness")

            assert witness["transport"] == "tcp"
            assert witness["host"] == "127.0.0.1"
            assert 0 < int(witness["port"]) <= 65535
            assert len(witness["token"]) >= 32

            verdict = await asyncio.to_thread(
                gateway_cli.probe_gateway_loop_liveness,
                payload["pid"],
                home=tmp_path,
                tick_timeout=1.0,
            )
            assert verdict == gateway_cli.GATEWAY_LOOP_ALIVE

            wrong_token = dict(witness, token="z" * len(witness["token"]))
            assert (
                await asyncio.to_thread(_probe_loop_tick_tcp, wrong_token, timeout=1.0)
                is False
            )
            assert (
                _probe_loop_tick_tcp(dict(witness, host="0.0.0.0"), timeout=1.0) is None
            )
            return payload
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    with caplog.at_level(logging.WARNING, logger="gateway.shutdown_watchdog"):
        payload = asyncio.run(scenario())

    assert payload["loop_tick_socket"] is True
    assert not caplog.records
