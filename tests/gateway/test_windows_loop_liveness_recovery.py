"""Windows process-level recovery test for the loop liveness watchdog."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli
from gateway.shutdown_watchdog import get_loop_heartbeat_path


_WORKER = r"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from gateway.shutdown_watchdog import (
    get_loop_heartbeat_path,
    loop_heartbeat_forever,
    start_loop_liveness_watchdog,
)

home = Path(sys.argv[1])
mode = sys.argv[2]

async def main():
    heartbeat_task = asyncio.create_task(
        loop_heartbeat_forever(interval_s=60.0, home=home)
    )
    heartbeat = get_loop_heartbeat_path(home)
    for _ in range(200):
        if heartbeat.exists():
            payload = json.loads(heartbeat.read_text(encoding="utf-8"))
            if (
                payload.get("pid") == os.getpid()
                and isinstance(payload.get("loop_tick_witness"), dict)
            ):
                break
        await asyncio.sleep(0.01)
    else:
        raise RuntimeError("loop-tick witness was not armed")

    (home / f"ready-{mode}.json").write_text(
        json.dumps({"pid": os.getpid(), "payload": payload}), encoding="utf-8"
    )
    if mode == "wedge":
        handle = start_loop_liveness_watchdog(
            asyncio.get_running_loop(),
            probe_interval=0.05,
            probe_timeout=0.05,
            max_strikes=1,
        )
        if handle is None:
            raise RuntimeError("watchdog did not start")
        await asyncio.sleep(0.12)
        time.sleep(2.0)  # Freeze the real event loop; watchdog must os._exit(75).
        raise RuntimeError("watchdog failed to terminate the wedged process")

    await asyncio.sleep(30.0)

asyncio.run(main())
"""


def _wait_json(path: Path, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


@pytest.mark.windows_only
def test_watchdog_exit_75_is_restarted_with_a_live_tcp_witness(tmp_path):
    """A service-manager-style restart recovers after a real loop freeze."""
    worker = tmp_path / "loop_watchdog_worker.py"
    worker.write_text(_WORKER, encoding="utf-8")
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    first = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), "wedge"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert first.returncode == 75, (first.stdout, first.stderr, first.returncode)
    first_ready = _wait_json(tmp_path / "ready-wedge.json")

    # This is the external-supervisor boundary: exit 75 automatically starts a
    # fresh process, matching systemd/launchd/Windows service-manager policy.
    recovered = subprocess.Popen(
        [sys.executable, str(worker), str(tmp_path), "recovered"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        second_ready = _wait_json(tmp_path / "ready-recovered.json")
        assert second_ready["pid"] != first_ready["pid"]
        assert second_ready["payload"]["pid"] == second_ready["pid"]
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                second_ready["pid"], home=tmp_path, tick_timeout=1.0
            )
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )
        assert get_loop_heartbeat_path(tmp_path).exists()
    finally:
        recovered.terminate()
        try:
            recovered.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            recovered.kill()
            recovered.wait(timeout=5.0)
