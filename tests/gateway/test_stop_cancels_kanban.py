"""Regression tests: ``/stop`` cancels the Kanban work routed to the caller's
conversation — driven END TO END, with no mock standing in for the code itself.

These tests exist because an earlier version of this feature shipped dead: the
config gate was read with the wrong ``cfg_get`` signature and always evaluated
false, and the tests missed it because they monkeypatched ``cfg_get`` with a
fake whose signature matched the BUG instead of the real helper. So here:

* the config gate is a REAL ``config.yaml`` under a temp ``HERMES_HOME``;
* the board is a REAL Kanban DB with real tasks and real subscriptions;
* ``_cancel_kanban_for_source`` is never mocked;
* assertions are on DB state and on which locale keys rendered, not on
  whether some digit happens to appear in a string.
"""

import json
from pathlib import Path

import pytest

from gateway.platforms.base import MessageEvent, MessageType, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


# ---------------------------------------------------------------------------
# Real HERMES_HOME with a real config.yaml + a real kanban DB
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME. ``cancel_on_stop`` is left UNSET (feature off)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("kanban:\n  board: default\n", encoding="utf-8")
    kb.init_db()
    return home


def _enable_cancel_on_stop(home):
    """Write the real opt-in into the real config.yaml. load_config's cache is
    keyed on (path, mtime_ns, size), so a rewrite is picked up without busting it."""
    (home / "config.yaml").write_text(
        "kanban:\n  board: default\n  cancel_on_stop: true\n", encoding="utf-8"
    )


def _seed_task(*, chat_id="chan1", thread_id="thr1", title="dispatched work"):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=title, assignee="riskcontroller")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="discord", chat_id=chat_id,
            thread_id=thread_id, user_id="userA", notifier_profile="orchestrator",
        )
        return tid
    finally:
        conn.close()


def _status(task_id):
    conn = kbc.connect()
    try:
        return kb.get_task(conn, task_id).status
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Minimal real-ish runner: only session plumbing is faked, never the feature
# ---------------------------------------------------------------------------


class _FakeAgent:
    pass


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


def _source(uid="userA", thread_id="thr1", chat_id="chan1"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_type="forum",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=uid,
    )


def _runner(*, running=True, source=None):
    source = source or _source()
    runner = object.__new__(GatewayRunner)
    key = build_session_key(source)
    runner._running_agents = {key: _FakeAgent()} if running else {}
    runner.session_store = _FakeStore(key)
    runner._is_user_authorized = lambda src: True
    runner.adapters = {}

    async def _fake_interrupt(session_key, src, *, interrupt_reason, invalidation_reason):
        return None

    runner._interrupt_and_clear_session = _fake_interrupt
    return runner, source


async def _stop(runner, source):
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    result = await runner._handle_stop_command(event)
    return str(getattr(result, "text", result))


# ---------------------------------------------------------------------------
# The config gate, exercised through the real config file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_by_default_leaves_kanban_work_alone(hermes_home):
    """Without the opt-in, /stop keeps its historical agent-turn-only meaning."""
    tid = _seed_task()
    runner, source = _runner()

    text = await _stop(runner, source)

    assert _status(tid) != "cancelled", "must not touch the board when disabled"
    assert "Kanban" not in text


@pytest.mark.asyncio
async def test_real_config_opt_in_actually_cancels(hermes_home):
    """The documented config key must really open the gate.

    This is the test the earlier broken version could not pass: no mock of
    ``cfg_get``, ``load_config``, or ``_cancel_kanban_for_source`` — just the
    real file on disk.
    """
    _enable_cancel_on_stop(hermes_home)
    tid = _seed_task()
    runner, source = _runner()

    text = await _stop(runner, source)

    assert _status(tid) == "cancelled"
    assert "Cancelled 1" in text


@pytest.mark.asyncio
async def test_cancel_is_scoped_to_the_calling_thread(hermes_home):
    """A /stop in one thread must not kill a sibling thread's work."""
    _enable_cancel_on_stop(hermes_home)
    mine = _seed_task(thread_id="thr1", title="mine")
    theirs = _seed_task(thread_id="thr2", title="theirs")
    runner, source = _runner(source=_source(thread_id="thr1"))

    await _stop(runner, source)

    assert _status(mine) == "cancelled"
    assert _status(theirs) != "cancelled"


# ---------------------------------------------------------------------------
# Honest reporting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_work_is_reported_as_detached_not_cancelled(hermes_home):
    """Work another conversation still needs is reported as left running."""
    _enable_cancel_on_stop(hermes_home)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="shared", assignee="riskcontroller")
        for chat, thr in (("chan1", "thr1"), ("chan9", "thr9")):
            kbn.add_notify_sub(
                conn, task_id=tid, platform="discord", chat_id=chat,
                thread_id=thr, user_id="userA", notifier_profile="orchestrator",
            )
    finally:
        conn.close()
    runner, source = _runner()

    text = await _stop(runner, source)

    assert _status(tid) != "cancelled"
    assert "shared with another conversation" in text
    assert "Cancelled" not in text


@pytest.mark.asyncio
async def test_no_active_agent_still_reports_both_facts(hermes_home):
    """Post-handoff /stop: idle turn AND dispatched work both reported."""
    _enable_cancel_on_stop(hermes_home)
    tid = _seed_task()
    runner, source = _runner(running=False)

    text = await _stop(runner, source)

    assert _status(tid) == "cancelled"
    assert "No active task to stop" in text, "must not hide the idle-turn fact"
    assert "Cancelled 1" in text


@pytest.mark.asyncio
async def test_nothing_to_cancel_keeps_the_plain_reply(hermes_home):
    """With the feature on but no matching work, the reply is unchanged."""
    _enable_cancel_on_stop(hermes_home)
    _seed_task(chat_id="somewhere-else")
    runner, source = _runner(running=False)

    text = await _stop(runner, source)

    assert text.strip() == "No active task to stop."


# ---------------------------------------------------------------------------
# /stop must survive a broken board, and must not block the event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broken_kanban_db_does_not_break_stop(hermes_home, monkeypatch):
    """A locked/corrupt board must not turn the emergency brake into an error."""
    _enable_cancel_on_stop(hermes_home)
    runner, source = _runner()

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(kbc, "connect", _boom)

    text = await _stop(runner, source)

    assert "Stopped" in text
    assert "Kanban" not in text


@pytest.mark.asyncio
async def test_slow_cancel_is_bounded_and_reported_honestly(hermes_home, monkeypatch):
    """/stop must answer within its budget even when workers are slow to die,
    and must say the cancel is still running rather than claim a result."""
    import asyncio
    import threading

    import gateway.slash_commands as sc

    _enable_cancel_on_stop(hermes_home)
    runner, source = _runner()
    release = threading.Event()

    def _slow(*, source):
        release.wait(5)
        return kb.CancelSourceResult(cancelled=["t_late"])

    runner._cancel_kanban_for_source_blocking = _slow
    monkeypatch.setattr(sc, "_KANBAN_CANCEL_TIMEOUT_S", 0.2)
    try:
        text = await asyncio.wait_for(_stop(runner, source), timeout=3)
    finally:
        release.set()

    assert "Stopped" in text
    assert "still running" in text
    assert "Cancelled" not in text, "must not report a result it has not seen"


@pytest.mark.asyncio
async def test_busy_path_stop_also_cancels(hermes_home):
    """/stop arriving while the agent is mid-turn goes through the busy guard,
    not _handle_stop_command — it must cancel the dispatched work too."""
    _enable_cancel_on_stop(hermes_home)
    tid = _seed_task()
    runner, source = _runner()
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)

    result = await runner._busy_stop_command(event, build_session_key(source), source)

    assert _status(tid) == "cancelled"
    assert "Cancelled 1" in str(getattr(result, "text", result))


@pytest.mark.asyncio
async def test_stop_reaches_work_on_a_non_active_board(hermes_home):
    """The notifier delivers from every board; /stop must reach every board too,
    or an orchestrator that dispatched onto a project board cannot stop it."""
    _enable_cancel_on_stop(hermes_home)
    kb.create_board("project-x")
    conn = kbc.connect(board="project-x")
    try:
        tid = kb.create_task(conn, title="elsewhere", assignee="riskcontroller")
        kbn.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="chan1",
                           thread_id="thr1", user_id="userA", notifier_profile="orchestrator")
    finally:
        conn.close()
    runner, source = _runner()

    text = await _stop(runner, source)

    conn = kbc.connect(board="project-x")
    try:
        assert kb.get_task(conn, tid).status == "cancelled"
    finally:
        conn.close()
    assert "Cancelled 1" in text


@pytest.mark.asyncio
async def test_cancel_runs_off_the_event_loop(hermes_home):
    """The blocking cancel must be offloaded, not run inline.

    ``/stop`` is the command an operator reaches for when work has already gone
    wrong; blocking the loop here would stall every other conversation on every
    platform. Asserted by proving the blocking helper executes on a DIFFERENT
    thread than the running event loop.
    """
    import threading

    _enable_cancel_on_stop(hermes_home)
    _seed_task()
    runner, source = _runner()
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    original = runner._cancel_kanban_for_source_blocking

    def _record(*, source):
        seen["thread"] = threading.get_ident()
        return original(source=source)

    runner._cancel_kanban_for_source_blocking = _record

    await _stop(runner, source)

    assert seen.get("thread") is not None, "blocking helper never ran"
    assert seen["thread"] != loop_thread, (
        "cancel ran on the event loop thread — it must be offloaded"
    )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_is_auditable_on_the_board(hermes_home):
    """An operator can see later that /stop killed the task, and from where."""
    _enable_cancel_on_stop(hermes_home)
    tid = _seed_task()
    runner, source = _runner()

    await _stop(runner, source)

    conn = kbc.connect()
    try:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events "
            " WHERE task_id = ? AND kind = 'cancelled'",
            (tid,),
        ).fetchall()
    finally:
        conn.close()

    assert rows, "no cancelled event recorded"
    payload = json.loads(rows[0]["payload"]) if rows[0]["payload"] else {}
    assert "stop" in str(payload.get("reason", "")).lower()
