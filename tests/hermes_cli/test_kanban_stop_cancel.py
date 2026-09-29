"""Regression tests: ``/stop`` must cancel the Kanban work it dispatched —
safely, atomically, and without fanning out MORE work.

Context. QueenAlice (an orchestrator profile) turns one Discord thread into a
bounded set of Kanban tasks assigned to specialist profiles. Those tasks carry
a ``kanban_notify_subs`` row pointing back at the originating
platform/chat/thread, which is how their results get delivered. Before this,
``/stop`` interrupted only the gateway's own agent turn: the dispatched workers
kept running and kept reporting into a conversation the operator had already
stopped.

Every test here pins a behaviour that a naive implementation gets WRONG. All
three of the following were observed against a real DB during review:

* archiving a cancelled parent PROMOTED its children to ``ready``, so ``/stop``
  dispatched the very work it was meant to kill;
* a task subscribed to two conversations was destroyed by a ``/stop`` from
  either one, silently killing work the other was still waiting on;
* reclaim-then-archive in separate transactions let the dispatcher re-claim
  and spawn a worker in the gap, and the archive then erased its PID.
"""

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _sub(conn, task_id, *, thread_id="thr1", chat_id="chan1", platform="discord"):
    kbn.add_notify_sub(
        conn,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        user_id="userA",
        notifier_profile="orchestrator",
    )


def _cancel(conn, **kw):
    kw.setdefault("platform", "discord")
    kw.setdefault("chat_id", "chan1")
    kw.setdefault("thread_id", "thr1")
    return kb.cancel_tasks_for_notify_source(conn, **kw)


# ---------------------------------------------------------------------------
# Scoping: only this conversation's work dies
# ---------------------------------------------------------------------------


def test_cancel_scoped_to_thread(kanban_home):
    """Only nonterminal tasks subscribed to the calling thread are cancelled."""
    conn = kbc.connect()
    try:
        mine = kb.create_task(conn, title="in my thread", assignee="riskcontroller")
        other = kb.create_task(conn, title="other thread", assignee="riskcontroller")
        finished = kb.create_task(conn, title="already done", assignee="riskcontroller")
        _sub(conn, mine)
        _sub(conn, other, thread_id="thr2")
        _sub(conn, finished)
        kb.archive_task(conn, finished)

        result = _cancel(conn, reason="operator /stop")

        assert result.cancelled == [mine]
        assert kb.get_task(conn, mine).status == "cancelled"
        assert kb.get_task(conn, other).status != "cancelled"
        # An already-terminal task is neither re-cancelled nor reported.
        assert kb.get_task(conn, finished).status == "archived"
    finally:
        conn.close()


def test_cancel_matches_platform_case_insensitively(kanban_home):
    """Notifier routing is case-insensitive on platform; cancel must match it."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="riskcontroller")
        _sub(conn, tid, platform="discord")

        assert _cancel(conn, platform="DISCORD").cancelled == [tid]
    finally:
        conn.close()


def test_channel_level_stop_does_not_kill_thread_work(kanban_home):
    """``thread_id`` None/'' means the CHANNEL, not "every thread in it".

    ``kanban_notify_subs.thread_id`` is ``NOT NULL DEFAULT ''``, so a
    channel-level subscription stores ``''``. A ``/stop`` typed in the channel
    must not reach work belonging to a child thread, and vice versa.
    """
    conn = kbc.connect()
    try:
        in_channel = kb.create_task(conn, title="channel work", assignee="w")
        in_thread = kb.create_task(conn, title="thread work", assignee="w")
        _sub(conn, in_channel, thread_id="")
        _sub(conn, in_thread, thread_id="thr1")

        # None and "" are the same channel-level scope.
        assert _cancel(conn, thread_id=None).cancelled == [in_channel]
        assert kb.get_task(conn, in_thread).status != "cancelled"

        assert _cancel(conn, thread_id="thr1").cancelled == [in_thread]
    finally:
        conn.close()


def test_cancel_returns_empty_for_unrelated_source(kanban_home):
    """Nothing is cancelled when no subscription matches the conversation."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="elsewhere", assignee="riskcontroller")
        _sub(conn, tid, chat_id="other-chan")

        assert _cancel(conn).cancelled == []
        assert kb.get_task(conn, tid).status != "cancelled"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Shared work: detach, never destroy (reviewed BLOCKER)
# ---------------------------------------------------------------------------


def test_task_shared_with_another_conversation_is_detached_not_cancelled(kanban_home):
    """A task another conversation is ALSO waiting on must survive.

    ``_inherit_notify_subs`` fans a child's subscriptions across every parent,
    so multi-subscriber tasks are a normal product of this orchestration flow.
    Destroying one would silently kill work a different operator/thread is
    still waiting on, so the matching subscription is dropped instead.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="shared work", assignee="w")
        _sub(conn, tid, chat_id="chanA", thread_id="thr1")
        _sub(conn, tid, chat_id="chanB", thread_id="thr9")

        result = _cancel(conn, chat_id="chanA", thread_id="thr1")

        assert result.cancelled == []
        assert result.detached == [tid]
        assert kb.get_task(conn, tid).status != "cancelled"
        # Only this conversation stopped listening; the other still gets results.
        remaining = kbn.list_notify_subs(conn, tid)
        assert [(s["chat_id"], s["thread_id"]) for s in remaining] == [("chanB", "thr9")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dependency closure: cancelling must not FAN OUT work (reviewed BLOCKER)
# ---------------------------------------------------------------------------


def test_cancelled_parent_does_not_promote_its_child(kanban_home):
    """The child of a cancelled parent must never become dispatchable.

    ``recompute_ready`` treats ``done``/``archived`` parents as satisfying a
    dependency. Reusing ``archive_task`` as the cancel primitive therefore
    flipped children ``todo`` -> ``ready``, i.e. ``/stop`` dispatched new work.
    ``cancelled`` must NOT satisfy a dependency.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        # Subscribe only the parent, and only AFTER the child exists, so the
        # child does not inherit the subscription and is invisible to the
        # source match — the exact shape that leaked a runaway worker.
        _sub(conn, parent)
        assert kbn.list_notify_subs(conn, child) == []

        result = _cancel(conn)

        assert kb.get_task(conn, child).status != "ready", (
            "cancelling a parent must not make its child dispatchable"
        )
        # The child is dead work now: it is reported and terminal.
        assert child in result.cancelled
        assert kb.get_task(conn, child).status == "cancelled"
        assert kb.get_task(conn, parent).status == "cancelled"
    finally:
        conn.close()


def test_cancel_walks_the_whole_descendant_chain(kanban_home):
    """Cancellation reaches grandchildren, not just direct children."""
    conn = kbc.connect()
    try:
        root = kb.create_task(conn, title="root", assignee="w")
        mid = kb.create_task(conn, title="mid", assignee="w", parents=[root])
        leaf = kb.create_task(conn, title="leaf", assignee="w", parents=[mid])
        _sub(conn, root)

        result = _cancel(conn)

        assert set(result.cancelled) == {root, mid, leaf}
        for tid in (root, mid, leaf):
            assert kb.get_task(conn, tid).status == "cancelled"
    finally:
        conn.close()


def test_descendant_shared_with_another_parent_survives(kanban_home):
    """A child that another LIVE parent still needs is not cancelled.

    Fan-in is legitimate: a join task can depend on two independent parents.
    Killing one lane must not destroy the join the other lane is still
    working toward.
    """
    conn = kbc.connect()
    try:
        mine = kb.create_task(conn, title="my parent", assignee="w")
        theirs = kb.create_task(conn, title="their parent", assignee="w")
        join = kb.create_task(
            conn, title="join", assignee="w", parents=[mine, theirs]
        )
        _sub(conn, mine)

        result = _cancel(conn)

        assert result.cancelled == [mine]
        assert kb.get_task(conn, theirs).status != "cancelled"
        # The join is not destroyed (another lane feeds it) but it can never
        # become ready off a cancelled parent: it must be parked VISIBLY for an
        # operator, not left waiting in ``todo`` forever.
        assert result.orphaned == [join]
        joined = kb.get_task(conn, join)
        assert joined.status == "blocked"
        # ...and it stays parked even after the surviving lane finishes.
        kb.archive_task(conn, theirs)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, join).status == "blocked"
    finally:
        conn.close()


def test_orphaned_fan_in_is_released_by_explicit_unblock(kanban_home):
    """The parked join is an operator decision, and unblocking it works once the
    operator removes the dead dependency."""
    conn = kbc.connect()
    try:
        mine = kb.create_task(conn, title="my parent", assignee="w")
        theirs = kb.create_task(conn, title="their parent", assignee="w")
        join = kb.create_task(conn, title="join", assignee="w", parents=[mine, theirs])
        _sub(conn, mine)
        _cancel(conn)
        kb.archive_task(conn, theirs)

        conn.execute("DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (mine, join))
        conn.commit()
        assert kb.unblock_task(conn, join)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, join).status == "ready"
    finally:
        conn.close()


def test_descendant_reporting_elsewhere_is_never_cancelled(kanban_home):
    """A child that reports to ANOTHER conversation is not ours to kill.

    Same failure class as destroying a shared root: the child was created inside
    this lane, but its results go to a different thread, and that thread is
    still waiting. Cancelling an ancestor here must not destroy it.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        grandchild = kb.create_task(
            conn, title="grandchild", assignee="w", parents=[child]
        )
        _sub(conn, parent, chat_id="chan1", thread_id="thr1")
        # The child (and thus its subtree) answers to a different conversation.
        _sub(conn, child, chat_id="chanB", thread_id="thr9")

        result = _cancel(conn, chat_id="chan1", thread_id="thr1")

        assert result.cancelled == [parent]
        assert child in result.preserved
        assert kb.get_task(conn, child).status != "cancelled"
        # And the walk must not reach past it into that lane's own work.
        assert grandchild not in result.cancelled
        assert kb.get_task(conn, grandchild).status != "cancelled"
        # Nor may the preserved child become dispatchable off a cancelled parent.
        assert kb.get_task(conn, child).status != "ready"
    finally:
        conn.close()


def test_descendant_shared_with_us_is_detached_not_cancelled(kanban_home):
    """A descendant subscribed to us AND elsewhere: drop only our sub."""
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        _sub(conn, parent, chat_id="chan1", thread_id="thr1")
        _sub(conn, child, chat_id="chan1", thread_id="thr1")
        _sub(conn, child, chat_id="chanB", thread_id="thr9")

        result = _cancel(conn, chat_id="chan1", thread_id="thr1")

        assert child in result.detached
        assert child not in result.cancelled
        assert kb.get_task(conn, child).status != "cancelled"
        remaining = [
            (s["chat_id"], s["thread_id"])
            for s in kbn.list_notify_subs(conn, child)
        ]
        assert remaining == [("chanB", "thr9")]
    finally:
        conn.close()


def test_already_done_descendant_is_left_alone(kanban_home):
    """Completed work is history; cancelling a parent must not rewrite it."""
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        _sub(conn, parent)
        kb.archive_task(conn, child)

        result = _cancel(conn)

        assert child not in result.cancelled
        assert kb.get_task(conn, child).status == "archived"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker termination and honest reporting
# ---------------------------------------------------------------------------


def test_cancel_terminates_running_worker_and_marks_terminal_first(kanban_home):
    """A running match is made terminal BEFORE its worker is signalled.

    Ordering matters: if the claim were released first (the old
    reclaim-then-archive shape), the dispatcher could re-claim the task and
    spawn a fresh worker in the gap, and the later archive would erase that
    worker's PID — an untrackable runaway. Marking terminal first means the
    dispatcher can never claim it again.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="running task", assignee="w")
        _sub(conn, tid)
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        own_pid = os.getpid()
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (own_pid, tid))
        conn.commit()

        status_when_signalled = {}

        def _fake_kill(pid, sig):
            # Observe the board as the worker-killing code sees it.
            probe = kbc.connect()
            try:
                status_when_signalled["status"] = kb.get_task(probe, tid).status
            finally:
                probe.close()
            raise ProcessLookupError  # report the worker as already gone

        result = kb.cancel_tasks_for_notify_source(
            conn, platform="discord", chat_id="chan1", thread_id="thr1",
            signal_fn=_fake_kill,
        )

        assert result.cancelled == [tid]
        assert status_when_signalled.get("status") == "cancelled", (
            "task must be terminal before the worker is signalled"
        )
        assert kb.get_task(conn, tid).status == "cancelled"
        assert kb.get_task(conn, tid).worker_pid is None
    finally:
        conn.close()


def test_non_host_local_worker_is_reported_as_unverified(kanban_home):
    """A worker on another host cannot be killed — say so, don't claim success.

    ``/stop`` is the operator's emergency brake; reporting "cancelled" while a
    remote process keeps running and reporting is the one lie that matters
    here. The task is still made terminal (so it cannot be re-dispatched), but
    the caller learns the worker's fate is unverified.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="remote task", assignee="w")
        _sub(conn, tid)
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, tid, claimer="some-other-host:pid:1") is not None
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (4242, tid))
        conn.commit()

        signalled = []
        result = kb.cancel_tasks_for_notify_source(
            conn, platform="discord", chat_id="chan1", thread_id="thr1",
            signal_fn=lambda pid, sig: signalled.append(pid),
        )

        assert result.cancelled == [tid]
        assert signalled == [], "must not signal a PID on a different host"
        assert tid in result.workers_unverified
        assert kb.get_task(conn, tid).status == "cancelled"
    finally:
        conn.close()


def test_idempotent_second_cancel_is_a_noop(kanban_home):
    """Double-tapping /stop reports nothing the second time."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        _sub(conn, tid)

        assert _cancel(conn).cancelled == [tid]
        again = _cancel(conn)
        assert again.cancelled == []
        assert again.detached == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ``cancelled`` is a real terminal status
# ---------------------------------------------------------------------------


def test_cancelled_task_cannot_be_claimed_or_promoted(kanban_home):
    """The dispatcher must not be able to claim or promote cancelled work."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        _sub(conn, tid)
        _cancel(conn)

        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is None
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "cancelled"
    finally:
        conn.close()


def test_cancel_emits_an_audit_event(kanban_home):
    """The board keeps a record of who killed the task and why."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        _sub(conn, tid)

        _cancel(conn, reason="operator /stop in #inaka-trader")

        kinds = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            ).fetchall()
        ]
        assert "cancelled" in kinds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Round-2 review BLOCKERs
# ---------------------------------------------------------------------------


def test_worker_spawned_after_concurrent_cancel_is_terminated(kanban_home, all_assignees_spawnable, monkeypatch):
    """BLOCKER 1 (claim/spawn race). The dispatcher claims, then spawns, then
    records the PID. A /stop landing between claim and PID-record used to see
    worker_pid=NULL, report a clean cancel, and let _set_worker_pid stamp a live
    worker onto a cancelled row. The dispatcher must refuse the stamp and kill
    its own child, and /stop must not report the stop as verified."""
    from hermes_cli import kanban_db_dispatch as kbd

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="racy", assignee="w")
        _sub(conn, tid)
        kb.recompute_ready(conn)
        observed = {}

        def spawn_then_stop(task, workspace, board=None):
            # The operator's /stop lands while the child is being created.
            other = kbc.connect()
            try:
                observed["result"] = kb.cancel_tasks_for_notify_source(
                    other, platform="discord", chat_id="chan1", thread_id="thr1",
                    signal_fn=lambda pid, sig: None)
            finally:
                other.close()
            return 424242

        terminated = []

        def spy_terminate(pid, lock, **kw):
            terminated.append(pid)
            return {"host_local": True, "terminated": True, "prev_pid": pid}

        monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", spy_terminate)
        kbd.dispatch_once(conn, spawn_fn=spawn_then_stop)

        task = kb.get_task(conn, tid)
        assert task.status == "cancelled"
        assert task.worker_pid is None, "a live worker's PID was stamped onto a cancelled row"
        assert terminated == [424242], "the dispatcher must kill the child it spawned for a dead claim"
        assert tid in observed["result"].workers_unverified, (
            "/stop saw a claim with no PID yet — it cannot prove the stop, so it must not claim one")
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
        assert "spawn_aborted_cancelled" in kinds
    finally:
        conn.close()


def test_normal_spawn_still_records_pid(kanban_home, all_assignees_spawnable):
    """Sibling path through the fenced _set_worker_pid: no cancel, PID recorded."""
    from hermes_cli import kanban_db_dispatch as kbd

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="normal", assignee="w")
        kb.recompute_ready(conn)
        kbd.dispatch_once(conn, spawn_fn=lambda task, ws, board=None: 515151)
        task = kb.get_task(conn, tid)
        assert (task.status, task.worker_pid) == ("running", 515151)
    finally:
        conn.close()


def _scratch_dir(tmp_path, name):
    d = tmp_path / ".hermes" / "kanban" / "workspaces" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "work.txt").write_text("in progress", encoding="utf-8")
    return d


def test_cancelling_child_keeps_running_parent_workspace(kanban_home, tmp_path):
    """BLOCKER 2. Cancelling a child ran the parent-workspace sweep, which only
    checked the CHILDREN's statuses — so a still-running, out-of-scope parent had
    its scratch dir deleted under its worker."""
    from hermes_cli import kanban_db_workspace as kbw

    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, parent, claimer=kb._claimer_id()) is not None
        pdir = _scratch_dir(tmp_path, parent)
        assert kbw._is_managed_scratch_path(pdir), pdir
        kbw.set_workspace_path(conn, parent, str(pdir))
        _sub(conn, child)  # only the child answers to this conversation

        result = _cancel(conn)

        assert result.cancelled == [child]
        assert kb.get_task(conn, parent).status == "running"
        assert pdir.is_dir() and (pdir / "work.txt").exists(), "running parent's workspace deleted"
    finally:
        conn.close()


def test_finished_parent_workspace_is_still_reaped_after_last_child(kanban_home, tmp_path):
    """Sibling path through the parent sweep: a DONE parent's deferred cleanup
    still runs once its last child is terminal."""
    from hermes_cli import kanban_db_workspace as kbw

    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="w")
        child = kb.create_task(conn, title="child", assignee="w", parents=[parent])
        pdir = _scratch_dir(tmp_path, parent)
        kbw.set_workspace_path(conn, parent, str(pdir))
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        conn.commit()
        _sub(conn, child)

        _cancel(conn)

        assert not pdir.exists(), "a finished parent's deferred cleanup must still run"
    finally:
        conn.close()


def test_archiving_cancelled_parent_does_not_reactivate_child(kanban_home):
    """BLOCKER 3. ``archived`` satisfies a dependency and ``cancelled`` does not,
    so archiving a cancelled parent (the dashboard's normal hygiene button)
    used to promote the children /stop had just killed or parked."""
    conn = kbc.connect()
    try:
        mine = kb.create_task(conn, title="mine", assignee="w")
        theirs = kb.create_task(conn, title="theirs", assignee="w")
        join = kb.create_task(conn, title="join", assignee="w", parents=[mine, theirs])
        _sub(conn, mine)
        _cancel(conn)
        kb.archive_task(conn, theirs)

        assert kb.archive_task(conn, mine) is False, "cancelled work must not be archivable"
        kb.recompute_ready(conn)

        assert kb.get_task(conn, mine).status == "cancelled"
        assert kb.get_task(conn, join).status != "ready"
    finally:
        conn.close()


def test_cancelled_task_can_be_hard_deleted(kanban_home):
    """Since cancelled work cannot be archived, it must be purgeable directly."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        _sub(conn, tid)
        _cancel(conn)
        assert kb.delete_archived_task(conn, tid) is True
        assert kb.get_task(conn, tid) is None
        live = kb.create_task(conn, title="live", assignee="w")
        assert kb.delete_archived_task(conn, live) is False, "live work still needs two steps"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ``cancelled`` adopted across the board, not just in the cancel path
# ---------------------------------------------------------------------------


def test_cancelled_idempotency_key_can_be_reused(kanban_home):
    """Re-dispatching the same work after /stop must create NEW work, not hand
    back the cancelled task (which would silently never run)."""
    conn = kbc.connect()
    try:
        first = kb.create_task(conn, title="t", assignee="w", idempotency_key="job-1")
        _sub(conn, first)
        _cancel(conn)
        second = kb.create_task(conn, title="t", assignee="w", idempotency_key="job-1")
        assert second != first
        assert kb.create_task(conn, title="t", assignee="w", idempotency_key="job-1") == second
    finally:
        conn.close()


def test_cancelled_hidden_from_default_list_but_queryable(kanban_home):
    conn = kbc.connect()
    try:
        dead = kb.create_task(conn, title="dead", assignee="w")
        live = kb.create_task(conn, title="live", assignee="w")
        _sub(conn, dead)
        _cancel(conn)
        ids = {t.id for t in kb.list_tasks(conn)}
        assert live in ids and dead not in ids
        assert [t.id for t in kb.list_tasks(conn, status="cancelled")] == [dead]
        assert dead in {t.id for t in kb.list_tasks(conn, include_archived=True)}
    finally:
        conn.close()


def test_cancelled_does_not_count_as_live_board_load(kanban_home):
    """board_stats drives backlog/load views: dead work must not inflate them."""
    conn = kbc.connect()
    try:
        dead = kb.create_task(conn, title="dead", assignee="w")
        kb.create_task(conn, title="live", assignee="w")
        _sub(conn, dead)
        _cancel(conn)
        stats = kb.board_stats(conn)
        assert "cancelled" not in stats["by_status"]
        assert "cancelled" not in stats["by_assignee"].get("w", {})
        assert sum(stats["by_assignee"]["w"].values()) == 1
    finally:
        conn.close()


def test_gc_prunes_old_cancelled_task_events(kanban_home):
    conn = kbc.connect()
    try:
        dead = kb.create_task(conn, title="dead", assignee="w")
        live = kb.create_task(conn, title="live", assignee="w")
        _sub(conn, dead)
        _cancel(conn)
        conn.execute("UPDATE task_events SET created_at = 1")
        conn.commit()
        kb.gc_events(conn, older_than_seconds=60)
        remaining = {r["task_id"] for r in conn.execute("SELECT DISTINCT task_id FROM task_events")}
        assert dead not in remaining
        assert live in remaining, "live tasks keep their history"
    finally:
        conn.close()


def test_cancel_removes_this_conversations_subscription(kanban_home):
    """Nothing will ever be delivered on the stopping conversation's row, so it
    must not be polled forever."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        _sub(conn, tid)
        _cancel(conn)
        assert kbn.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_cycle_in_task_links_terminates(kanban_home):
    """link_tasks refuses cycles, but a DB edit can create one; the walk must
    still terminate and cancel both."""
    conn = kbc.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="w")
        b = kb.create_task(conn, title="b", assignee="w", parents=[a])
        conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (b, a))
        conn.commit()
        _sub(conn, a)
        result = _cancel(conn)
        assert set(result.cancelled) == {a, b}
    finally:
        conn.close()


def test_cancel_order_is_leaves_first_even_when_roots_nest(kanban_home):
    """A->B->C with A and B BOTH subscribed: every child must be cancelled
    before each of its parents (round-2 found A cancelled before B)."""
    conn = kbc.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="w")
        b = kb.create_task(conn, title="b", assignee="w", parents=[a])
        c = kb.create_task(conn, title="c", assignee="w", parents=[b])
        _sub(conn, a)
        _sub(conn, b)
        order = _cancel(conn).cancelled
        assert order.index(c) < order.index(b) < order.index(a)
    finally:
        conn.close()
