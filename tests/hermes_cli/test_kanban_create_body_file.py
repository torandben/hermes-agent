"""`kanban create --body-file` — shell-proof body delivery (#115432).

`--body` is a plain string: embedded newlines and trailing flag-like tokens
(e.g. a body line reading ``--json``) die in shell-to-Python marshalling
(MSYS git-bash / terminal shell=True). ``--body-file <path>|-`` reads the
body from a file (``-`` = stdin) so the bytes never cross the shell.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

BODY = "line one\nline two\n--json\n--body should survive verbatim\n"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create(argv, monkeypatch=None, stdin=None):
    """Drive the real ``hermes kanban create`` parser into ``_cmd_create``."""
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    args = root.parse_args(["kanban", "create", *argv])
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return kc._cmd_create(args)


def _latest_body():
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn)
    assert tasks, "expected _cmd_create to have stored a task"
    return tasks[-1]


def test_body_file_stores_body_verbatim_with_trailing_flags_intact(kanban_home, tmp_path, capsys):
    f = tmp_path / "body.md"
    f.write_text(BODY, encoding="utf-8")
    assert _create(["PROBE", "--body-file", str(f), "--assignee", "baxter", "--priority", "7"]) == 0
    capsys.readouterr()
    task = _latest_body()
    assert (task.body, task.assignee, task.priority) == (BODY, "baxter", 7)
    # --body and --body-file cannot both win; refuse instead of picking one.
    assert _create(["PROBE", "--body", "inline", "--body-file", str(f)]) == 2


def test_leaf_only_dispatcher_worker_cannot_bypass_via_cli(kanban_home, monkeypatch, capsys):
    """A worker with terminal access cannot evade the tool's leaf-only guard by
    running `hermes kanban create` directly."""
    from tools import kanban_tools as kt

    (kanban_home / "config.yaml").write_text(
        "kanban:\n  allow_worker_child_tasks: false\n", encoding="utf-8")
    kt._reset_worker_child_tasks_policy_cache()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

    assert _create(["must not exist"]) != 0
    assert "child task creation is disabled" in capsys.readouterr().err.lower()
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []


def test_dispatcher_worker_cli_create_allowed_by_default(kanban_home, monkeypatch):
    """The default permits workers that were not configured as leaf-only."""
    from tools import kanban_tools as kt
    kt._reset_worker_child_tasks_policy_cache()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert _create(["allowed child"]) == 0


def test_leaf_only_worker_cannot_use_swarm_cli(kanban_home, monkeypatch, capsys):
    """`swarm` also creates tasks and must honor the same leaf-only policy."""
    from tools import kanban_tools as kt
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  allow_worker_child_tasks: false\n", encoding="utf-8")
    kt._reset_worker_child_tasks_policy_cache()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    args = root.parse_args(["kanban", "swarm", "a goal", "--worker", "peer:a job",
                            "--verifier", "peer", "--synthesizer", "peer"])
    assert kc._cmd_swarm(args) != 0
    assert "child task creation is disabled" in capsys.readouterr().err.lower()
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []


def test_body_file_dash_reads_stdin(kanban_home, monkeypatch, capsys):
    assert _create(["PROBE", "--body-file", "-"], monkeypatch, stdin=BODY) == 0
    capsys.readouterr()
    assert _latest_body().body == BODY
