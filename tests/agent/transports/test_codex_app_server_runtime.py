"""Tests for the optional codex app-server runtime gate.

These are unit tests for the api_mode rewriter and the wire-level transport
module. They do NOT require the `codex` CLI to be installed — that's
covered by a separate live test gated on `codex --version`.
"""

from __future__ import annotations

import pytest

from hermes_cli.runtime_provider import (
    _VALID_API_MODES,
    _maybe_apply_codex_app_server_runtime,
)


class TestApiModeRegistration:
    """The new api_mode must be registered or downstream parsing rejects it."""

    def test_codex_app_server_is_a_valid_api_mode(self) -> None:
        assert "codex_app_server" in _VALID_API_MODES

    def test_existing_api_modes_still_present(self) -> None:
        # Regression guard: don't accidentally delete other api_modes when
        # touching this set.
        for mode in (
            "chat_completions",
            "codex_responses",
            "anthropic_messages",
            "bedrock_converse",
        ):
            assert mode in _VALID_API_MODES


class TestMaybeApplyCodexAppServerRuntime:
    """The opt-in helper that rewrites api_mode → codex_app_server."""

    @pytest.mark.parametrize(
        "model_cfg",
        [
            None,
            {},
            {"openai_runtime": ""},
            {"openai_runtime": "auto"},
            {"openai_runtime": "AUTO"},
            {"other_key": "codex_app_server"},  # wrong key
        ],
    )
    def test_default_off_for_openai(self, model_cfg) -> None:
        """Default behavior is preserved when the flag is unset/auto."""
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai", api_mode="chat_completions", model_cfg=model_cfg
        )
        assert got == "chat_completions"

    def test_opt_in_rewrites_openai(self) -> None:
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai",
            api_mode="chat_completions",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "codex_app_server"



    @pytest.mark.parametrize(
        "provider",
        [
            "anthropic",
            "openrouter",
            "xai",
            "qwen-oauth",
            "opencode-zen",
            "bedrock",
            "",
        ],
    )
    def test_other_providers_never_rerouted(self, provider) -> None:
        """Non-OpenAI providers MUST NOT be rerouted even with the flag set —
        codex's app-server can only run OpenAI/Codex auth flows."""
        got = _maybe_apply_codex_app_server_runtime(
            provider=provider,
            api_mode="anthropic_messages",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "anthropic_messages", (
            f"provider={provider!r} should not be rerouted to codex_app_server"
        )


class TestCodexAppServerModule:
    """Module-surface tests for the JSON-RPC speaker. Don't require codex CLI."""




    def test_check_binary_handles_missing_executable(self) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        ok, msg = check_codex_binary(codex_bin="/nonexistent/codex/binary/path")
        assert ok is False
        assert "not found" in msg.lower() or "no such" in msg.lower()

    def test_codex_error_class_is_runtimeerror(self) -> None:
        from agent.transports.codex_app_server import CodexAppServerError

        err = CodexAppServerError(code=-32600, message="boom")
        assert isinstance(err, RuntimeError)
        assert "boom" in str(err)
        assert "-32600" in str(err)


class TestCodexExecutableResolution:
    @staticmethod
    def _capture_spawn(
        monkeypatch,
        *,
        codex_bin: str,
        resolved: str | None = None,
        which=None,
        env: dict[str, str] | None = None,
        captured: dict | None = None,
    ):
        import subprocess
        from agent.transports import codex_app_server as cas
        from hermes_cli import _subprocess_compat

        captured = {} if captured is None else captured

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["cmd"] = list(cmd)
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setattr(
            _subprocess_compat.shutil,
            "which",
            which or (lambda cmd, mode=0, path=None: resolved),
        )

        client = cas.CodexAppServerClient(codex_bin=codex_bin, env=env)
        client._closed = True
        return captured["cmd"]

    def test_spawn_resolves_codex_on_child_spawn_path(self, monkeypatch) -> None:
        import os

        child_path = os.pathsep.join(["child-bin", "child-npm"])
        resolved = os.path.join("child-bin", "codex.exe")
        seen_paths = []

        def fake_which(cmd, mode=0, path=None):
            if cmd != "codex":
                return None
            seen_paths.append(path)
            return resolved if path == child_path else None

        cmd = self._capture_spawn(
            monkeypatch,
            codex_bin="codex",
            which=fake_which,
            env={"PATH": child_path},
        )

        assert seen_paths == [child_path]
        assert cmd[:2] == [resolved, "app-server"]

    def test_spawn_falls_back_to_bare_codex_when_unresolved(self, monkeypatch) -> None:
        cmd = self._capture_spawn(
            monkeypatch,
            codex_bin="codex",
            resolved=None,
        )

        assert cmd[:2] == ["codex", "app-server"]

    @pytest.mark.parametrize(
        "which_suffix",
        [None, ".CMD"],
        ids=["which-finds-nothing", "which-rewrites-via-pathext"],
    )
    def test_spawn_preserves_explicit_executable_path(
        self, monkeypatch, which_suffix
    ) -> None:
        import os

        explicit = os.path.abspath(os.path.join("tools", "codex", "codex"))

        cmd = self._capture_spawn(
            monkeypatch,
            codex_bin=explicit,
            resolved=None if which_suffix is None else explicit + which_suffix,
        )

        assert cmd[:2] == [explicit, "app-server"]

    def test_spawn_resolves_npm_cmd_to_shell_free_node_launcher(
        self, monkeypatch, tmp_path
    ) -> None:
        import json
        from hermes_cli import _subprocess_compat

        npm_root = tmp_path / "npm"
        package_root = npm_root / "node_modules" / "@openai" / "codex"
        entrypoint = package_root / "bin" / "codex.js"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("// fixture", encoding="utf-8")
        (package_root / "package.json").write_text(
            json.dumps({"bin": {"codex": "bin/codex.js"}}), encoding="utf-8"
        )
        shim = npm_root / "codex.CMD"
        shim.write_text("@echo off", encoding="utf-8")
        node = tmp_path / "node.exe"
        node.write_bytes(b"")

        def fake_which(cmd, mode=0, path=None):
            return str(shim) if cmd == "codex" else str(node) if cmd == "node" else None

        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
        cmd = self._capture_spawn(
            monkeypatch,
            codex_bin="codex",
            which=fake_which,
            env={"PATH": str(tmp_path)},
        )

        assert cmd == [str(node), str(entrypoint.resolve()), "app-server"]
        assert all("cmd.exe" not in part.lower() for part in cmd)
        assert not cmd[0].lower().endswith((".cmd", ".bat"))

    def test_spawn_refuses_non_npm_windows_batch_shim(
        self, monkeypatch, tmp_path
    ) -> None:
        from hermes_cli import _subprocess_compat

        shim = tmp_path / "codex.cmd"
        shim.write_text("@echo off", encoding="utf-8")
        captured = {}
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

        with pytest.raises(OSError) as excinfo:
            self._capture_spawn(
                monkeypatch,
                codex_bin="codex",
                resolved=str(shim),
                captured=captured,
            )

        assert "cmd" not in captured
        assert "npm Codex package" in str(excinfo.value)

    def test_spawn_rejects_non_mapping_npm_manifest(
        self, monkeypatch, tmp_path
    ) -> None:
        from hermes_cli import _subprocess_compat

        npm_root = tmp_path / "npm"
        package_root = npm_root / "node_modules" / "@openai" / "codex"
        package_root.mkdir(parents=True)
        (package_root / "package.json").write_text("[]", encoding="utf-8")
        shim = npm_root / "codex.cmd"
        shim.write_text("@echo off", encoding="utf-8")
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

        with pytest.raises(OSError, match="npm Codex package"):
            self._capture_spawn(
                monkeypatch,
                codex_bin="codex",
                resolved=str(shim),
            )

    def test_preflight_uses_same_shell_free_node_launcher(
        self, monkeypatch, tmp_path
    ) -> None:
        import json
        import subprocess
        from agent.transports import codex_app_server as cas
        from hermes_cli import _subprocess_compat

        npm_root = tmp_path / "npm"
        package_root = npm_root / "node_modules" / "@openai" / "codex"
        entrypoint = package_root / "bin" / "codex.js"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("// fixture", encoding="utf-8")
        (package_root / "package.json").write_text(
            json.dumps({"bin": {"codex": "bin/codex.js"}}), encoding="utf-8"
        )
        shim = npm_root / "codex.cmd"
        shim.write_text("@echo off", encoding="utf-8")
        node = tmp_path / "node.exe"
        node.write_bytes(b"")
        captured = {}

        def fake_which(name, mode=0, path=None):
            return str(shim) if name == "codex" else str(node) if name == "node" else None

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = list(cmd)
            return subprocess.CompletedProcess(
                cmd, 0, stdout="codex-cli 0.130.0\n", stderr=""
            )

        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
        monkeypatch.setattr(_subprocess_compat.shutil, "which", fake_which)
        monkeypatch.setattr(subprocess, "run", fake_run)

        ok, version = cas.check_codex_binary()

        assert ok is True
        assert version == "0.130.0"
        assert captured["cmd"] == [
            str(node),
            str(entrypoint.resolve()),
            "--version",
        ]


class TestSpawnEnvIsolation:
    """The codex spawn must NOT rewrite HOME — codex's shell tool spawns
    subprocesses (gh, git, npm, aws, gcloud, ...) that need to find their
    config in the real user $HOME. CODEX_HOME isolates codex's own state,
    HOME stays unchanged.

    OpenClaw hit this footgun (openclaw/openclaw#81562) — they were
    rewriting HOME to a synthetic per-agent dir alongside CODEX_HOME,
    and then `gh auth status` / git config / etc. all broke inside codex
    shell calls. We avoid the same bug by only overlaying CODEX_HOME and
    RUST_LOG on top of os.environ.copy().
    """

    def test_spawn_env_preserves_HOME(self, monkeypatch):
        """The spawn env must contain the parent process's HOME unchanged.
        Verifies via a subprocess-monkey-patch."""
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                # Provide minimal Popen surface so __init__ doesn't crash
                # on attribute access during construction.
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True  # so close() is a no-op

        # The spawn env must have HOME=/users/alice unchanged
        assert captured["env"].get("HOME") == "/users/alice", (
            f"HOME got rewritten in codex spawn env: "
            f"{captured['env'].get('HOME')!r}. Codex's shell tool's "
            "subprocesses (gh, git, aws, npm) need the user's real HOME."
        )

    def test_spawn_env_sets_CODEX_HOME_when_provided(self, monkeypatch):
        """CODEX_HOME isolation must still work — that's the whole point
        of the codex_home arg."""
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(
            codex_bin="codex", codex_home="/tmp/profile/codex"
        )
        client._closed = True

        assert captured["env"].get("CODEX_HOME") == "/tmp/profile/codex"
        # And HOME still passes through unchanged
        assert captured["env"].get("HOME") == "/users/alice"

    def test_kanban_worker_adds_only_kanban_writable_root(self, monkeypatch):
        """Codex-runtime Kanban workers need to write board state outside
        their scratch/worktree workspace, but should not fall back to
        danger-full-access. Hermes passes a narrow app-server config override
        for the Kanban root only.
        """
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["cmd"] = list(cmd)
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setenv("HOME", "/users/alice")
        monkeypatch.setenv("HERMES_HOME", "/users/alice/.hermes/profiles/backend-worker")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_smoke")
        monkeypatch.setenv(
            "HERMES_KANBAN_DB",
            "/users/alice/.hermes/kanban/boards/smoke/kanban.db",
        )
        monkeypatch.setattr(
            "hermes_cli._subprocess_compat.shutil.which",
            lambda cmd, mode=0, path=None: None,
        )

        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True

        cmd = captured["cmd"]
        assert cmd[:2] == ["codex", "app-server"]
        assert 'sandbox_mode="workspace-write"' in cmd
        assert (
            'sandbox_workspace_write.writable_roots=["/users/alice/.hermes/kanban/boards/smoke"]'
            in cmd
        )
        assert "sandbox_workspace_write.network_access=false" in cmd
        assert all("danger" not in part for part in cmd)


class TestSpawnEnvSecretStripping:
    """codex app-server routes its spawn env through hermes_subprocess_env(
    inherit_credentials=True) instead of a raw os.environ.copy().

    codex is a model-driving CLI executor: it legitimately needs LLM provider
    credentials to authenticate, but it must NOT inherit Tier-1 Hermes secrets
    (gateway bot tokens, GitHub/infra auth, dashboard session token) or the
    dynamic-internal secrets (AUXILIARY_*_API_KEY / _BASE_URL side-LLM keys,
    GATEWAY_RELAY_* relay-auth) — a coding subprocess has no use for those and
    a model-controlled action could exfiltrate them. This closes the #29157
    sibling spawn-site gap (copilot_acp_client already routes through the
    helper; codex app-server predated it).
    """

    @staticmethod
    def _capture_spawn_env(monkeypatch):
        import subprocess
        from agent.transports import codex_app_server as cas

        captured = {}

        class FakePopen:
            def __init__(self, cmd, *args, **kwargs):
                captured["env"] = kwargs.get("env", {}).copy()
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self.pid = 1
                self.returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True
        return captured["env"]

    def test_tier1_and_internal_secrets_stripped_from_spawn_env(self, monkeypatch):
        for var, val in {
            "GH_TOKEN": "ghp-secret",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "HERMES_DASHBOARD_SESSION_TOKEN": "dash-secret",
            "AUXILIARY_VISION_API_KEY": "aux-secret",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "GATEWAY_RELAY_ID": "relay-id",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery",
        }.items():
            monkeypatch.setenv(var, val)

        env = self._capture_spawn_env(monkeypatch)
        for var in (
            "GH_TOKEN", "TELEGRAM_BOT_TOKEN", "MODAL_TOKEN_SECRET",
            "HERMES_DASHBOARD_SESSION_TOKEN", "AUXILIARY_VISION_API_KEY",
            "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_DELIVERY_KEY",
        ):
            assert var not in env, f"{var} leaked into codex app-server spawn env"

    def test_provider_credentials_still_reach_codex(self, monkeypatch):
        """codex authenticates against the model endpoint — provider keys must
        still flow through (inherit_credentials=True)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-codex-needs-this")
        env = self._capture_spawn_env(monkeypatch)
        assert env.get("OPENAI_API_KEY") == "sk-codex-needs-this"

