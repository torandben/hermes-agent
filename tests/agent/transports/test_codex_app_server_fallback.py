"""Regression: codex_app_server must fall back when Codex quota is exhausted.

Live incident (QueenAlice / Telegram, 2026-09-13): Codex hit its usage limit and
the bot simply failed instead of switching to the configured fallback provider.

Confirmed A/B from profiles/orchestrator/logs/agent.log, same config, same
minute-ish, same error:

* 08:41 session 20260913_084145_49c974 (codex_app_server runtime):
  "codex app-server thread started" -> usage-limit error -> NO fallback lines,
  turn failed.
* 08:58 session 20260913_085853_365d17 (normal chat_completions runtime):
  "Fallback activated: gpt-5.6-terra -> claude-sonnet-5 (anthropic)" -> answered.

Cause: ``agent/conversation_loop.py`` dispatches the codex_app_server runtime
with an early ``return`` *before* the retry/fallback loop, and
``agent/codex_runtime.py`` reports provider failures as ``turn.error`` on a
normally-shaped result dict rather than raising. Nothing on that path ever
consults the fallback chain, so ``fallback_providers`` is silently dead for
every codex_app_server user.

The classifier already understands the message — ``classify_api_error`` returns
``FailoverReason.rate_limit`` with ``should_fallback=True`` for Codex's exact
"You've hit your usage limit ... try again at 11:11 AM." text. Only the runtime
path fails to act on it.

Invariant pinned here: a codex_app_server turn that fails with a
fallback-worthy provider error must report itself as such, so the caller can
route it to the fallback chain instead of surfacing a dead end to the user.
"""

from __future__ import annotations

import pytest

from agent.error_classifier import classify_api_error

# The verbatim text Codex CLI emits when the ChatGPT plan quota is exhausted.
CODEX_QUOTA_ERROR = (
    "You've hit your usage limit. Upgrade to Pro "
    "(https://chatgpt.com/explore/pro), visit "
    "https://chatgpt.com/codex/settings/usage to purchase more credits "
    "or try again at 11:11 AM."
)

# What the app-server transport actually wraps it in (see
# CodexAppServerSession._format_error_with_stderr).
CODEX_TURN_ERROR = f"turn ended status=failed: {CODEX_QUOTA_ERROR}"


class TestCodexQuotaIsFallbackWorthy:
    """The shared classifier must already agree this deserves a fallback."""

    def test_bare_quota_message_classifies_as_fallback_worthy(self):
        verdict = classify_api_error(
            Exception(CODEX_QUOTA_ERROR), provider="openai-codex"
        )
        assert verdict.should_fallback is True

    def test_wrapped_turn_error_classifies_as_fallback_worthy(self):
        """The transport's 'turn ended status=failed: ...' wrapper must not hide it."""
        verdict = classify_api_error(
            Exception(CODEX_TURN_ERROR), provider="openai-codex"
        )
        assert verdict.should_fallback is True

    def test_ordinary_codex_failure_is_not_fallback_worthy(self):
        """A plain tool/protocol error must NOT burn the fallback chain."""
        verdict = classify_api_error(
            Exception("turn ended status=failed: apply_patch rejected the hunk"),
            provider="openai-codex",
        )
        assert verdict.should_fallback is False


class TestCodexRuntimeSignalsFallback:
    """The runtime result must expose the fallback verdict to its caller."""

    def test_quota_failure_is_flagged_for_fallback(self):
        """A quota-failed codex turn must be marked so the caller can fail over.

        Without this the caller cannot distinguish 'Codex is out of quota,
        retry on Anthropic' from 'this turn legitimately produced an error',
        which is exactly why the Telegram bot dead-ended instead of using
        claude-sonnet-5.
        """
        from agent.codex_runtime import codex_turn_error_should_fallback

        assert codex_turn_error_should_fallback(CODEX_TURN_ERROR) is True

    def test_ordinary_failure_is_not_flagged_for_fallback(self):
        from agent.codex_runtime import codex_turn_error_should_fallback

        assert (
            codex_turn_error_should_fallback(
                "turn ended status=failed: apply_patch rejected the hunk"
            )
            is False
        )

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_no_error_is_not_flagged_for_fallback(self, value):
        from agent.codex_runtime import codex_turn_error_should_fallback

        assert codex_turn_error_should_fallback(value) is False

    def test_oauth_hint_is_not_flagged_for_fallback(self):
        """`codex login` problems are user-actionable, not provider capacity."""
        from agent.codex_runtime import codex_turn_error_should_fallback

        assert (
            codex_turn_error_should_fallback(
                "codex app-server startup failed: not logged in. Run `codex login`."
            )
            is False
        )


class TestNoShadowedModuleImports:
    """A function-local re-import must not shadow a module-level name.

    Live regression (2026-09-13): the eager-fallback branch added
    ``from agent.error_classifier import FailoverReason`` *inside*
    ``run_conversation``. Python then treats ``FailoverReason`` as a local for
    the WHOLE function, so every other reference to it — there are 20+, all on
    the normal error-handling path — raised
    ``UnboundLocalError: cannot access local variable 'FailoverReason'``
    whenever the codex branch was not taken. Effect: Telegram answered
    "Sorry, I encountered an unexpected error." for any turn that hit a
    provider error, which is exactly when error handling matters most.
    """

    def test_failover_reason_is_imported_only_at_module_level(self):
        import ast
        import inspect

        from agent import conversation_loop

        tree = ast.parse(inspect.getsource(conversation_loop))

        module_level = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "FailoverReason" in module_level, (
            "expected a module-level FailoverReason import to rely on"
        )

        # Any nested (function-scoped) import of the same name re-binds it as a
        # local for that entire function and breaks unrelated references.
        offenders = []
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for alias in node.names:
                    if (alias.asname or alias.name) in module_level:
                        offenders.append(
                            f"{func.name}() re-imports "
                            f"{alias.name!r} at line {node.lineno}"
                        )

        assert not offenders, (
            "function-local imports shadow module-level names for the whole "
            "function scope (UnboundLocalError risk): " + "; ".join(offenders)
        )


class TestDegradationIsPerTurn:
    """Degrading to the fallback runtime must not disable Codex permanently.

    The dispatcher in ``agent/conversation_loop.py`` flips ``api_mode`` to
    ``chat_completions`` for the failing turn so the retry/fallback loop can
    serve it. If that flip were permanent, the agent would keep answering on
    the fallback provider long after the Codex quota window reset — the user
    silently loses the runtime they configured.
    """

    def _dispatcher_source(self) -> str:
        import inspect

        from agent import conversation_loop

        return inspect.getsource(conversation_loop)

    def test_degradation_sets_a_restore_marker(self):
        src = self._dispatcher_source()
        assert "_codex_runtime_degraded = True" in src

    def test_next_turn_restores_the_codex_runtime(self):
        src = self._dispatcher_source()
        assert 'agent.api_mode = "codex_app_server"' in src
        assert "_codex_runtime_degraded = False" in src

    def test_degraded_turn_drops_the_codex_session(self):
        """A wedged/retired client must not be reused after failover."""
        src = self._dispatcher_source()
        assert "agent._codex_session = None" in src
