"""Regression: interim commentary that IS the turn final must not duplicate.

Live incident (QueenAlice / Telegram, 2026-09-13): every answer arrived twice.

Mechanism, confirmed from ``profiles/orchestrator/logs/gateway.log``:

* ``display.streaming`` is ``false``, so no stream deltas ever fire and the
  consumer's ``final_response_sent`` / ``final_content_delivered`` stay False.
* The ``codex_app_server`` runtime nevertheless emits the completed
  ``agentMessage`` through ``_emit_interim_assistant_message`` →
  ``interim_assistant_callback(text, already_streamed=False)`` →
  ``GatewayStreamConsumer.on_commentary(text)``, which SENDS that text.
* Because nothing streamed, ``already_streamed`` is False, so the gateway sees
  ``previewed=False`` and ``_stream_confirmed_final_delivery`` returns False.
* The gateway then runs its own final send with the identical text — the user
  gets the same message twice, and the run logs
  ``Normal final-send NOT suppressed despite active stream consumer ...
  streamed=False previewed=False content_delivered=False``.

The consumer already records what commentary put on the wire
(``_delivered_commentary_texts``) and already exposes ``has_delivered_text``.
The suppression helper just never consults it unless ``previewed`` is True.

The invariant pinned here: if the exact final text was already delivered as a
visible frame, the gateway must not send it again — regardless of which
callback delivered it.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

FINAL = "Gateway restart completed. Telegram polling is healthy again."


class RecordingAdapter(BasePlatformAdapter):
    """Adapter that records every frame the user would actually see."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.wire: list[tuple[str, str]] = []
        self._next_id = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._next_id += 1
        self.wire.append(("send", content))
        return SendResult(success=True, message_id=f"m-{self._next_id}")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.wire.append(("edit", content))
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        return True


async def _drive_commentary_only(adapter, text: str) -> GatewayStreamConsumer:
    """Reproduce the non-streaming codex_app_server shape.

    No ``on_delta`` calls at all — the whole answer arrives as one completed
    commentary message, exactly as ``_emit_interim_assistant_message`` delivers
    it when ``display.streaming`` is false.
    """
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor=" ▉", edit_interval=0.0)
    )
    task = asyncio.create_task(consumer.run())
    consumer.on_commentary(text)
    await asyncio.sleep(0.05)
    consumer.finish()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    return consumer


@pytest.mark.asyncio
async def test_commentary_delivered_final_is_recorded_on_the_wire():
    """Baseline: the commentary path really did put the final text on the wire."""
    adapter = RecordingAdapter()
    await _drive_commentary_only(adapter, FINAL)

    assert any(
        payload.strip() == FINAL for _, payload in adapter.wire
    ), f"commentary never rendered; wire={adapter.wire!r}"


@pytest.mark.asyncio
async def test_consumer_reports_commentary_delivered_final_text():
    """``has_delivered_text`` must recognise commentary-delivered final text.

    This is the signal the gateway needs to suppress its duplicate send.
    """
    adapter = RecordingAdapter()
    consumer = await _drive_commentary_only(adapter, FINAL)

    assert consumer.has_delivered_text(FINAL) is True


@pytest.mark.asyncio
async def test_gateway_suppresses_final_send_after_commentary_delivered_it():
    """The duplicate-send regression itself.

    ``_stream_confirmed_final_delivery`` previously consulted
    ``has_delivered_text`` only when ``previewed=True``. On the non-streaming
    codex_app_server path nothing streams, so ``previewed`` is False, the helper
    returned False, and the gateway sent the same text a second time.
    """
    from gateway.run import _stream_confirmed_final_delivery

    adapter = RecordingAdapter()
    consumer = await _drive_commentary_only(adapter, FINAL)

    # Sanity: the streaming flags really are unset on this path — the bug is
    # not that some other signal was available and ignored.
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False

    assert (
        _stream_confirmed_final_delivery(consumer, FINAL, previewed=False) is True
    ), (
        "gateway would send the final text again even though commentary already "
        f"delivered it verbatim; wire={adapter.wire!r}"
    )


@pytest.mark.asyncio
async def test_unrelated_commentary_does_not_suppress_a_different_final():
    """Progress chatter must never suppress a genuinely different answer.

    This is the #14238 guarantee that the fix must not regress: commentary
    delivered during the turn is not evidence that THIS final text was seen.
    """
    from gateway.run import _stream_confirmed_final_delivery

    adapter = RecordingAdapter()
    consumer = await _drive_commentary_only(adapter, "Working on it, one moment…")

    assert (
        _stream_confirmed_final_delivery(consumer, FINAL, previewed=False) is False
    ), "unrelated commentary must not suppress the real final response"


@pytest.mark.asyncio
async def test_no_commentary_at_all_does_not_suppress():
    """With nothing delivered, the gateway must still send the final."""
    from gateway.run import _stream_confirmed_final_delivery

    adapter = RecordingAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor=" ▉", edit_interval=0.0)
    )

    assert (
        _stream_confirmed_final_delivery(consumer, FINAL, previewed=False) is False
    )


@pytest.mark.asyncio
async def test_none_consumer_never_suppresses():
    """No consumer means no delivery evidence — always send."""
    from gateway.run import _stream_confirmed_final_delivery

    assert _stream_confirmed_final_delivery(None, FINAL, previewed=False) is False
