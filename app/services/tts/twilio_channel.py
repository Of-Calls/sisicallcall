"""TwilioTTSOutputChannel — 실제 Twilio Media Stream 연동 구현 (RFC 001 v0.2).

Twilio outbound media event format:
    {"event": "media", "streamSid": "...", "media": {"payload": "<base64 μ-law>"}}

lifecycle binding:
    call.py WebSocket 핸들러가 open() 시 websocket + streamSid 전달.
    Channel 이 per-call 매핑을 내부 state 에 저장.
"""
import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Optional

from fastapi import WebSocket

from app.services.tts.base import BaseTTSOutputChannel, BaseTTSService
from app.services.tts.google import GoogleTTSService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Twilio Media Stream chunk 권장 크기: 160 bytes per 20ms at μ-law 8kHz
_TWILIO_CHUNK_BYTES = 160


@dataclass
class _CallBinding:
    websocket: WebSocket
    stream_sid: str
    tenant_id: str
    lock: asyncio.Lock  # 동일 call_id 로 동시 send 방지 (순서 보장)


class TwilioTTSOutputChannel(BaseTTSOutputChannel):
    def __init__(self, tts_service: Optional[BaseTTSService] = None):
        # 지연 초기화 — 주입된 tts_service 가 없으면 첫 synthesize() 호출 시점에 생성.
        self._injected_tts = tts_service
        self._tts: Optional[BaseTTSService] = tts_service
        self._bindings: dict[str, _CallBinding] = {}
        self._stall_emitted: dict[str, bool] = {}

    def _get_tts(self) -> BaseTTSService:
        if self._tts is None:
            self._tts = GoogleTTSService()
        return self._tts

    # ── lifecycle ─────────────────────────────────────────────

    async def open(
        self,
        call_id: str,
        tenant_id: str,
        websocket: Optional[WebSocket] = None,
        stream_sid: Optional[str] = None,
    ) -> None:
        if websocket is None or stream_sid is None:
            logger.error(
                "TwilioTTSOutputChannel.open requires websocket+stream_sid — "
                "call_id=%s", call_id,
            )
            return
        self._bindings[call_id] = _CallBinding(
            websocket=websocket,
            stream_sid=stream_sid,
            tenant_id=tenant_id,
            lock=asyncio.Lock(),
        )
        self._stall_emitted[call_id] = False
        logger.info("tts_channel opened call_id=%s stream_sid=%s", call_id, stream_sid)

    async def flush(self, call_id: str) -> None:
        self._bindings.pop(call_id, None)
        self._stall_emitted.pop(call_id, None)
        logger.info("tts_channel flushed call_id=%s", call_id)

    # ── emit ──────────────────────────────────────────────────

    async def push_stall(self, call_id: str, text: str, audio_field: str) -> None:
        if call_id not in self._bindings:
            logger.warning("push_stall on un-opened call_id=%s — ignored", call_id)
            return
        if self._stall_emitted.get(call_id):
            logger.debug("stall already emitted for call_id=%s — skip", call_id)
            return
        self._stall_emitted[call_id] = True
        await self._synthesize_and_send(call_id, text, tag=f"stall:{audio_field}")

    async def push_response(self, call_id: str, text: str, response_path: str) -> None:
        if call_id not in self._bindings:
            logger.warning("push_response on un-opened call_id=%s — ignored", call_id)
            return
        if not text:
            logger.debug("empty response text call_id=%s — skip", call_id)
            return
        await self._synthesize_and_send(call_id, text, tag=f"response:{response_path}")

    async def cancel(self, call_id: str) -> None:
        """barge-in — Twilio 에 clear 이벤트 송출로 큐 비움."""
        binding = self._bindings.get(call_id)
        if binding is None:
            logger.warning("cancel on un-opened call_id=%s — ignored", call_id)
            return
        clear_msg = json.dumps({
            "event": "clear",
            "streamSid": binding.stream_sid,
        })
        try:
            async with binding.lock:
                await binding.websocket.send_text(clear_msg)
        except Exception as e:
            logger.error("twilio clear send failed call_id=%s: %s", call_id, e)

    # ── 내부 helper ───────────────────────────────────────────

    async def _synthesize_and_send(self, call_id: str, text: str, tag: str) -> None:
        """text → μ-law 8kHz → 160 byte chunk 로 쪼개 Twilio outbound media 송출."""
        binding = self._bindings[call_id]
        try:
            audio = await self._get_tts().synthesize(text)
        except Exception as e:
            logger.error("tts synthesize failed call_id=%s tag=%s: %s", call_id, tag, e)
            return

        logger.info(
            "tts_channel emit call_id=%s tag=%s bytes=%d", call_id, tag, len(audio)
        )

        # 160 byte (20ms) 청크로 쪼개 순차 송신
        async with binding.lock:
            for i in range(0, len(audio), _TWILIO_CHUNK_BYTES):
                chunk = audio[i : i + _TWILIO_CHUNK_BYTES]
                payload = base64.b64encode(chunk).decode("ascii")
                msg = json.dumps({
                    "event": "media",
                    "streamSid": binding.stream_sid,
                    "media": {"payload": payload},
                })
                try:
                    await binding.websocket.send_text(msg)
                except Exception as e:
                    logger.error(
                        "twilio send failed call_id=%s tag=%s chunk=%d: %s",
                        call_id, tag, i // _TWILIO_CHUNK_BYTES, e,
                    )
                    return
