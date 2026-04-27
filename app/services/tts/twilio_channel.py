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
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from app.services.tts.base import BaseTTSOutputChannel, BaseTTSService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Twilio Media Stream chunk 권장 크기: 160 bytes per 20ms at μ-law 8kHz
_TWILIO_CHUNK_BYTES = 160


@dataclass
class _CallBinding:
    websocket: WebSocket
    stream_sid: str
    tenant_id: str
    lock: asyncio.Lock                                              # 동일 call_id 로 동시 send 방지 (순서 보장)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)  # barge-in 즉시 중단 신호
    is_speaking: bool = False                                       # 현재 응답 송신 중 여부 (barge-in 감지용)
    current_text: str = ""                                          # 현재 송신 중 응답 원문 (barge-in 컨텍스트용)


class TwilioTTSOutputChannel(BaseTTSOutputChannel):
    def __init__(self, tts_service: Optional[BaseTTSService] = None):
        # 지연 초기화 — 주입된 tts_service 가 없으면 첫 synthesize() 호출 시점에 생성.
        self._injected_tts = tts_service
        self._tts: Optional[BaseTTSService] = tts_service
        self._bindings: dict[str, _CallBinding] = {}
        self._stall_emitted: dict[str, bool] = {}

    def _get_tts(self) -> BaseTTSService:
        if self._tts is None:
            from app.services.tts.azure import AzureTTSService
            logger.info("TTS provider=azure (Speech SDK, Korean Neural Voice)")
            self._tts = AzureTTSService()
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
        """barge-in — 송신 루프 즉시 중단 + Twilio clear 이벤트 송출.

        cancel_event 를 set 하면 _synthesize_and_send 의 청크 루프가 다음 청크
        boundary(≤20ms) 에서 break. clear 이벤트는 락 안에서 송출되므로
        송신 루프가 break 하고 락을 풀 때까지 짧게 대기 후 보내진다.
        """
        binding = self._bindings.get(call_id)
        if binding is None:
            logger.warning("cancel on un-opened call_id=%s — ignored", call_id)
            return
        binding.cancel_event.set()
        clear_msg = json.dumps({
            "event": "clear",
            "streamSid": binding.stream_sid,
        })
        try:
            async with binding.lock:
                await binding.websocket.send_text(clear_msg)
        except Exception as e:
            logger.error("twilio clear send failed call_id=%s: %s", call_id, e)

    # ── 상태 조회 (barge-in 지원) ─────────────────────────────

    def is_speaking(self, call_id: str) -> bool:
        binding = self._bindings.get(call_id)
        return bool(binding and binding.is_speaking)

    def current_text(self, call_id: str) -> str:
        binding = self._bindings.get(call_id)
        return binding.current_text if binding else ""

    # ── 내부 helper ───────────────────────────────────────────

    async def _synthesize_and_send(self, call_id: str, text: str, tag: str) -> None:
        """text → μ-law 8kHz → 160 byte chunk 로 쪼개 Twilio outbound media 송출.

        barge-in 지원:
          - 진입 시 cancel_event 초기화 + is_speaking/current_text 토글
          - 청크 송신 루프 매 iteration 마다 cancel_event 체크 → set 시 즉시 break
          - finally 블록에서 항상 is_speaking=False 로 복원
        """
        binding = self._bindings[call_id]
        # 새 송신 시작 — 이전 cancel 흔적 제거 + 상태 노출
        binding.cancel_event.clear()
        binding.is_speaking = True
        binding.current_text = text

        try:
            try:
                audio = await self._get_tts().synthesize(text)
            except Exception as e:
                logger.error("tts synthesize failed call_id=%s tag=%s: %s", call_id, tag, e)
                return

            # 합성 도중 cancel 가 들어왔으면 송신 자체 skip
            if binding.cancel_event.is_set():
                logger.info("tts cancelled before send call_id=%s tag=%s", call_id, tag)
                return

            logger.info(
                "tts_channel emit call_id=%s tag=%s bytes=%d", call_id, tag, len(audio)
            )

            # 160 byte (20ms) 청크로 쪼개 순차 송신 — 매 청크마다 cancel 체크
            total_chunks = max(len(audio) // _TWILIO_CHUNK_BYTES, 1)
            async with binding.lock:
                for i in range(0, len(audio), _TWILIO_CHUNK_BYTES):
                    if binding.cancel_event.is_set():
                        logger.info(
                            "tts send interrupted (barge-in) call_id=%s tag=%s at chunk=%d/%d",
                            call_id, tag, i // _TWILIO_CHUNK_BYTES, total_chunks,
                        )
                        return
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
        finally:
            binding.is_speaking = False
            binding.current_text = ""
