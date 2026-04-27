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
import time
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
    is_speaking: bool = False                                       # 송신 루프 active 여부 (Python 측 송신 중)
    current_text: str = ""                                          # 현재 송신 중 응답 원문 (barge-in 컨텍스트용)
    # Twilio jitter buffer 의 재생 시간 추정 — 송신은 1초 안에 끝나지만 실제 재생은
    # 11~14초 진행되므로, is_speaking() 이 송신 후에도 이 값으로 재생 진행 중인지 판단.
    play_started_at: float = 0.0                                    # 송신 시작 monotonic 시각
    play_duration_sec: float = 0.0                                  # 전체 재생 예상 시간 = len(audio) / 8000


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
        # play_until 즉시 reset — currently-playing 청크의 echo 가 is_speaking 으로
        # barge-in 재트리거되는 무한 루프 방지. cancel = "지금 끝낸다" 의미.
        binding.play_started_at = 0.0
        binding.play_duration_sec = 0.0
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
        """송신 active OR Twilio jitter buffer 가 재생 진행 중이면 True.

        송신 루프는 빠르게 끝나지만 (~1초) Twilio 가 음성을 11~14초 재생하므로,
        play_started_at + play_duration_sec + tail margin 안에 있는 동안 True 유지.
        """
        binding = self._bindings.get(call_id)
        if binding is None:
            return False
        if binding.is_speaking:
            return True
        if binding.play_started_at == 0.0:
            return False
        elapsed = time.monotonic() - binding.play_started_at
        return elapsed < (binding.play_duration_sec + settings.tts_play_tail_margin_sec)

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

            # play_until 추정 — 송신 시작 시각 + 전체 재생 예상 시간 (μ-law 8kHz mono)
            # finally 블록에서 is_speaking=False 로 토글된 후에도 is_speaking() 메서드가
            # tail margin 동안 True 를 유지하도록 두 필드는 reset 하지 않는다.
            binding.play_started_at = time.monotonic()
            binding.play_duration_sec = len(audio) / 8000

            # 160 byte (20ms) 청크로 쪼개 순차 송신 — 매 청크마다 cancel 체크
            # Throttle (settings.tts_throttle_enabled): preroll 이후 청크부터 재생 속도(20ms)
            # 만큼 sleep 삽입 → Twilio buffer 항상 얇음 → cancel 즉시 효과.
            # asyncio.wait_for(cancel_event.wait()) 패턴으로 sleep 도중 cancel 도 즉시 깨움.
            total_chunks = max(len(audio) // _TWILIO_CHUNK_BYTES, 1)
            preroll = settings.tts_preroll_chunks
            interval = settings.tts_chunk_interval_sec
            throttle_on = settings.tts_throttle_enabled
            bytes_sent = 0
            async with binding.lock:
                for i in range(0, len(audio), _TWILIO_CHUNK_BYTES):
                    idx = i // _TWILIO_CHUNK_BYTES
                    if binding.cancel_event.is_set():
                        # 잔여 재생 시간 단축 — currently-playing 청크가 곧 끝나 is_speaking 도 곧 False
                        binding.play_duration_sec = bytes_sent / 8000
                        logger.info(
                            "tts send interrupted (barge-in) call_id=%s tag=%s at chunk=%d/%d "
                            "played_sec=%.2f bytes_sent=%d total_bytes=%d",
                            call_id, tag, idx, total_chunks,
                            bytes_sent / 8000, bytes_sent, len(audio),
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
                        bytes_sent += len(chunk)
                    except Exception as e:
                        logger.error(
                            "twilio send failed call_id=%s tag=%s chunk=%d: %s",
                            call_id, tag, idx, e,
                        )
                        return
                    # Throttle: preroll 채운 후, 마지막 청크 전까지만 sleep
                    if (
                        throttle_on
                        and idx >= preroll - 1
                        and i + _TWILIO_CHUNK_BYTES < len(audio)
                    ):
                        try:
                            await asyncio.wait_for(
                                binding.cancel_event.wait(), timeout=interval,
                            )
                            # cancel 가 sleep 도중 set → 다음 iteration 의 cancel 분기에서 break
                        except asyncio.TimeoutError:
                            pass  # 정상 throttle
        finally:
            binding.is_speaking = False
            binding.current_text = ""
