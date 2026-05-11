"""TTS chunk-by-chunk 송출 — barge-in pause/resume/cancel 가능.

기존 `_send_audio_to_twilio` (batch 송출) 은 greeting / push_speak 호환 유지.
이 함수는 main TTS / filler 송출 경로에서 BARGE_IN_ENABLED=true 일 때만 사용.

핵심:
1. 매 chunk 송출 전 turn_id stale 검사 — confirm_speculative 가 active_turn_id 증가시키면 즉시 return
2. tts_paused 시 resume_event 대기 (1s timeout 으로 stale 재검사)
3. chunk_ms 만큼 페이싱 — 함수 끝 = 실제 재생 끝 (Twilio 빠른 ack 으로 발생하는 over-send 방지)
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING

from app.barge_in.constants import MARK_INTERVAL_CHUNKS, TTS_CHUNK_BYTES

if TYPE_CHECKING:
    from fastapi import WebSocket
    from app.barge_in.session_state import BargeInSessionState

# mulaw 8kHz → 1 byte = 1 sample = 125us → 20ms = 160 bytes
_CHUNK_DURATION_MS = 20
_CHUNK_DURATION_SEC = _CHUNK_DURATION_MS / 1000


async def stream_tts_audio_chunks(
    websocket: "WebSocket",
    stream_sid: str,
    audio: bytes,
    turn_id: int,
    session: "BargeInSessionState",
    send_marks: bool = False,
) -> None:
    """20ms chunk 단위 TTS 송출 + pause/resume + stale 검사.

    Args:
        turn_id: 호출 시점 캡처한 active_turn_id. confirm_speculative 가
            session.active_turn_id 를 증가시키면 stale 검사로 즉시 return.
        send_marks: Phase 5 에서 True 로 전환 — Twilio mark 송출.
            Phase 2 단계에서는 False (mark 수신 핸들러 미구현).
    """
    if not audio:
        return

    total_chunks = (len(audio) + TTS_CHUNK_BYTES - 1) // TTS_CHUNK_BYTES

    for idx in range(total_chunks):
        # stale 검사 1 — chunk loop 진입 직전
        if turn_id != session.active_turn_id:
            return

        # pause 대기 — speculative 진입 시 tts_paused=True
        # 1s timeout 으로 stale 재검사 (cancel 신호 빠른 반응)
        while session.tts_paused and turn_id == session.active_turn_id:
            try:
                await asyncio.wait_for(
                    session.tts_resume_event.wait(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass
            session.tts_resume_event.clear()

        # stale 검사 2 — pause 도중 turn 바뀌었을 가능성
        if turn_id != session.active_turn_id:
            return

        # 송출
        chunk = audio[idx * TTS_CHUNK_BYTES : (idx + 1) * TTS_CHUNK_BYTES]
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode()},
                }
            )
        )
        session.active_tts_chunk_index = idx

        # Twilio mark (Phase 5 전환)
        if send_marks and idx % MARK_INTERVAL_CHUNKS == 0:
            mark_name = f"turn-{turn_id}-chunk-{idx}"
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": mark_name},
                    }
                )
            )
            session.last_sent_mark = mark_name

        # 실시간 페이싱 — Twilio 빠른 ack 으로 인한 over-send 방지
        # 함수 종료 == 실제 재생 끝 보장
        await asyncio.sleep(_CHUNK_DURATION_SEC)


async def clear_twilio_audio(websocket: "WebSocket", stream_sid: str) -> None:
    """Twilio 미재생 audio 큐 비우기 — confirm_speculative 단계에서만 호출.

    Speculative 단계에서 호출 금지 (사용자 발화 검증 전 클리어 시 false positive 위험).
    """
    await websocket.send_text(
        json.dumps(
            {
                "event": "clear",
                "streamSid": stream_sid,
            }
        )
    )
