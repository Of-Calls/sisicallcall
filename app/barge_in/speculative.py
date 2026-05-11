"""Speculative interrupt 상태머신.

Phase 3 단계: enter / reject / confirm / timeout. verification 없이 timeout 으로만 동작.
즉 BARGE_IN_ENABLED=true 시 사용자 350ms 발화 → TTS 800ms pause → 자동 재개.

Phase 4 에서 verify_gate 가 confirm/reject 호출 — Phase 3 에서는 confirm 호출자 없음.
"""
from __future__ import annotations

import asyncio
import audioop
from typing import TYPE_CHECKING

from app.barge_in.constants import (
    FALSE_SPECULATIVE_RESET_COUNT,
    SPECULATIVE_TIMEOUT_MS,
    TTS_GRACE_PERIOD_MS,
    VAD_SPECULATIVE_MIN_MS,
    VAD_SPECULATIVE_MIN_MS_COOLDOWN_BUMP,
)
from app.barge_in.types import BargeInRejectReason, BargeInState

if TYPE_CHECKING:
    from fastapi import WebSocket
    from app.barge_in.session_state import BargeInSessionState

_VAD_FRAME_BYTES = 1024  # call.py 와 동일 — Silero 16kHz 512 samples


async def evaluate_speculative_frame(
    pcm_16k_chunk: bytes,
    session: "BargeInSessionState",
    vad,
    pre_buf: bytearray,
    now: float,
    verifier=None,
    call_id: str | None = None,
    websocket=None,
    stream_sid: str | None = None,
) -> None:
    """media 이벤트마다 호출 — pcm 누적 + VAD + speculative 진입 시도.

    is_speaking=True 인 동안만 호출 — TTS 송출 중 사용자 발화 평가.
    pre_buf 는 nonlocal bytearray (1024B 정렬용 임시 누적).
    verifier/call_id/websocket/stream_sid 가 제공되면 enter_speculative 시
    verify_gate 백그라운드 spawn (Phase 4).
    """
    pre_buf.extend(pcm_16k_chunk)

    while len(pre_buf) >= _VAD_FRAME_BYTES:
        frame = bytes(pre_buf[:_VAD_FRAME_BYTES])
        del pre_buf[:_VAD_FRAME_BYTES]

        is_speech = await vad.detect(frame)

        # 임시 진단 — RMS > 200 인 frame 만 (의미 있는 신호 후보)
        rms = audioop.rms(frame, 2)
        if rms > 200:
            print(f"[BG-DBG] frame rms={rms} vad_speech={is_speech}")

        if is_speech:
            if not session.speech_active:
                session.speech_active = True
                session.speech_started_at = now
                print("[BG-DBG] speech 시작 detected")  # 임시 진단

            speech_duration_ms = (now - session.speech_started_at) * 1000

            if session.barge_in_state == BargeInState.SPECULATIVE_INTERRUPT:
                # 이미 진입 — candidate 누적 (verification 입력)
                session.candidate_pcm.extend(frame)
            elif session.barge_in_state == BargeInState.IDLE:
                # 진입 시도
                await enter_speculative(
                    session, now, speech_duration_ms,
                    verifier=verifier, call_id=call_id,
                    websocket=websocket, stream_sid=stream_sid,
                )
                if session.barge_in_state == BargeInState.SPECULATIVE_INTERRUPT:
                    session.candidate_pcm.extend(frame)
        else:
            # 침묵 — speech_active 해제. candidate 는 보존 (verification 진행 중일 수 있음).
            session.speech_active = False


async def enter_speculative(
    session: "BargeInSessionState",
    now: float,
    speech_duration_ms: float,
    verifier=None,
    call_id: str | None = None,
    websocket=None,
    stream_sid: str | None = None,
) -> None:
    """SPECULATIVE_INTERRUPT 진입 시도.

    게이트:
    1. IDLE 상태일 것
    2. TTS 시작 후 grace period 경과 (echo/잔향 차단)
    3. speech 누적 시간이 임계 이상 (cooldown 발동 시 상향)

    verifier/ws/sid 제공 시 verify_gate spawn, 없으면 _speculative_timeout 만 (Phase 3 fallback).
    """
    if session.barge_in_state != BargeInState.IDLE:
        print(f"[BG-DBG] gate: not_idle ({session.barge_in_state.value})")  # 임시 진단
        return

    grace_ms = (now - session.tts_started_at) * 1000
    if grace_ms < TTS_GRACE_PERIOD_MS:
        print(f"[BG-DBG] gate: grace ({grace_ms:.0f}ms < {TTS_GRACE_PERIOD_MS})")  # 임시 진단
        return

    # cooldown — 연속 false reject 누적 시 임계 상향
    effective_min_ms = VAD_SPECULATIVE_MIN_MS
    if session.false_speculative_count >= FALSE_SPECULATIVE_RESET_COUNT:
        effective_min_ms += VAD_SPECULATIVE_MIN_MS_COOLDOWN_BUMP

    if speech_duration_ms < effective_min_ms:
        print(f"[BG-DBG] gate: short_dur ({speech_duration_ms:.0f}ms < {effective_min_ms})")  # 임시 진단
        return

    # 진입
    session.barge_in_state = BargeInState.SPECULATIVE_INTERRUPT
    session.tts_paused = True
    session.tts_resume_event.clear()
    session.speculative_started_at = now
    print(
        f"[BG] speculative 진입 dur={speech_duration_ms:.0f}ms "
        f"false_count={session.false_speculative_count}"
    )

    # Phase 4 — verifier 제공 시 verify_gate. 없으면 timeout 만.
    if verifier is not None and call_id and websocket is not None and stream_sid:
        from app.barge_in.verify_gate import run_verification_gate
        asyncio.create_task(
            run_verification_gate(session, websocket, stream_sid, verifier, call_id)
        )
    else:
        asyncio.create_task(_speculative_timeout(session))


async def _speculative_timeout(session: "BargeInSessionState") -> None:
    """SPECULATIVE_TIMEOUT_MS 안에 결과 안 나오면 자동 reject."""
    await asyncio.sleep(SPECULATIVE_TIMEOUT_MS / 1000)
    if session.barge_in_state == BargeInState.SPECULATIVE_INTERRUPT:
        await reject_speculative(session, BargeInRejectReason.TIMEOUT)


async def reject_speculative(
    session: "BargeInSessionState",
    reason: BargeInRejectReason,
) -> None:
    """SPECULATIVE → IDLE. TTS 재개. candidate 폐기."""
    if session.barge_in_state != BargeInState.SPECULATIVE_INTERRUPT:
        return
    session.barge_in_state = BargeInState.IDLE
    session.tts_paused = False
    session.tts_resume_event.set()
    session.candidate_pcm.clear()
    session.speech_active = False
    session.false_speculative_count += 1
    print(
        f"[BG] reject reason={reason.value} "
        f"false_count={session.false_speculative_count}"
    )


async def confirm_speculative(
    session: "BargeInSessionState",
    websocket: "WebSocket",
    stream_sid: str,
) -> None:
    """SPECULATIVE → BARGE_IN_CONFIRMED. turn_id 증가 + Twilio clear + task cancel.

    Phase 3 단계에서는 호출자 없음 (verification 미구현).
    Phase 4 에서 verify_gate 통과 시 호출.
    """
    from app.services.tts.streaming import clear_twilio_audio

    if session.barge_in_state != BargeInState.SPECULATIVE_INTERRUPT:
        return

    session.barge_in_state = BargeInState.BARGE_IN_CONFIRMED
    session.active_turn_id += 1
    session.false_speculative_count = 0
    print(f"[BG] CONFIRMED turn_id={session.active_turn_id}")

    # Twilio 미재생 큐 비우기 — confirm 단계에서만 (speculative 에서 X)
    try:
        await clear_twilio_audio(websocket, stream_sid)
    except Exception as exc:
        print(f"[BG] clear failed: {exc}")

    # 진행 중 task cancel — streaming loop 의 stale 검사가 이미 잡지만 명시적
    if session.active_tts_task and not session.active_tts_task.done():
        session.active_tts_task.cancel()
    if session.active_filler_task and not session.active_filler_task.done():
        session.active_filler_task.cancel()

    # pause 해제 — cancel 빠른 반응
    session.tts_paused = False
    session.tts_resume_event.set()
