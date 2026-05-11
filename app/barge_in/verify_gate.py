"""Speaker verification gate for barge-in.

Speculative interrupt 진입 직후 spawn 되는 백그라운드 task.
우리 TitaNet ONNX `_verifier.verify(pcm, call_id)` 인터페이스 직접 호출.

흐름:
1. candidate_pcm 길이 ≥ VERIFY_MIN_AUDIO_MS 까지 대기
2. RMS energy 체크 — leak/저음량 노이즈 차단 (STT interim 없는 우리 echo filter 대안)
3. enrollment 체크 — 미등록이면 안전 모드 (barge-in 비활성)
4. verify 호출 → pass: confirm / fail: reject
5. timeout: reject (TIMEOUT)
"""
from __future__ import annotations

import audioop
import asyncio
import time
from typing import TYPE_CHECKING

from app.barge_in.constants import (
    ENERGY_RMS_MIN_THRESHOLD,
    SAMPLE_RATE_HZ,
    SPECULATIVE_TIMEOUT_MS,
    VERIFY_MIN_AUDIO_MS,
)
from app.barge_in.types import BargeInRejectReason, BargeInState

if TYPE_CHECKING:
    from fastapi import WebSocket
    from app.barge_in.session_state import BargeInSessionState


def compute_rms(pcm_16k: bytes) -> float:
    """linear16 PCM RMS energy (0~32767)."""
    return audioop.rms(pcm_16k, 2)


async def run_verification_gate(
    session: "BargeInSessionState",
    websocket: "WebSocket",
    stream_sid: str,
    verifier,
    call_id: str,
) -> None:
    """Speculative 진입 직후 spawn — verification 결과로 confirm/reject."""
    # 지연 import — 순환 회피 (speculative ↔ verify_gate)
    from app.barge_in.speculative import confirm_speculative, reject_speculative

    start = time.time()

    while True:
        # 다른 곳에서 reject 됐을 수도 (timeout race)
        if session.barge_in_state != BargeInState.SPECULATIVE_INTERRUPT:
            return

        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms >= SPECULATIVE_TIMEOUT_MS:
            await reject_speculative(session, BargeInRejectReason.TIMEOUT)
            return

        # candidate 길이 — linear16 = 2 bytes/sample
        candidate_ms = len(session.candidate_pcm) / 2 / SAMPLE_RATE_HZ * 1000
        if candidate_ms < VERIFY_MIN_AUDIO_MS:
            await asyncio.sleep(0.05)
            continue

        candidate_pcm = bytes(session.candidate_pcm)

        # Echo filter 대안 — RMS energy. leak/저음량 노이즈 차단.
        rms = compute_rms(candidate_pcm)
        if rms < ENERGY_RMS_MIN_THRESHOLD:
            print(f"[BG] verify_gate: low energy rms={rms:.0f} < {ENERGY_RMS_MIN_THRESHOLD}")
            await reject_speculative(session, BargeInRejectReason.LOW_ENERGY)
            return

        # Enrollment 체크 — 미등록 시 verify 가 (True, 1.0) bypass 반환하므로 명시 차단.
        # 안전 모드: 본인 voiceprint 없으면 barge-in 자체 비활성.
        if not verifier.is_enrolled(call_id):
            print("[BG] verify_gate: not enrolled — safe mode reject")
            await reject_speculative(session, BargeInRejectReason.NOT_ENROLLED)
            return

        # rolling buffer prepend — 앞쪽 잘림 방지 (300ms)
        if session.rolling_pcm_buffer is not None:
            prepend = session.rolling_pcm_buffer.get_recent_ms(300)
            audio_for_verify = prepend + candidate_pcm
        else:
            audio_for_verify = candidate_pcm

        try:
            verified, sim = await verifier.verify(audio_for_verify, call_id)
        except Exception as e:
            print(f"[BG] verify error: {type(e).__name__}: {e}")
            await reject_speculative(session, BargeInRejectReason.VERIFY_ERROR)
            return

        if verified:
            print(f"[BG] verify pass sim={sim:.3f} rms={rms:.0f} candidate_ms={candidate_ms:.0f}")
            await confirm_speculative(session, websocket, stream_sid)
        else:
            print(f"[BG] verify fail sim={sim:.3f}")
            await reject_speculative(session, BargeInRejectReason.VERIFY_FAILED)
        return
