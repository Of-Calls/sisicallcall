"""TTS batch 송출 중 사용자 발화 검증 (Step 2 — inline verify).

흐름 (다른 AI 제안 + 우리 코드 베이스):
1. mulaw → pcm_8k → RMS 측정
2. RMS < bargein_rms_pre_threshold → 무음 카운트, buffer 감쇠 → False
3. RMS 통과 → pcm_8k → pcm_16k 변환 → buffer 누적
4. buffer < bargein_verify_chunk_bytes (0.8s) → False
5. silero VAD speech ratio 체크 → < 0.3 reject
6. TitaNet verify(pcm, call_id) → fail reject
7. 모두 통과 → True

state 는 BargeInSessionState 의 bargein_* 필드.
"""
from __future__ import annotations

import audioop
from typing import TYPE_CHECKING

from app.utils.config import settings

if TYPE_CHECKING:
    from app.barge_in.session_state import BargeInSessionState

_VAD_FRAME_BYTES = 1024  # silero 16kHz 512 samples = 32ms


async def compute_speech_ratio(pcm_16k: bytes, vad) -> float:
    """PCM 16kHz 16-bit mono → silero VAD frame 단위 speech 비율."""
    total = 0
    speech = 0
    for i in range(0, len(pcm_16k) - _VAD_FRAME_BYTES + 1, _VAD_FRAME_BYTES):
        frame = pcm_16k[i : i + _VAD_FRAME_BYTES]
        if await vad.detect(frame):
            speech += 1
        total += 1
    return speech / max(1, total)


async def evaluate_bargein_during_tts(
    mulaw: bytes,
    vad,
    verifier,
    call_id: str,
    session: "BargeInSessionState",
) -> bool:
    """TTS 송출 중 매 media 이벤트마다 호출.

    Returns True 이면 barge-in 확정 (호출자가 cancel + clear + lock 해제).
    """
    if not settings.bargein_verify_enabled:
        return False

    # 동시 verify 호출 방지 (TitaNet 동안 다음 frame 도 verify 가지 않게)
    if session.bargein_verify_inflight:
        return False

    # 1단계: RMS pre-gate
    pcm_8k = audioop.ulaw2lin(mulaw, 2)
    rms = audioop.rms(pcm_8k, 2)

    # 진단 로그 — 매 20프레임 (400ms) 마다
    session.bargein_diag_count += 1
    if session.bargein_diag_count % 20 == 0:
        print(
            f"[BG-DIAG] rms={rms} thr={settings.bargein_rms_pre_threshold} "
            f"buf={len(session.bargein_pcm_buffer)}B"
        )

    if rms < settings.bargein_rms_pre_threshold:
        # pre-gate 미통과 — buffer 점진 감쇠 (echo 잔향이 누적되지 않게)
        # 50 frame = ~1초 무음 — 자연 발화 사이 침묵 보존, 긴 무음만 reset
        if len(session.bargein_pcm_buffer) > 0:
            session.bargein_silence_after_speech += 1
            if session.bargein_silence_after_speech > 50:
                session.bargein_pcm_buffer.clear()
                session.bargein_ratecv_state = None
                session.bargein_silence_after_speech = 0
        return False

    # RMS 통과 — buffer 누적
    session.bargein_silence_after_speech = 0
    pcm_16k, new_state = audioop.ratecv(
        pcm_8k, 2, 1, 8000, 16000, session.bargein_ratecv_state
    )
    session.bargein_ratecv_state = new_state
    session.bargein_pcm_buffer.extend(pcm_16k)

    # 0.8s 미달이면 더 누적
    if len(session.bargein_pcm_buffer) < settings.bargein_verify_chunk_bytes:
        return False

    # verify chunk 추출 + buffer reset
    pcm_for_verify = bytes(session.bargein_pcm_buffer[: settings.bargein_verify_chunk_bytes])
    session.bargein_pcm_buffer.clear()
    session.bargein_ratecv_state = None

    # inflight 플래그 — verify 중 다음 frame 의 verify 차단
    session.bargein_verify_inflight = True
    try:
        # 2단계: silero VAD speech ratio
        try:
            speech_ratio = await compute_speech_ratio(pcm_for_verify, vad)
        except Exception as e:
            print(f"[BG-VERIFY] vad error: {type(e).__name__}: {e}")
            return False
        if speech_ratio < 0.3:
            print(f"[BG-VERIFY] vad reject ratio={speech_ratio:.2f}")
            return False

        # 3단계: TitaNet verify (우리 시그니처: verify(pcm, call_id) → (bool, float))
        try:
            verified, sim = await verifier.verify(pcm_for_verify, call_id)
        except Exception as e:
            print(f"[BG-VERIFY] titanet error: {type(e).__name__}: {e}")
            return False
        if not verified:
            print(f"[BG-VERIFY] titanet reject sim={sim:.2f}")
            return False

        print(
            f"[BARGEIN] triggered rms_last={rms} "
            f"vad_ratio={speech_ratio:.2f} sim={sim:.2f}"
        )
        return True
    finally:
        session.bargein_verify_inflight = False


def reset_bargein_state(session: "BargeInSessionState") -> None:
    """TTS 끝/cancel 후 호출 — 다음 사용자 turn 깨끗하게."""
    session.bargein_pcm_buffer.clear()
    session.bargein_ratecv_state = None
    session.bargein_silence_after_speech = 0
    session.bargein_diag_count = 0
    session.bargein_verify_inflight = False
