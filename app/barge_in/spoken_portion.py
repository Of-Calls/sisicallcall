"""봇이 실제 재생한 발화 부분 추정 — Twilio mark 기반.

confirm_speculative 이후 호출 — last_interrupted_text 계산에 활용.
mark callback 못 받았으면 시간 기반 fallback (덜 정확).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.barge_in.session_state import BargeInSessionState

_CHUNK_DURATION_MS = 20  # mulaw 8kHz 160 bytes


def get_spoken_portion(session: "BargeInSessionState") -> str:
    """last_bot_utterance 중 사용자가 실제로 들은 prefix 반환.

    한국어 TTS 평균 ~6 글자/초 가정. mark 비율 기반 char_count 추정.
    """
    if not session.last_bot_utterance:
        return ""

    total_chunks = max(1, session.active_tts_chunk_index + 1)
    played_chunks = session.last_played_chunk_index

    if played_chunks <= 0:
        # mark callback 미수신 — 시간 기반 fallback
        if session.tts_started_at <= 0:
            return ""
        elapsed_ms = (time.time() - session.tts_started_at) * 1000
        ratio = min(1.0, elapsed_ms / max(1, total_chunks * _CHUNK_DURATION_MS))
    else:
        ratio = min(1.0, played_chunks / total_chunks)

    char_count = int(len(session.last_bot_utterance) * ratio)
    return session.last_bot_utterance[:char_count]
