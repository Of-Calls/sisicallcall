"""TTS filler audio prewarm — STT 직후 즉시 송출용 짧은 멘트.

Latency 옵션 A: graph + 본 응답 TTS 동안 사용자 무음 → filler 음성으로 갭 채움.
startup 1회 합성, 메모리 dict 재사용 (cold start 회피).
"""
import random

from app.services.tts.base import BaseTTSService
from app.utils.logger import get_logger

_logger = get_logger(__name__)

_FILLER_TEXTS = [
    "네, 알겠습니다",
    "잠시만요",
    "확인해드릴게요",
]
_filler_audios: list[bytes] = []


async def prewarm_fillers(tts: BaseTTSService) -> None:
    """startup 1회 — _FILLER_TEXTS 각각 mulaw 8kHz 합성하여 module 캐시.

    실패해도 startup 진행 (filler 없이 운용 가능). pick_filler() 가 빈 cache 면 None 반환.
    """
    global _filler_audios
    audios: list[bytes] = []
    for text in _FILLER_TEXTS:
        try:
            audio = await tts.synthesize(text)
            if audio:
                audios.append(audio)
        except Exception as exc:
            _logger.warning("filler prewarm 실패 text=%r: %s", text, exc)
    _filler_audios = audios
    _logger.info("filler ready count=%d/%d", len(audios), len(_FILLER_TEXTS))


def pick_filler() -> bytes | None:
    """랜덤 1개 mulaw 8kHz 음성 반환. cache 비어있으면 None (filler skip 신호)."""
    if not _filler_audios:
        return None
    return random.choice(_filler_audios)
