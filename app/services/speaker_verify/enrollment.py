"""Voiceprint enrollment 헬퍼 — call.py 가 graph 진입 전 직접 호출.

2026-04-30 구조 개편으로 enrollment_node (graph 안) 에서 이관. 그래프가 audio
도메인을 모르게 하는 일환. STT 성공 발화 PCM 만 누적해 settings.titanet_enrollment_sec
도달 시 voiceprint 등록. 빈 STT (잡음) 오디오는 누적 자체 차단.

cleanup() 은 call.py 의 통화 종료 finally 에서 호출 (메모리 해제).
"""
from app.services.speaker_verify.titanet import get_titanet_service
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PCM_BYTES_PER_SEC = 16000 * 2  # 16kHz 16-bit mono

# per-call enrollment 상태
_buffers: dict[str, bytearray] = {}
_done: dict[str, bool] = {}


async def accumulate(call_id: str, audio_chunk: bytes, transcript: str) -> bool:
    """STT 성공 발화 누적 후 임계 도달 시 voiceprint 등록.

    Returns:
        True: 등록 완료 상태 (이번 호출 또는 이전 호출). False: 미완료.
    """
    if not transcript or _done.get(call_id, False):
        return _done.get(call_id, False)

    enrollment_target = int(settings.titanet_enrollment_sec * _PCM_BYTES_PER_SEC)
    _buffers.setdefault(call_id, bytearray()).extend(audio_chunk)

    logger.debug(
        "enrollment 누적 call_id=%s %d/%d bytes",
        call_id, len(_buffers[call_id]), enrollment_target,
    )

    if len(_buffers[call_id]) >= enrollment_target:
        enrollment_audio = bytes(_buffers.pop(call_id))
        _done[call_id] = True
        try:
            await get_titanet_service().extract_and_store(enrollment_audio, call_id)
        except Exception as e:
            logger.error("enrollment 실패 call_id=%s: %s", call_id, e)
            _done[call_id] = False

    return _done.get(call_id, False)


def cleanup(call_id: str) -> None:
    """통화 종료 시 per-call 모듈 전역 dict 메모리 해제."""
    buf_removed = _buffers.pop(call_id, None) is not None
    done_removed = _done.pop(call_id, None) is not None
    if buf_removed or done_removed:
        logger.info("enrollment 상태 삭제 call_id=%s", call_id)
