"""Voiceprint enrollment 헬퍼 — call.py 가 graph 진입 전 직접 호출.

2026-04-30 구조 개편으로 enrollment_node (graph 안) 에서 이관. 그래프가 audio
도메인을 모르게 하는 일환. STT 성공 발화 PCM 만 누적해 settings.titanet_enrollment_sec
도달 시 voiceprint 등록. 빈 STT (잡음) 오디오는 누적 자체 차단.

cleanup() 은 call.py 의 통화 종료 finally 에서 호출 (메모리 해제).
"""
from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service
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
    if not transcript:
        return _done.get(call_id, False)
    # 등록 완료 후에는 PCM 을 더 쌓지 않음 (early return)
    if _done.get(call_id, False):
        return True

    enrollment_target = int(settings.titanet_enrollment_sec * _PCM_BYTES_PER_SEC)
    buf = _buffers.setdefault(call_id, bytearray())
    buf.extend(audio_chunk)

    if len(buf) >= enrollment_target:
        excess = len(buf) - enrollment_target
        # 목표 바이트만 등록에 쓰고 나머지는 즉시 폐기 (로그도 혼동 없이 고정 길이만 표시)
        logger.info(
            "enrollment 목표 PCM 도달 call_id=%s 등록사용=%d bytes (목표=%.1fs) 초과폐기=%d bytes",
            call_id,
            enrollment_target,
            settings.titanet_enrollment_sec,
            excess,
        )
        pcm_for_enrollment = bytes(buf[:enrollment_target])
        buf.clear()
        try:
            await get_titanet_service().extract_and_store(pcm_for_enrollment, call_id)
            await get_finetuned_onnx_service().extract_and_store(pcm_for_enrollment, call_id)
        except Exception as e:
            logger.error("enrollment 실패 call_id=%s: %s", call_id, e)
            buf.extend(pcm_for_enrollment)
            return False

        # extract_and_store 는 내부에서 예외를 삼킬 수 있어, _done 은 실제 voiceprint 확인 후에만 True
        medium = get_titanet_service()
        finetuned = get_finetuned_onnx_service()
        ok_medium = bool(medium.load_error) or medium.has_voiceprint(call_id)
        ok_finetuned = bool(finetuned.load_error) or finetuned.has_voiceprint(call_id)
        if ok_medium and ok_finetuned:
            _done[call_id] = True
            logger.info("voiceprint enrollment 완료 call_id=%s", call_id)
            _buffers.pop(call_id, None)
        else:
            logger.error(
                "enrollment 후 voiceprint 없음(재시도 가능) call_id=%s medium_ok=%s finetuned_ok=%s",
                call_id,
                ok_medium,
                ok_finetuned,
            )
            buf.extend(pcm_for_enrollment)
    else:
        logger.info(
            "enrollment PCM 누적 call_id=%s %d/%d bytes (목표=%.1fs, 빈 STT 제외)",
            call_id,
            len(buf),
            enrollment_target,
            settings.titanet_enrollment_sec,
        )

    return _done.get(call_id, False)


def cleanup(call_id: str) -> None:
    """통화 종료 시 per-call 모듈 전역 dict 메모리 해제."""
    buf_removed = _buffers.pop(call_id, None) is not None
    done_removed = _done.pop(call_id, None) is not None
    if buf_removed or done_removed:
        logger.info("enrollment 상태 삭제 call_id=%s", call_id)


def reset_all_enrollment_state() -> None:
    """모든 통화 enrollment 버퍼·완료 플래그 삭제 (기동 시 voiceprint 초기화와 함께 쓰임)."""
    nb = len(_buffers)
    nd = len(_done)
    _buffers.clear()
    _done.clear()
    logger.info("enrollment 전역 초기화 (buffers=%d, done=%d)", nb, nd)
