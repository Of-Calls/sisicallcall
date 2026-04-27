"""화자 voiceprint 등록 노드 — STT 성공 발화만 enrollment에 사용.

위치: stt_node 이후 → cache_node 이전
역할: raw_transcript 있는 오디오만 누적해 voiceprint 등록.
      빈 STT(잡음) 오디오는 enrollment에서 완전 차단.

voiceprint 등록 완료 이후에는 speaker_verify_node 가 verify 를 수행.
"""
from app.agents.conversational.state import CallState
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PCM_BYTES_PER_SEC = 16000 * 2  # 16kHz 16-bit mono

# per-call enrollment 상태 — STT 성공 오디오만 누적
_enrollment_buffers: dict[str, bytearray] = {}
_enrollment_done: dict[str, bool] = {}


def _get_service():
    from app.services.speaker_verify.titanet import get_titanet_service
    return get_titanet_service()


async def enrollment_node(state: CallState) -> dict:
    """STT 성공 발화만 누적해 voiceprint 등록."""
    call_id = state["call_id"]
    transcript = state.get("raw_transcript", "")
    audio_chunk = state.get("audio_chunk", b"")

    # STT 실패 or voiceprint 이미 등록 완료 → 즉시 반환
    if not transcript or _enrollment_done.get(call_id, False):
        return {"enrollment_done": _enrollment_done.get(call_id, False)}

    enrollment_target = int(settings.titanet_enrollment_sec * _PCM_BYTES_PER_SEC)
    _enrollment_buffers.setdefault(call_id, bytearray()).extend(audio_chunk)

    logger.debug(
        "enrollment 누적 call_id=%s %d/%d bytes",
        call_id, len(_enrollment_buffers[call_id]), enrollment_target,
    )

    if len(_enrollment_buffers[call_id]) >= enrollment_target:
        enrollment_audio = bytes(_enrollment_buffers.pop(call_id))
        _enrollment_done[call_id] = True
        try:
            await _get_service().extract_and_store(enrollment_audio, call_id)
        except Exception as e:
            logger.error("enrollment 실패 call_id=%s: %s", call_id, e)
            _enrollment_done[call_id] = False

    return {"enrollment_done": _enrollment_done.get(call_id, False)}
