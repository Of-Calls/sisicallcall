"""TitaNet 화자 검증 노드 (대영 R-01 연구 결과 적용).

동작 흐름:
    1. 첫 발화부터 settings.titanet_enrollment_sec 초 누적 → voiceprint 등록
    2. 등록 완료 이후 → 매 발화마다 코사인 유사도 검증
    3. voiceprint 미등록(누적 중) → bypass (is_speaker_verified=True)

목적: 주변 잡음·타인 발화가 VAD를 통과해도 등록된 화자 목소리만 STT로 전달.
"""
from app.agents.conversational.state import CallState
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# PCM 16kHz 16-bit mono 기준 초당 바이트 수
_PCM_BYTES_PER_SEC = 16000 * 2

# per-call enrollment 상태 — module-level 관리 (call.py 무수정)
_enrollment_buffers: dict[str, bytearray] = {}
_enrollment_done: dict[str, bool] = {}


def _get_service():
    """TitaNet 서비스 지연 초기화 — 첫 호출 시 모델 로딩."""
    from app.services.speaker_verify.titanet import TitaNetSpeakerVerifyService
    global _service
    if "_service" not in globals():
        globals()["_service"] = TitaNetSpeakerVerifyService()
    return globals()["_service"]


async def speaker_verify_node(state: CallState) -> dict:
    call_id = state["call_id"]
    audio_chunk = state["audio_chunk"]

    enrollment_target = int(settings.titanet_enrollment_sec * _PCM_BYTES_PER_SEC)

    # ── enrollment 단계 ──────────────────────────────────────────────────────
    if not _enrollment_done.get(call_id, False):
        _enrollment_buffers.setdefault(call_id, bytearray()).extend(audio_chunk)

        if len(_enrollment_buffers[call_id]) >= enrollment_target:
            enrollment_audio = bytes(_enrollment_buffers.pop(call_id))
            _enrollment_done[call_id] = True
            try:
                await _get_service().extract_and_store(enrollment_audio, call_id)
            except Exception as e:
                logger.error("enrollment 실패 call_id=%s: %s", call_id, e)
                _enrollment_done[call_id] = False

        # 누적 중 → bypass
        return {"is_speaker_verified": True}

    # ── verify 단계 ──────────────────────────────────────────────────────────
    try:
        is_verified, similarity = await _get_service().verify(audio_chunk, call_id)
        return {"is_speaker_verified": is_verified}
    except Exception as e:
        logger.error("speaker_verify 실패 call_id=%s: %s — bypass", call_id, e)
        return {"is_speaker_verified": True}
