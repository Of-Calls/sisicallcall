"""TitaNet 화자 검증 노드 (대영 R-01 연구 결과 적용).

역할: verify 전담. enrollment 는 enrollment_node 가 담당.
voiceprint 미등록 시 → bypass (is_speaker_verified=True).
"""
from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_service():
    from app.services.speaker_verify.titanet import TitaNetSpeakerVerifyService
    global _service
    if "_service" not in globals():
        globals()["_service"] = TitaNetSpeakerVerifyService()
    return globals()["_service"]


async def speaker_verify_node(state: CallState) -> dict:
    call_id = state["call_id"]
    audio_chunk = state.get("audio_chunk", b"")

    try:
        is_verified, _ = await _get_service().verify(audio_chunk, call_id)
        return {"is_speaker_verified": is_verified}
    except Exception as e:
        logger.error("speaker_verify 실패 call_id=%s: %s — bypass", call_id, e)
        return {"is_speaker_verified": True}
