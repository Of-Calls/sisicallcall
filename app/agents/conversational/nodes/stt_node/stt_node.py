from app.agents.conversational.state import CallState
from app.services.stt.deepgram import DeepgramSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)
_stt_service: DeepgramSTTService | None = None
_stt_unavailable_reason = ""


def _get_stt_service() -> DeepgramSTTService | None:
    global _stt_service, _stt_unavailable_reason
    if _stt_service:
        return _stt_service
    if _stt_unavailable_reason:
        return None
    try:
        _stt_service = DeepgramSTTService()
    except Exception as e:
        _stt_unavailable_reason = str(e)
        logger.warning("Deepgram STT 비활성화: %s", e)
        return None
    return _stt_service


async def stt_node(state: CallState) -> dict:
    stt_service = _get_stt_service()
    if not stt_service:
        return {"raw_transcript": ""}
    return {
        "raw_transcript": await stt_service.transcribe(state["audio_chunk"]),
    }
