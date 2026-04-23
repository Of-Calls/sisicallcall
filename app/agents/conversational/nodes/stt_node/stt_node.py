from app.agents.conversational.state import CallState
from app.services.stt.base import BaseSTTService
from app.services.stt.deepgram import DeepgramSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_stt_service: BaseSTTService = DeepgramSTTService()

# 빈 STT 연속 횟수 추적 (향후 안내 멘트 / escalation 연동 시 사용)
_EMPTY_STT_ESCALATION_THRESHOLD = 3


async def stt_node(state: CallState) -> dict:
    call_id = state.get("call_id", "unknown")

    # Streaming 모드: call.py 에서 flush_transcript() 결과가 미리 담겨 옴 → 패스스루
    pre_filled = (state.get("raw_transcript") or "").strip()
    if pre_filled:
        logger.info("STT 패스스루 (streaming) call_id=%s transcript='%s'", call_id, pre_filled)
        return {"raw_transcript": pre_filled, "empty_stt_count": 0}

    # Fallback: prerecorded (streaming 결과 없거나 연결 실패 시)
    audio_data = state.get("audio_chunk", b"")
    audio_length = len(audio_data)

    logger.info("STT prerecorded fallback call_id=%s | %d bytes", call_id, audio_length)

    try:
        if audio_length == 0:
            logger.warning("STT 경고 call_id=%s | 빈 오디오 청크", call_id)
            return _handle_empty_transcript(state, call_id)

        transcript = await _stt_service.transcribe(audio_data)
        logger.info("STT prerecorded call_id=%s | '%s'", call_id, transcript)

        if not transcript:
            return _handle_empty_transcript(state, call_id)

        return {"raw_transcript": transcript, "empty_stt_count": 0}

    except Exception as e:
        logger.error("STT 실패 call_id=%s | 에러: %s", call_id, e)
        return _handle_empty_transcript(state, call_id)


def _handle_empty_transcript(state: CallState, call_id: str) -> dict:
    """빈 STT 처리 — 횟수만 추적하고 END로 직행 (안내 멘트 없음).

    향후 Barge-in 설계 완성 및 VAD(주미) 도입 후
    안내 멘트 / escalation 연동 재설계 예정.
    """
    count = state.get("empty_stt_count", 0) + 1
    logger.debug("빈 STT %d회 → skip call_id=%s", count, call_id)
    return {"raw_transcript": "", "empty_stt_count": count}