from app.agents.conversational.state import CallState
from app.services.stt.base import BaseSTTService
from app.services.stt.deepgram import DeepgramSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_stt_service: BaseSTTService = DeepgramSTTService()

async def stt_node(state: CallState) -> dict:
    call_id = state.get("call_id", "unknown")
    audio_data = state.get("audio_chunk", b"")
    audio_length = len(audio_data)

    # 1. 입력 데이터 크기 디버깅
    logger.info(f"STT 디버깅 call_id={call_id} | 입력 데이터 크기: {audio_length} bytes")

    try:
        if audio_length == 0:
            logger.warning(f"STT 경고 call_id={call_id} | 빈 오디오 청크가 전달됨")
            return {"raw_transcript": ""}

        # 2. STT 변환
        transcript = await _stt_service.transcribe(audio_data)
        
        # 3. 결과 텍스트 디버깅
        logger.info(f"STT 디버깅 call_id={call_id} | 변환 결과: '{transcript}'")
        
        return {"raw_transcript": transcript}
        
    except Exception as e:
        logger.error(f"STT 실패 call_id={call_id} | 에러: {e}")
        return {"raw_transcript": "", "error": str(e)}