from app.agents.conversational.state import CallState
from app.services.stt.whisper import WhisperSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)

stt_service = WhisperSTTService()


async def stt_node(state: CallState) -> dict:
    return {
        "raw_transcript": await stt_service.transcribe(state["audio_chunk"]),
    }
