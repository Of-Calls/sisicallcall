from app.agents.conversational.state import CallState
from app.services.vad.silero_vad import SileroVADService

_vad_service = SileroVADService()


async def vad_node(state: CallState) -> dict:
    is_speech = await _vad_service.detect(state["audio_chunk"])
    return {"is_speech": is_speech}
