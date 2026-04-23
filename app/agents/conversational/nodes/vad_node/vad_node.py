from app.agents.conversational.state import CallState
from app.services.vad.webrtc_vad import WebRTCVADService

_vad_service = WebRTCVADService()

async def vad_node(state: CallState) -> dict:
    is_speech = await _vad_service.detect(state["audio_chunk"])
    return {"is_speech": is_speech}
