from app.agents.conversational.state import CallState
from app.services.vad.webrtc_vad import WebRTCVADService


async def vad_node(state: CallState) -> dict:
    vad_service = WebRTCVADService()
    is_speech = await vad_service.detect(state["audio_chunk"])
    return {
        "is_speech": is_speech,
    }
