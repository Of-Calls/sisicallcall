from app.agents.conversational.state import CallState
from app.services.tts.channel import tts_channel
from app.utils.logger import get_logger

# 최종 응답은 TTSOutputChannel 경유로 push_response 호출.

logger = get_logger(__name__)


async def tts_node(state: CallState) -> dict:
    response_text = state.get("response_text") or ""
    if not response_text:
        return {"is_timeout": state.get("is_timeout", False)}

    try:
        await tts_channel.push_response(
            call_id=state["call_id"],
            text=response_text,
            response_path=state.get("response_path") or "unknown",
        )
    except Exception as e:
        logger.error("tts_channel push_response failed call_id=%s: %s", state["call_id"], e)
    return {"is_timeout": state.get("is_timeout", False)}
