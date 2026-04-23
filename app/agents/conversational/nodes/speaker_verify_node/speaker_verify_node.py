from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def speaker_verify_node(state: CallState) -> dict:
    return {"is_speaker_verified": True}
