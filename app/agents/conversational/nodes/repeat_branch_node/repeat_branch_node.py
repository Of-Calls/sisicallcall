from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_REPEAT = "죄송합니다, 이전 답변이 없습니다. 무엇을 도와드릴까요?"


async def repeat_branch_node(state: CallState) -> dict:
    """이전 AI 응답을 그대로 재전달 — '다시 한번 말해주세요' 류 요청 처리.

    session_view["last_assistant_text"] 에서 이전 응답을 꺼내 response_text 로 반환.
    이전 응답이 없으면(첫 턴) _DEFAULT_REPEAT 로 폴백.
    """
    call_id = state.get("call_id", "unknown")
    sv = state.get("session_view") or {}
    last_text = (sv.get("last_assistant_text") or "").strip()

    if last_text:
        logger.info(
            "repeat branch call_id=%s prev_len=%d",
            call_id, len(last_text),
        )
        return {
            "response_text": last_text,
            "response_path": "repeat",
            "is_fallback": False,
            "is_timeout": False,
        }

    logger.info("repeat branch call_id=%s — no last_assistant_text, fallback", call_id)
    return {
        "response_text": _DEFAULT_REPEAT,
        "response_path": "repeat",
        "is_fallback": True,
        "is_timeout": False,
    }
