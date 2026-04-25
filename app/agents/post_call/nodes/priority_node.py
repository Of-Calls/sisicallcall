from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def priority_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        voc = state.get("voc_analysis") or {}  # type: ignore[call-overload]
        # TODO: 실제 LLM 호출로 교체 (prompts.PRIORITY_PROMPT 사용)
        score = 4 if voc.get("sentiment") == "negative" else 2
        dummy = {
            "score": score,
            "tier": "high" if score >= 4 else "medium",
            "reason": "고객 부정 감정 감지" if score >= 4 else "일반 문의",
        }
        logger.info("priority 완료 call_id=%s tier=%s score=%d", call_id, dummy["tier"], score)
        return {"priority_result": dummy}
    except Exception as exc:
        logger.error("priority 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "priority", "error": str(exc)})
        return {"priority_result": None, "errors": errors, "partial_success": True}
