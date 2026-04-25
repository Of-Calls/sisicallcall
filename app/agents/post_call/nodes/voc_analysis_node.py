from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def voc_analysis_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        # TODO: 실제 LLM 호출로 교체 (prompts.VOC_ANALYSIS_PROMPT 사용)
        dummy = {
            "sentiment": "negative",
            "issues": ["긴 대기 시간", "반복 안내"],
            "keywords": ["불만", "요금", "해지"],
            "escalation_reason": None,
            "faq_candidates": ["요금제 변경 방법"],
        }
        logger.info("voc_analysis 완료 call_id=%s sentiment=%s", call_id, dummy["sentiment"])
        return {"voc_analysis": dummy}
    except Exception as exc:
        logger.error("voc_analysis 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "voc_analysis", "error": str(exc)})
        return {"voc_analysis": None, "errors": errors, "partial_success": True}
