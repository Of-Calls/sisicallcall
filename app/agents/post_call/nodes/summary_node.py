from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def summary_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        # TODO: 실제 LLM 호출로 교체 (prompts.SUMMARY_PROMPT 사용)
        dummy = {
            "summary_text": f"[dummy] call_id={call_id} 상담 내용 요약",
            "call_duration_sec": 180,
            "customer_intent": "요금 문의",
            "resolution_status": "resolved",
            "key_topics": ["요금 문의", "해지 안내"],
        }
        logger.info("summary 완료 call_id=%s status=%s", call_id, dummy["resolution_status"])
        return {"summary": dummy}
    except Exception as exc:
        logger.error("summary 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "summary", "error": str(exc)})
        return {"summary": None, "errors": errors}
