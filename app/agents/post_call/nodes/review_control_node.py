"""
Review 제어 보조 노드.

increment_review_retry_node   : retry branch에서 재분석 카운터를 증가시킨다.
mark_human_review_required_node: fail/retry 초과 시 human_review_required=True를 설정한다.
"""
from __future__ import annotations

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def increment_review_retry_node(state: PostCallAgentState) -> dict:
    current = int(state.get("review_retry_count") or 0)  # type: ignore[call-overload]
    new_count = current + 1
    logger.info(
        "increment_review_retry call_id=%s retry_count %d → %d",
        state["call_id"], current, new_count,
    )
    return {"review_retry_count": new_count}


async def mark_human_review_required_node(state: PostCallAgentState) -> dict:
    errors: list = list(state.get("errors", []))  # type: ignore[call-overload]
    review_result: dict = state.get("review_result") or {}  # type: ignore[call-overload]
    reason = review_result.get("reason") or "review_failed"
    errors.append({"node": "review", "error": reason})

    logger.info(
        "mark_human_review_required call_id=%s reason=%r",
        state["call_id"], reason,
    )
    return {
        "human_review_required": True,
        "action_plan": {"action_required": False, "actions": [], "rationale": "human_review_required"},
        "executed_actions": [],
        "errors": errors,
        "partial_success": True,
    }
