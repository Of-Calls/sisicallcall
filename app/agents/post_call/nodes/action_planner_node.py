from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def action_planner_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    try:
        priority = state.get("priority_result") or {}  # type: ignore[call-overload]
        # TODO: 실제 LLM 호출로 교체 (prompts.ACTION_PLAN_PROMPT 사용)
        actions: list[dict] = [
            {
                "action_type": "create_voc_issue",
                "tool": "company_db",
                "params": {"call_id": call_id, "tier": priority.get("tier", "medium")},
                "status": "pending",
            }
        ]
        if priority.get("tier") in ("critical", "high"):
            actions.append({
                "action_type": "send_manager_email",
                "tool": "gmail",
                "params": {
                    "call_id": call_id,
                    "subject": f"[긴급] {call_id} 상담 에스컬레이션",
                    "to": "manager@example.com",
                },
                "status": "pending",
            })
        plan = {"actions": actions, "rationale": "priority 기반 기본 액션 계획"}
        logger.info("action_planner 완료 call_id=%s actions=%d", call_id, len(actions))
        return {"action_plan": plan}
    except Exception as exc:
        logger.error("action_planner 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "action_planner", "error": str(exc)})
        return {"action_plan": None, "errors": errors, "partial_success": True}
