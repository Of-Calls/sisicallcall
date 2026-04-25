from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.actions.executor import ActionExecutor
from app.utils.logger import get_logger

logger = get_logger(__name__)
_executor = ActionExecutor()


async def action_router_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    plan = state.get("action_plan")  # type: ignore[call-overload]
    if not plan:
        logger.warning("action_router: action_plan 없음 call_id=%s — 건너뜀", call_id)
        return {"executed_actions": [], "partial_success": True}

    try:
        executed = await _executor.execute_all(plan["actions"], call_id=call_id)
        failed = [a for a in executed if a.get("status") == "failed"]
        logger.info(
            "action_router 완료 call_id=%s executed=%d failed=%d",
            call_id, len(executed), len(failed),
        )
        return {
            "executed_actions": executed,
            "partial_success": len(failed) > 0,
        }
    except Exception as exc:
        logger.error("action_router 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "action_router", "error": str(exc)})
        return {"executed_actions": [], "errors": errors, "partial_success": True}
