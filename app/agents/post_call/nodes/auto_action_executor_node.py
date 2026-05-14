"""human_queue 분기에서도 자동 액션 (Notion 회사 DB 기록) 만 실행한다.

reviewer 가 verdict=fail (max retry 초과) 을 내려 human_queue 에 도달했더라도
회사 DB 기록은 분석 품질과 무관하게 보장돼야 한다. 이 노드는 state["proposed_actions"]
중 params.auto_injected=True 인 것만 골라 ActionExecutor 로 위임한다.

LLM-proposed 액션은 차단된 상태 그대로 — human_queue 가 사람 검토 대상에 포함시킨다.
"""
from __future__ import annotations

from app.agents.post_call.actions.executor import ActionExecutor
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)
_executor = ActionExecutor()


async def auto_action_executor_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    tenant_id: str = state.get("tenant_id", "") or ""  # type: ignore[call-overload]
    proposed: list = list(state.get("proposed_actions") or [])  # type: ignore[call-overload]

    auto_actions = [
        a for a in proposed if (a.get("params") or {}).get("auto_injected")
    ]
    existing_executed = list(state.get("executed_actions") or [])  # type: ignore[call-overload]

    if not auto_actions:
        logger.info(
            "auto_action_executor: auto_injected 액션 없음 call_id=%s — skip",
            call_id,
        )
        return {"executed_actions": existing_executed}

    try:
        executed = await _executor.execute_actions(
            call_id=call_id,
            tenant_id=tenant_id,
            actions=auto_actions,
        )
        failed = [a for a in executed if a.get("status") == "failed"]
        logger.info(
            "auto_action_executor 완료 call_id=%s executed=%d failed=%d (human_queue 분기)",
            call_id, len(executed), len(failed),
        )
        out: dict = {
            "executed_actions": existing_executed + executed,
        }
        if failed:
            out["partial_success"] = True
        return out
    except Exception as exc:
        logger.error("auto_action_executor 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "auto_action_executor", "error": str(exc)})
        return {
            "executed_actions": existing_executed,
            "errors": errors,
            "partial_success": True,
        }
