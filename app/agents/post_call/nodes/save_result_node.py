from __future__ import annotations
from app.agents.post_call.state import PostCallAgentState
from app.repositories.call_summary_repo import CallSummaryRepository
from app.repositories.voc_analysis_repo import VOCAnalysisRepository
from app.repositories.mcp_action_log_repo import MCPActionLogRepository
from app.repositories.dashboard_repo import DashboardRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)
_summary_repo = CallSummaryRepository()
_voc_repo = VOCAnalysisRepository()
_action_log_repo = MCPActionLogRepository()
_dashboard_repo = DashboardRepository()


async def save_result_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    errors: list[dict] = list(state.get("errors", []))  # type: ignore[call-overload]

    for label, coro in [
        ("summary", _maybe_save_summary(call_id, state)),
        ("voc", _maybe_save_voc(call_id, state)),
        ("action_log", _maybe_save_actions(call_id, state)),
    ]:
        try:
            await coro
        except Exception as exc:
            errors.append({"node": f"save_result:{label}", "error": str(exc)})

    payload = {
        "call_id": call_id,
        "tenant_id": state["tenant_id"],
        "trigger": state["trigger"],
        "summary": state.get("summary"),  # type: ignore[call-overload]
        "voc_analysis": state.get("voc_analysis"),  # type: ignore[call-overload]
        "priority_result": state.get("priority_result"),  # type: ignore[call-overload]
        "action_plan": state.get("action_plan"),  # type: ignore[call-overload]
        "executed_actions": state.get("executed_actions", []),  # type: ignore[call-overload]
        "errors": errors,
        "partial_success": state.get("partial_success", False),  # type: ignore[call-overload]
    }

    try:
        await _dashboard_repo.upsert_dashboard(call_id, payload)
    except Exception as exc:
        errors.append({"node": "save_result:dashboard", "error": str(exc)})

    logger.info("save_result 완료 call_id=%s errors=%d", call_id, len(errors))
    return {"dashboard_payload": payload, "errors": errors}


async def _maybe_save_summary(call_id: str, state: PostCallAgentState) -> None:
    if state.get("summary"):  # type: ignore[call-overload]
        await _summary_repo.save_summary(call_id, state["summary"])  # type: ignore[typeddict-item]


async def _maybe_save_voc(call_id: str, state: PostCallAgentState) -> None:
    if state.get("voc_analysis"):  # type: ignore[call-overload]
        await _voc_repo.save_voc_analysis(call_id, state["voc_analysis"])  # type: ignore[typeddict-item]


async def _maybe_save_actions(call_id: str, state: PostCallAgentState) -> None:
    actions = state.get("executed_actions", [])  # type: ignore[call-overload]
    if actions:
        await _action_log_repo.save_action_log(call_id, actions)
