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
    # 이전 노드들이 누적한 errors 를 수집
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

    # errors 유무로 partial_success 최종 결정.
    # action_router_node 등 이전 노드가 partial_success 를 False 로 덮어썼을 수 있으므로
    # 이 노드가 authoritative setter 역할을 한다.
    final_partial_success = len(errors) > 0

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
        "partial_success": final_partial_success,
        # ── Review Gate 필드 ─────────────────────────────────────────────
        "analysis_result": state.get("analysis_result"),  # type: ignore[call-overload]
        "review_result": state.get("review_result"),  # type: ignore[call-overload]
        "review_verdict": state.get("review_verdict"),  # type: ignore[call-overload]
        "review_retry_count": state.get("review_retry_count", 0),  # type: ignore[call-overload]
        "human_review_required": state.get("human_review_required", False),  # type: ignore[call-overload]
        "blocked_actions": state.get("blocked_actions", []),  # type: ignore[call-overload]
    }

    try:
        await _dashboard_repo.upsert_dashboard(call_id, payload)
    except Exception as exc:
        errors.append({"node": "save_result:dashboard", "error": str(exc)})
        final_partial_success = True
        payload["errors"] = errors
        payload["partial_success"] = final_partial_success

    logger.info("save_result 완료 call_id=%s errors=%d partial_success=%s",
                call_id, len(errors), final_partial_success)
    return {
        "dashboard_payload": payload,
        "errors": errors,
        "partial_success": final_partial_success,
    }


async def _maybe_save_summary(call_id: str, state: PostCallAgentState) -> None:
    summary = _summary_payload(state)
    if summary:
        await _summary_repo.save_summary(
            call_id,
            summary,
            tenant_id=state["tenant_id"],
        )


async def _maybe_save_voc(call_id: str, state: PostCallAgentState) -> None:
    voc = _voc_payload(state)
    if voc:
        await _voc_repo.save_voc_analysis(
            call_id,
            voc,
            tenant_id=state["tenant_id"],
            partial_success=bool(state.get("partial_success", False)),  # type: ignore[call-overload]
            failed_subagents=_failed_subagents(state),
        )


async def _maybe_save_actions(call_id: str, state: PostCallAgentState) -> None:
    actions = state.get("executed_actions", [])  # type: ignore[call-overload]
    if actions:
        await _action_log_repo.save_action_log(
            call_id,
            actions,
            tenant_id=state["tenant_id"],
        )


def _summary_payload(state: PostCallAgentState) -> dict:
    summary = dict(state.get("summary") or {})  # type: ignore[call-overload]
    analysis = state.get("analysis_result") or {}  # type: ignore[call-overload]
    if not summary and isinstance(analysis, dict):
        summary = dict(analysis.get("summary") or {})
    if not summary:
        return {}

    summary.setdefault("summary_short", "")
    summary.setdefault("summary_detailed", None)
    summary.setdefault("customer_intent", summary.get("intent"))
    summary.setdefault("customer_emotion", summary.get("emotion", "neutral"))
    summary.setdefault("resolution_status", "resolved")
    summary.setdefault("keywords", [])
    summary.setdefault("handoff_notes", None)
    summary.setdefault("generation_mode", "async")
    summary.setdefault("model_used", "demo-mock-llm")
    return summary


def _voc_payload(state: PostCallAgentState) -> dict:
    voc = dict(state.get("voc_analysis") or {})  # type: ignore[call-overload]
    analysis = state.get("analysis_result") or {}  # type: ignore[call-overload]
    if not voc and isinstance(analysis, dict):
        voc = dict(analysis.get("voc_analysis") or {})
    if not voc:
        return {}

    priority = state.get("priority_result") or {}  # type: ignore[call-overload]
    if not voc.get("priority_result") and priority:
        voc["priority_result"] = dict(priority)
    voc.setdefault("sentiment_result", {})
    voc.setdefault("intent_result", {})
    voc.setdefault("priority_result", {})
    return voc


def _failed_subagents(state: PostCallAgentState) -> list[str]:
    failed: list[str] = []
    for err in state.get("errors", []):  # type: ignore[call-overload]
        if isinstance(err, dict):
            node = err.get("node")
            if node:
                failed.append(str(node))
    return failed
