"""
POST-CALL API — 통화 후처리 결과 조회 및 수동 실행.

KDT-79 통합 시 app/main.py 에 아래 라인을 추가한다:
    from app.api.v1.post_call import router as post_call_router
    app.include_router(post_call_router, prefix="/post-call", tags=["post-call"])
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.agents.post_call.agent import PostCallAgent
from app.repositories import (
    get_action_logs_by_call_id,
    get_dashboard_payload,
    get_post_call_detail,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/{call_id}/actions")
async def get_call_actions(call_id: str):
    """call_id 에 해당하는 MCP action log list 를 반환한다."""
    logs = await get_action_logs_by_call_id(call_id)
    return {"call_id": call_id, "actions": logs}


@router.get("/{call_id}")
async def get_post_call(call_id: str):
    """통화 후처리 전체 결과를 반환한다.

    summary, voc_analysis, priority_result, action_plan,
    executed_actions, errors, partial_success 포함.
    저장된 결과가 없으면 404 를 반환한다.
    """
    payload = await get_dashboard_payload(call_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"post-call result not found: {call_id!r}",
        )
    detail = await get_post_call_detail(call_id)
    return {"call_id": call_id, **detail}


@router.post("/{call_id}/run")
async def run_post_call(
    call_id: str,
    trigger: str = Query(default="manual"),
    tenant_id: str = Query(default="default"),
):
    """수동 후처리 실행 API.

    trigger: manual(기본) | call_ended | escalation_immediate
    LLM 은 환경변수 POST_CALL_USE_REAL_LLM=true 가 아니면 mock 을 사용한다.
    """
    logger.info("run_post_call call_id=%s trigger=%s tenant_id=%s", call_id, trigger, tenant_id)
    try:
        agent = PostCallAgent()
        result = await agent.run(call_id=call_id, trigger=trigger, tenant_id=tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "ok": True,
        "result": {
            "call_id": call_id,
            "trigger": trigger,
            "partial_success": result.get("partial_success", False),  # type: ignore[call-overload]
            "errors": result.get("errors", []),                       # type: ignore[call-overload]
            "summary": result.get("summary"),                         # type: ignore[call-overload]
            "voc_analysis": result.get("voc_analysis"),               # type: ignore[call-overload]
            "priority_result": result.get("priority_result"),         # type: ignore[call-overload]
            "action_plan": result.get("action_plan"),                 # type: ignore[call-overload]
            "executed_actions": result.get("executed_actions", []),   # type: ignore[call-overload]
        },
    }
