"""
POST-CALL API — 통화 후처리 결과 조회 및 수동 실행.

등록 (app/main.py):
    app.include_router(post_call_router, prefix="/post-call", tags=["post-call"])
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.agents.post_call.completed_call_runner import (
    _CALL_CONTEXT_NOT_FOUND,
    run_post_call_for_completed_call,
)
from app.repositories import (
    get_action_logs_by_call_id,
    get_dashboard_payload,
    get_post_call_detail,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_VALID_TRIGGERS = frozenset({"call_ended", "manual", "escalation_immediate"})


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
    trigger: str = Query(default="call_ended"),
    tenant_id: str = Query(default="default"),
):
    """종료된 통화 데이터를 기반으로 후처리를 수동 실행한다.

    trigger: call_ended(기본) | manual | escalation_immediate
    - 통화 context가 없으면 404를 반환한다.
    - LLM은 POST_CALL_USE_REAL_LLM=true 가 아니면 mock을 사용한다.
    """
    if trigger not in _VALID_TRIGGERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown trigger: {trigger!r}. valid: {sorted(_VALID_TRIGGERS)}",
        )

    logger.info(
        "run_post_call call_id=%s trigger=%s tenant_id=%s",
        call_id, trigger, tenant_id,
    )

    result = await run_post_call_for_completed_call(
        call_id=call_id,
        tenant_id=tenant_id,
        trigger=trigger,
    )

    if not result["ok"] and result.get("error") == _CALL_CONTEXT_NOT_FOUND:
        raise HTTPException(
            status_code=404,
            detail=f"call context not found: {call_id!r}",
        )

    return result
