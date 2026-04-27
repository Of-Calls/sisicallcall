"""
DASHBOARD API — 대시보드 집계·분포·큐 조회.

KDT-79 통합 시 app/main.py 에 아래 라인을 추가한다:
    from app.api.v1.dashboard import router as dashboard_router
    app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.repositories import (
    get_action_logs,
    get_dashboard_overview,
    get_emotion_distribution,
    get_priority_queue,
)

router = APIRouter()


@router.get("/stats")
async def get_stats(
    tenant_id: Optional[str] = Query(None, description="테넌트 필터"),
    started_from: Optional[str] = Query(None, description="시작 일시 (ISO 8601, 포함)"),
    started_to: Optional[str] = Query(None, description="종료 일시 (ISO 8601, 포함)"),
):
    """대시보드 통계 집계.

    total_calls, resolved_count, escalated_count, action_required_count,
    mcp_success_count, mcp_failed_count, partial_success_count 를 반환한다.
    """
    return await get_dashboard_overview(
        tenant_id=tenant_id,
        started_from=started_from,
        started_to=started_to,
    )


@router.get("/emotion-distribution")
async def get_emotion_dist(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[str] = Query(None),
    started_to: Optional[str] = Query(None),
):
    """고객 감정 분포 집계.

    positive, neutral, negative, angry 카운트를 반환한다.
    """
    return await get_emotion_distribution(
        tenant_id=tenant_id,
        started_from=started_from,
        started_to=started_to,
    )


@router.get("/priority-queue")
async def list_priority_queue(
    tenant_id: Optional[str] = Query(None),
):
    """우선순위 큐 조회.

    priority 가 high/critical 이거나 action_required=True 인 항목을 반환한다.
    critical → high 순서로 정렬된다.
    """
    return await get_priority_queue(tenant_id=tenant_id)


@router.get("/action-logs")
async def list_action_logs(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[str] = Query(None),
    started_to: Optional[str] = Query(None),
):
    """MCP action log 전체 조회.

    tenant_id, started_from(created_at ≥), started_to(created_at ≤) 기준 필터 지원.
    """
    return await get_action_logs(
        tenant_id=tenant_id,
        started_from=started_from,
        started_to=started_to,
    )
