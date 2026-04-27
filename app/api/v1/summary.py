"""
SUMMARY API — 통화 요약 조회.

KDT-79 통합 시 app/main.py 에 아래 라인을 추가한다:
    from app.api.v1.summary import router as summary_router
    app.include_router(summary_router, prefix="/summary", tags=["summary"])
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.repositories import get_summary_by_call_id

router = APIRouter()


@router.get("/{call_id}")
async def get_summary(call_id: str):
    """call_id 에 해당하는 통화 요약을 반환한다.

    저장된 summary 가 없으면 404 를 반환한다.
    """
    record = await get_summary_by_call_id(call_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"summary not found: {call_id!r}",
        )
    return record
