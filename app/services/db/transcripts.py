"""
DB transcript adapter.

calls / transcripts 테이블에서 종료된 통화 context를 조회한다.

── 현재 상태 ──────────────────────────────────────────────────────────────────
실제 DB ORM/테이블 구조가 확정되지 않아 skeleton 구현이다.
함수는 항상 None을 반환한다.
테스트에서 monkeypatch로 반환값을 주입해 동작을 검증한다.

── TODO: DB ORM 확정 후 채울 내용 ────────────────────────────────────────────
- DB 세션 팩토리(asyncpg / SQLAlchemy async) 연결
- calls 테이블에서 call_id 기준 metadata 조회
- transcripts 테이블에서 call_id 기준 row 조회 (role / text / timestamp)
- branch_stats: calls 테이블 컬럼 또는 별도 집계

── 반환 형식 ──────────────────────────────────────────────────────────────────
{
  "metadata": {
    "call_id": "...",
    "tenant_id": "...",
    "start_time": "ISO8601",
    "end_time": "ISO8601",
    "status": "completed"
  },
  "transcripts": [
    {"role": "customer", "text": "...", "timestamp": "ISO8601"},
    {"role": "agent",    "text": "...", "timestamp": "ISO8601"}
  ],
  "branch_stats": {"faq": int, "task": int, "escalation": int}
}

── 주의 ───────────────────────────────────────────────────────────────────────
- import 시점에 DB 연결을 만들지 않는다.
- sample transcript를 절대 반환하지 않는다.
- DB 조회 실패 시 예외를 전파하지 않고 logger.warning 후 None을 반환한다.
"""
from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


async def get_completed_call_context_from_db(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    """DB에서 종료된 통화 context를 조회한다.

    TODO: 실제 DB ORM 연결 후 아래 구조로 구현한다.

    # from app.db.session import get_async_session
    # from app.db.models import Call, Transcript
    # from sqlalchemy import select
    #
    # try:
    #     async with get_async_session() as session:
    #         call = await session.get(Call, call_id)
    #         if call is None:
    #             return None
    #         result = await session.execute(
    #             select(Transcript)
    #             .where(Transcript.call_id == call_id)
    #             .order_by(Transcript.created_at)
    #         )
    #         rows = result.scalars().all()
    #         return {
    #             "metadata": {
    #                 "call_id":   call_id,
    #                 "tenant_id": call.tenant_id,
    #                 "start_time": call.start_time.isoformat() if call.start_time else None,
    #                 "end_time":   call.end_time.isoformat()   if call.end_time   else None,
    #                 "status":    "completed",
    #             },
    #             "transcripts": [
    #                 {"role": r.role, "text": r.text,
    #                  "timestamp": r.created_at.isoformat() if r.created_at else None}
    #                 for r in rows
    #             ],
    #             "branch_stats": {
    #                 "faq":       call.faq_count       or 0,
    #                 "task":      call.task_count      or 0,
    #                 "escalation": call.escalation_count or 0,
    #             },
    #         }
    # except Exception as exc:
    #     logger.warning("DB 조회 실패 call_id=%s err=%s — None 반환", call_id, exc)
    #     return None
    """
    logger.debug(
        "DB adapter skeleton: call_id=%s tenant_id=%s — None 반환 (미구현)",
        call_id, tenant_id,
    )
    return None
